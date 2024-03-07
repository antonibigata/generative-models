import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import default, disabled_train, get_obj_from_str, instantiate_from_config, log_txt_as_img

from peft import LoraModel, LoraConfig
from ..modules.diffusionmodules.adapters.lora import get_module_names
from ..modules.diffusionmodules.adapters.lora_v2 import inject_trainable_lora


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str, Dict, ListConfig, OmegaConf] = None,
        ckpt_path: Union[None, str] = None,
        remove_keys_from_weights: Union[None, List, Tuple] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        use_lora: Optional[bool] = True,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()

        # self.automatic_optimization = False

        self.log_keys = log_keys
        self.no_log_keys = no_log_keys
        self.input_key = input_key
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.AdamW"})
        model = instantiate_from_config(network_config)
        if isinstance(network_wrapper, str) or network_wrapper is None:
            self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
                model, compile_model=compile_model
            )
        else:
            target = network_wrapper["target"]
            params = network_wrapper.get("params", dict())
            self.model = get_obj_from_str(target)(model, compile_model=compile_model, **params)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.is_guided = True
        if sampler_config is not None and sampler_config["params"].get("guider_config") is None:
            self.is_guided = False
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, remove_keys_from_weights=remove_keys_from_weights)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        if use_lora:
            # for p in self.model.parameters():
            #     p.requires_grad = False
            inject_trainable_lora(
                self.model,
                **lora_config,
            )
            # filters = [".transformer_blocks"]
            # module_names = get_module_names(self.model, filters=filters, all_modules_in_filter=True)
            # lora_config = LoraConfig(
            #     inference_mode=False,
            #     r=16,
            #     lora_alpha=32,
            #     lora_dropout=0.1,
            #     target_modules=module_names,
            # )
            # self.model = LoraModel(self.model, lora_config, "bite")

    def init_from_ckpt(
        self,
        path: str,
        remove_keys_from_weights: bool = True,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        if remove_keys_from_weights is not None:
            for k in list(sd.keys()):
                for remove_key in remove_keys_from_weights:
                    if remove_key in k:
                        del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model
        if self.input_key == "latents":
            # Remove encoder to save memory
            self.first_stage_model.encoder = None
            torch.cuda.empty_cache()

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        is_video = False
        if len(z.shape) == 5:
            is_video = True
            T = z.shape[2]
            z = rearrange(z, "b c t h w -> (b t) c h w")

        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        if is_video:
            out = rearrange(out, "(b t) c h w -> b c t h w", t=T)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        is_video = False
        if len(x.shape) == 5:
            is_video = True
            T = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        if is_video:
            z = rearrange(z, "(b t) c h w -> b c t h w", t=T)
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        if self.input_key != "latents":
            x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        # debugging_message = "Training step"
        # print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # debugging_message = "Training step - log"
        # print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # # to prevent other processes from moving forward until all processes are in sync
        # self.trainer.strategy.barrier()

        return loss

    # def on_train_epoch_start(self, *args, **kwargs):
    #     print(f"RANK - {self.trainer.global_rank}: on_train_epoch_start")

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     print(f"RANK - {self.trainer.global_rank}: on_before_batch_transfer - {dataloader_idx}")
    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     print(f"RANK - {self.trainer.global_rank}: on_after_batch_transfer - {dataloader_idx}")
    #     return batch

    def on_train_batch_end(self, *args, **kwargs):
        # print(f"RANK - {self.trainer.global_rank}: on_train_batch_end")
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(self.model, input, sigma, c, **kwargs)
        samples = self.sampler(denoiser, randn, cond, uc=uc)

        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[-2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                if embedder.input_key in self.no_log_keys:
                    continue
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:  # already an image
                        xc = x
                    elif x.dim() == 5:
                        xc = torch.cat([x[:, :, i] for i in range(x.shape[2])], dim=-1)
                    else:
                        print(x.shape, embedder.input_key)
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if self.input_key != "latents":
            log["inputs"] = x
            z = self.encode_first_stage(x)
        else:
            z = x
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log

    @torch.no_grad()
    def log_videos(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()
        batch_uc = {}

        x = self.get_input(batch)
        num_frames = x.shape[2]  # assuming bcthw format

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        # for k in ["crossattn", "concat"]:
        #     uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
        #     uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
        #     c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
        #     c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]

        if self.input_key != "latents":
            log["inputs"] = x
            z = self.encode_first_stage(x)
        else:
            z = x
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            n = 2 if self.is_guided else 1
            sampling_kwargs["image_only_indicator"] = torch.zeros(n, num_frames).to(self.device)
            sampling_kwargs["num_video_frames"] = batch["num_video_frames"]

            with self.ema_scope("Plotting"):
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log
