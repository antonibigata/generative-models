import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter

import pytorch_lightning as pl
import torch
import wandb
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only

from sgm.util import instantiate_from_config


MULTINODE_HACKS = True


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {param: argspec[param].default for param in argspec if argspec[param] != Parameter.empty}
    return default_args


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default="stablediffusion",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path",
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default=None,
        help="Startuptime from distributed script",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,  # TODO: later default to True
        help="log to wandb",
    )
    parser.add_argument(
        "--no_base_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,  # TODO: later default to True
        help="log to wandb",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str):
    print(f"setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = save_dir
    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.projectname,
            # config=hyperparameter,
            settings=wandb.Settings(code_dir="./sgm"),
            # group=group_name,
            name=name_str,
        )


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    torch.set_float32_matmul_precision("medium")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(opt.base[0])[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs") + 1 :
                    ]  # cut away the first one (we assert all configs are in "configs")
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    # move before model init, in case a torch.compile(...) is called somewhere
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # default to gpu
    trainer_config["accelerator"] = "gpu"
    #
    standard_args = default_trainer_args()
    for k in standard_args:
        if getattr(opt, k) != standard_args[k]:
            trainer_config[k] = getattr(opt, k)

    ckpt_resume_path = opt.resume_from_checkpoint

    if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["devices"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                # "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
                "project": opt.projectname,
                "log_model": False,
                # "dir": logdir,
            },
        },
        "csv": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",  # hack for sbord fanatics
                "save_dir": logdir,
            },
        },
    }
    default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "csv"]
    if opt.wandb:
        # TODO change once leaving "swiffer" config directory
        try:
            group_name = nowname.split(now)[-1].split("-")[1]
        except:
            group_name = nowname
        default_logger_cfg["params"]["group"] = group_name
        init_wandb(
            os.path.join(os.getcwd(), logdir),
            opt=opt,
            group_name=group_name,
            config=config,
            name_str=nowname,
        )
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        
    # if lightning_config.get("strategy") == "horovod":
    #     hvd.init()
    #     if hvd.rank() == 0:
    #         trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    #     else:
    #         trainer_kwargs["logger"] = None
    # else:
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    if opt.wandb:
        trainer_kwargs["logger"].log_hyperparams(config)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        },
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
    # default to ddp if not further specified
    # default_strategy_config = {"target": "pytorch_lightning.strategies.DDPStrategy"}
    # default_strategy_config = {}

    # if "strategy" in lightning_config:
    #     strategy_cfg = OmegaConf.create()
    #     default_strategy_config["strategy"] = lightning_config.strategy
    # else:
    #     strategy_cfg = OmegaConf.create()
    #     default_strategy_config["params"] = {
    #         "find_unused_parameters": False,
    #         # "static_graph": True,
    #         # "ddp_comm_hook": default.fp16_compress_hook  # TODO: experiment with this, also for DDPSharded
    #     }
    # strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
    # print(f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ ")
    # trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg) if "target" in strategy_cfg else strategy_cfg
    trainer_kwargs["strategy"] = lightning_config.strategy

    # add callback which sets up log directory
    default_callbacks_cfg = {
        # "setup_callback": {
        #     "target": "sgm.callbacks.setup_callback.SetupCallback",
        #     "params": {
        #         "resume": opt.resume,
        #         "now": now,
        #         "logdir": logdir,
        #         "ckptdir": ckptdir,
        #         "cfgdir": cfgdir,
        #         "config": config,
        #         "lightning_config": lightning_config,
        #         "debug": opt.debug,
        #         "ckpt_name": melk_ckpt_name,
        #     },
        # },
        # "video_logger": {
        #     "target": "sgm.callbacks.video_logger.VideoLogger",
        #     "params": {"batch_frequency": 1000, "max_videos": 4, "clamp": True},
        # },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print("Caution: Saving checkpoints every n train steps without deleting. This might require some free space.")
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    if not "plugins" in trainer_kwargs:
        trainer_kwargs["plugins"] = list()

    # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
    trainer_opt = vars(trainer_opt)
    trainer_kwargs = {key: val for key, val in trainer_kwargs.items() if key not in trainer_opt}
    trainer = Trainer(**trainer_opt, **trainer_kwargs)

    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    # data.setup()
    print("#### Data #####")
    try:
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    except:
        print("datasets not yet initialized.")

    # configure learning rate
    if "batch_size" in config.data.params:
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    else:
        bs, base_lr = (
            config.data.params.train.loader.batch_size,
            config.model.base_learning_rate,
        )
    if not cpu:
        ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if melk_ckpt_name is None:
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        if not opt.debug:
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        else:
            trainer.fit(model, data, ckpt_path=ckpt_resume_path)
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
    # except RuntimeError as err:
    #     if MULTINODE_HACKS:
    #         import datetime
    #         import os
    #         import socket

    #         import requests

    #         device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    #         hostname = socket.gethostname()
    #         ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    #         resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
    #         print(
    #             f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
    #             flush=True,
    #         )
    #     raise err
    # except Exception as err:
    #     print("Exception: ", err)
    #     if opt.debug and trainer.global_rank == 0:
    #         try:
    #             import pudb as debugger
    #         except ImportError:
    #             import pdb as debugger
    #         debugger.post_mortem()
    #     raise
    # finally:
    #     # move newly created debug project to debug_runs
    #     if opt.debug and not opt.resume and trainer.global_rank == 0:
    #         dst, name = os.path.split(logdir)
    #         dst = os.path.join(dst, "debug_runs", name)
    #         os.makedirs(os.path.split(dst)[0], exist_ok=True)
    #         os.rename(logdir, dst)

    #     if opt.wandb:
    #         wandb.finish()
    #     # if trainer.global_rank == 0:
    #     #    print(trainer.profiler.summary())
