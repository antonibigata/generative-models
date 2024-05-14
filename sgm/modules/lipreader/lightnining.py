import torch
from sgm.modules.lipreader.datamodule.transforms import TextTransform

# for testing
from sgm.modules.lipreader.espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from sgm.modules.lipreader.espnet.nets.batch_beam_search import BatchBeamSearch
from sgm.modules.lipreader.espnet.nets.lm_interface import dynamic_import_lm
from sgm.modules.lipreader.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from sgm.modules.lipreader.espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


class ModelModule(LightningModule):
    def __init__(self):
        super().__init__()

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list))

    def forward(self, sample, t=None, extract_position=None):
        if extract_position == "frontend":
            return self.model.encoder(sample.unsqueeze(0).to(self.device), None, t=t, extract_resnet_feats=True)
        else:
            enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None, t=t)
        if extract_position == "conformer":
            return enc_feat
        enc_feat = enc_feat.squeeze(0)
        # initialise beam search decdoer
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        return predicted


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
