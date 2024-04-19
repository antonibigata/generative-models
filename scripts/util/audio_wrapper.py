from src.models.components.audio.Whisper import Whisper
from src.models.components.audio.WavLM import WavLM_wrapper
import torch
import torch.nn as nn
from src.utils.utils import default


class AudioWrapper(nn.Module):
    def __init__(self, model_type="whisper", model_size="large-v3", fps=25) -> None:
        super().__init__()

        if model_type == "whisper":
            self.model = Whisper(model_size, fps, "None")
            self.encode_audio = self.whisper_encoding
        elif model_type == "wavlm":
            self.model = WavLM_wrapper(model_size, feed_as_frames=False, merge_type="None")
            self.encode_audio = self.wavlm_encoding

    def whisper_encoding(self, audio_frames, chunks=None):
        chunks = default(chunks, 750)
        # Get audio embeddings
        audio_embeddings = []
        for chunk in torch.split(
            audio_frames, chunks, dim=0
        ):  # 750 is the max size of the audio chunks that can be processed by the model (= 30 seconds)
            audio_embeddings.append(self.model(chunk.unsqueeze(0).cuda()))
        audio_embeddings = torch.cat(audio_embeddings, dim=1)
        # audio_embeddings = model(audio_frames.unsqueeze(0).cuda())

        # Save audio embeddings
        assert (
            audio_embeddings.shape[1] == audio_frames.shape[0]
        ), f"{audio_embeddings.shape[1]} != {audio_frames.shape[0]}"

        return audio_embeddings

    def wavlm_encoding(self, audio_frames):
        audio_embeddings = self.model(audio_frames.unsqueeze(0))

        assert (
            audio_embeddings.shape[1] == audio_frames.shape[0]
        ), f"{audio_embeddings.shape[1]} != {audio_frames.shape[0]}"
        return audio_embeddings
