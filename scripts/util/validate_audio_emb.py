from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from einops import rearrange
import argparse


def main(model, preprocessor, audio_emb):
    # Get logits
    logits = model.lm_head(audio_emb).unsqueeze(0)
    predicted_ids = torch.argmax(logits, dim=-1).squeeze()

    # transcribe
    transcription = processor.decode(predicted_ids)
    print("Transcription:", transcription)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate audio embeddings")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    args = parser.parse_args()

    # Load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h", cache_dir="/vol/bitbucket/abigata/.cache"
    )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/vol/bitbucket/abigata/.cache")

    # Load audio embeddings
    audio_emb = torch.load(args.audio_file)
    audio_emb = rearrange(audio_emb, "f d c -> (f d) c")

    main(model, processor, audio_emb)
