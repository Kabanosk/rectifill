import csv
import re
from pathlib import Path

import nltk
import torch
import torchaudio
from g2p_en import G2p
from loguru import logger
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from src.config.config import DataConfig, MelConfig, TextConfig
from src.data.alignment import ForcedAligner, map_to_phoneme_tokens, map_to_t5_tokens
from src.data.utils import get_mel_transform, load_wav


def load_transcripts_vctk(source_dir: Path) -> dict[str, str]:
    """
    Scans the VCTK dataset's 'txt' directory to load transcripts.
    VCTK stores text files in a separate 'txt' folder with the format pXXX_YYY.txt
    """
    transcripts = {}
    txt_dir = source_dir / "txt"
    logger.info(f"Searching for VCTK transcripts in: {txt_dir}...")

    if not txt_dir.exists():
        logger.error(f"Text directory not found at {txt_dir}. Dataset might be missing or incomplete.")
        return transcripts

    for txt_file in txt_dir.rglob("*.txt"):
        file_id = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            transcripts[file_id] = text

    logger.info(f"Found {len(transcripts)} transcript entries.")
    return transcripts


def process_audio_and_text_vctk(source_dir: Path, output_dir: Path, config: DataConfig):
    """
    Processes VCTK audio files, generates log-mel spectrograms and text embeddings/phonemes.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transcripts = load_transcripts_vctk(source_dir)

    # --- Initialization of models based on config ---
    if config.text_params.context_type == "t5":
        logger.info(f"Loading text model ({config.text_params.model_name}) on {device}...")
        tokenizer = T5Tokenizer.from_pretrained(config.text_params.model_name)
        logger.info("Adding <SIL> special token to T5 tokenizer...")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<SIL>']})
        text_encoder = T5EncoderModel.from_pretrained(config.text_params.model_name).to(device)
        text_encoder.resize_token_embeddings(len(tokenizer))
        text_encoder.eval()

    elif config.text_params.context_type == "phonemes":
        logger.info("Initializing G2P model for phonemes...")
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            logger.warning("NLTK data missing. Downloading automatically...")
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('cmudict', quiet=True)
        g2p_model = G2p()

    else:
        raise ValueError(f"Unknown context_type: {config.text_params.context_type}")

    logger.info("Initializing Mel-Spectrogram transform from config...")
    mel_transform = get_mel_transform(
        sample_rate=config.mel_params.sample_rate,
        n_mels=config.mel_params.n_mels
    )

    logger.info("Initializing Forced Aligner...")
    aligner = ForcedAligner(device=device)

    def clean_text_for_wav2vec2(text: str) -> str:
        """
        Cleans text to strictly match Wav2Vec2 dictionary:
        A-Z, spaces, and apostrophes.
        """
        text = text.upper()
        text = re.sub(r"[^A-Z\s']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    wav_files = list(source_dir.rglob("*.flac")) + list(source_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} audio files. Processing...")

    metadata = []

    for wav_path in tqdm(wav_files, desc="Processing VCTK"):
        parts = wav_path.stem.split('_')
        if len(parts) >= 2:
            file_id = f"{parts[0]}_{parts[1]}"
        else:
            file_id = wav_path.stem

        if file_id not in transcripts:
            continue

        text_content = transcripts[file_id]
        clean_text = clean_text_for_wav2vec2(text_content)

        if not clean_text:
            continue

        save_id = wav_path.stem

        try:
            # --- Audio Processing ---
            waveform = load_wav(wav_path, config.mel_params.sample_rate)

            with torch.no_grad():
                mel_spec = mel_transform(waveform).cpu()

            waveform = waveform.to(device)
            output_mel_path = output_dir / f"{save_id}_mel.pt"
            torch.save(mel_spec, output_mel_path)

            # --- Alignment ---
            aligned_seq = aligner.compute_durations(
                waveform=waveform,
                transcript=clean_text,
                sample_rate=config.mel_params.sample_rate,
                hop_length=config.mel_params.hop_length
            )

            # --- Context Processing ---
            if config.text_params.context_type == "t5":
                durations, text_with_silence = map_to_t5_tokens(aligned_seq, tokenizer)

                output_dur_path = output_dir / f"{save_id}_dur.pt"
                torch.save(durations, output_dur_path)

                inputs = tokenizer(text_with_silence, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = text_encoder(**inputs)
                    embedding = outputs.last_hidden_state[0].cpu()

                if embedding.shape[0] != durations.shape[0]:
                    logger.warning(
                        f"Shape mismatch for {save_id}: Emb {embedding.shape[0]} vs Dur {durations.shape[0]}. Skipping.")
                    continue

                output_emb_path = output_dir / f"{save_id}_emb.pt"
                torch.save(embedding, output_emb_path)

                output_txt_path = output_dir / f"{save_id}.txt"
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(text_with_silence)

                metadata.append([f"{save_id}_mel.pt", f"{save_id}_emb.pt", f"{save_id}_dur.pt", text_with_silence])

            elif config.text_params.context_type == "phonemes":
                durations, phoneme_ids, text_with_silence = map_to_phoneme_tokens(aligned_seq, g2p_model)

                if phoneme_ids.shape[0] != durations.shape[0]:
                    logger.warning(f"Shape mismatch for {save_id}. Skipping.")
                    continue

                torch.save(durations, output_dir / f"{save_id}_dur.pt")
                torch.save(phoneme_ids, output_dir / f"{save_id}_phonemes.pt")

                output_txt_path = output_dir / f"{save_id}.txt"
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(text_with_silence)

                metadata.append([f"{save_id}_mel.pt", f"{save_id}_phonemes.pt", f"{save_id}_dur.pt", text_with_silence])

        except Exception as e:
            logger.error(f"Error processing file {wav_path}: {e}")

    # --- Saving METADATA.CSV ---
    csv_path = output_dir / "metadata.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        if not file_exists:
            writer.writerow(["mel_file", "context_file", "duration_file", "transcription"])
        writer.writerows(metadata)

    logger.success(f"Done! Saved to: {output_dir}. Total samples: {len(metadata)}")


def download_and_prepare_vctk(config: DataConfig, output_subdir: str = "vctk"):
    """
    Downloads and extracts the VCTK dataset, then triggers the processing script.
    """
    root_path = Path(config.data_path)
    raw_path = root_path / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path = root_path / "processed" / output_subdir

    logger.info("--- Starting work on VCTK dataset ---")

    try:
        logger.info(f"Downloading/Verifying VCTK to {raw_path}...")
        _ = torchaudio.datasets.VCTK_092(
            root=str(raw_path),
            download=True
        )
        logger.info("Download/Verification complete.")

        extracted_path = raw_path / "VCTK-Corpus-0.92"
        if not extracted_path.exists():
            extracted_path = raw_path / "VCTK-Corpus"

    except Exception as e:
        logger.error(f"Download/Extraction error: {e}")
        return

    process_audio_and_text_vctk(extracted_path, processed_path, config)


if __name__ == "__main__":
    main_config = DataConfig(data_path="data", mel_params=MelConfig(), text_params=TextConfig())

    logger.info("Starting processing: VCTK DATASET")
    download_and_prepare_vctk(main_config, output_subdir="vctk")
