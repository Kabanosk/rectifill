import torch
import torchaudio
from loguru import logger
from transformers import T5Tokenizer


class ForcedAligner:
    """
    Handles Forced Alignment using torchaudio's Wav2Vec2 and forced_align functional.
    Calculates the exact duration (in Mel frames) for each word and inserts <SIL> tokens.
    """

    def __init__(self, device: str | torch.device):
        """
        Initializes the ForcedAligner with a pre-trained Wav2Vec2 model.

        :param device: The device to load the model on (e.g., 'cuda' or 'cpu').
        :type device: str | torch.device
        """
        self.device = device
        logger.info("Loading Wav2Vec2 ASR model for Forced Alignment...")

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(device)
        self.model.eval()
        self.labels = self.bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}

        self.blank_id = self.dictionary["-"]  # CTC Blank token

    def get_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Gets CTC emissions from the Wav2Vec2 model.

        :param waveform: The input audio waveform.
            Shape: ``[Batch, Channels, Time]`` or ``[Channels, Time]``.
        :type waveform: torch.Tensor
        :return: The log-softmax CTC emissions.
            Shape: ``[Batch, Frames, Num_Classes]``.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            emissions, _ = self.model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
        return emissions

    def get_transcript_targets(self, transcript: str) -> list[int]:
        """
        Converts the text transcript into dictionary IDs for Wav2Vec2 alignment.

        :param transcript: The ground-truth text transcript.
        :return: A list of integer IDs corresponding to the transcript characters.
        """
        transcript = transcript.replace(" ", "|").upper()
        return [self.dictionary[c] for c in transcript if c in self.dictionary]

    @torch.no_grad()
    def compute_durations(self, waveform: torch.Tensor, transcript: str, sample_rate: int, hop_length: int) -> list[dict]:
        """
        Computes word-level durations in Mel frames using Viterbi forced alignment.

        :param waveform: The raw audio waveform tensor.
            Shape: ``[1, Time]``.
        :param transcript: The text transcript of the audio.
        :param sample_rate: The sample rate of the waveform (usually 16000).
        :param hop_length: The hop length used for the Mel-spectrogram (e.g., 512).
        :return: A list of dictionaries containing 'word' (str) and its duration in 'mel_frames' (int).
        """
        emissions = self.get_emissions(waveform)
        targets = self.get_transcript_targets(transcript)
        targets_tensor = torch.tensor(targets, dtype=torch.int32, device=self.device).unsqueeze(0)

        alignment, scores = torchaudio.functional.forced_align(emissions, targets_tensor, blank=self.blank_id)
        token_spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
        w2v2_hop_length = 320

        aligned_sequence = []

        words = transcript.split()
        word_idx = 0
        current_word_frames = 0

        for span in token_spans:
            span_start_sec = (span.start * w2v2_hop_length) / sample_rate
            span_end_sec = (span.end * w2v2_hop_length) / sample_rate

            start_mel_frame = int((span_start_sec * sample_rate) / hop_length)
            end_mel_frame = int((span_end_sec * sample_rate) / hop_length)
            span_mel_duration = end_mel_frame - start_mel_frame

            if span.token == self.blank_id:
                if span_mel_duration > 0:
                    aligned_sequence.append({"word": "<SIL>", "mel_frames": span_mel_duration})
            else:
                current_word_frames += span_mel_duration
                if span.token == self.dictionary.get("|", -1) or span == token_spans[-1]:
                    if word_idx < len(words):
                        aligned_sequence.append({"word": words[word_idx], "mel_frames": current_word_frames})
                        word_idx += 1
                    current_word_frames = 0

        return aligned_sequence


def map_to_t5_tokens(aligned_sequence: list[dict], tokenizer: T5Tokenizer) -> tuple[torch.Tensor, str]:
    """
    Maps word-level mel-frame durations to T5 sub-token durations.

    :param aligned_sequence: List of dicts containing 'word' and 'mel_frames'.
    :param tokenizer: The T5 tokenizer instance used for text encoding.
    :return: A tuple containing:
        - duration_tensor: A tensor of integer durations for each sub-token.
          Shape: ``[Seq_Len]``.
        - final_transcript: A string of the transcript with explicitly added <SIL> tokens.
    """
    durations = []
    final_text_parts = []

    for item in aligned_sequence:
        word = item["word"]
        total_frames = item["mel_frames"]

        if word == "<SIL>":
            durations.append(total_frames)
            final_text_parts.append("<SIL>")
            continue

        tokens = tokenizer.encode(word, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_tokens == 0:
            durations.append(total_frames)
            final_text_parts.append("<SIL>")
            continue

        frames_per_token = total_frames // num_tokens
        remainder = total_frames % num_tokens

        word_durations = [frames_per_token] * num_tokens
        word_durations[-1] += remainder  # Add remainder to the last token

        durations.extend(word_durations)
        final_text_parts.append(word)

    durations.append(0)
    final_transcript = " ".join(final_text_parts)

    return torch.tensor(durations, dtype=torch.long), final_transcript


# --- PHONEME DICTIONARY (ARPAbet) ---
PHONEME_VOCAB = {
    "<PAD>": 0, "<SIL>": 1,
    "AA0": 2, "AA1": 3, "AA2": 4, "AE0": 5, "AE1": 6, "AE2": 7, "AH0": 8, "AH1": 9, "AH2": 10,
    "AO0": 11, "AO1": 12, "AO2": 13, "AW0": 14, "AW1": 15, "AW2": 16, "AY0": 17, "AY1": 18, "AY2": 19,
    "B": 20, "CH": 21, "D": 22, "DH": 23, "EH0": 24, "EH1": 25, "EH2": 26, "ER0": 27, "ER1": 28, "ER2": 29,
    "EY0": 30, "EY1": 31, "EY2": 32, "F": 33, "G": 34, "HH": 35, "IH0": 36, "IH1": 37, "IH2": 38,
    "IY0": 39, "IY1": 40, "IY2": 41, "JH": 42, "K": 43, "L": 44, "M": 45, "N": 46, "NG": 47,
    "OW0": 48, "OW1": 49, "OW2": 50, "OY0": 51, "OY1": 52, "OY2": 53, "P": 54, "R": 55, "S": 56,
    "SH": 57, "T": 58, "TH": 59, "UH0": 60, "UH1": 61, "UH2": 62, "UW": 63, "UW0": 64, "UW1": 65,
    "UW2": 66, "V": 67, "W": 68, "Y": 69, "Z": 70, "ZH": 71,
    "<MASK>": 72
}


def map_to_phoneme_tokens(aligned_sequence: list[dict], g2p_model) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Simplified mapping: divides word duration equally among its phonemes.
    No complex alignment needed for Cross-Attention!

    :param aligned_sequence: List of dictionaries containing 'word' (str) and its duration in 'mel_frames' (int). Generated by the ForcedAligner.
    :param g2p_model: An initialized Grapheme-to-Phoneme model (e.g., g2p_en) used to convert words into phoneme lists.
    :return: A tuple containing:
        - duration_tensor: A tensor of integer durations (in mel frames) for each phoneme. Shape: ``[Phoneme_Seq_Len]``.
        - phoneme_ids: A tensor of integer IDs representing the phonemes based on PHONEME_VOCAB. Shape: ``[Phoneme_Seq_Len]``.
        - final_transcript: A string of the transcript with explicitly added <SIL> tokens.
    """
    durations = []
    phoneme_ids = []
    final_text_parts = []

    for item in aligned_sequence:
        word = item["word"]
        total_frames = item["mel_frames"]

        if word == "<SIL>":
            durations.append(total_frames)
            phoneme_ids.append(PHONEME_VOCAB["<SIL>"])
            final_text_parts.append("<SIL>")
            continue

        ph_list = g2p_model(word.lower())
        ph_list = [p for p in ph_list if p in PHONEME_VOCAB]

        num_phonemes = len(ph_list)
        if num_phonemes == 0:
            durations.append(total_frames)
            phoneme_ids.append(PHONEME_VOCAB["<SIL>"])
            final_text_parts.append("<SIL>")
            continue

        frames_per_ph = total_frames // num_phonemes
        remainder = total_frames % num_phonemes

        word_durations = [frames_per_ph] * num_phonemes
        word_durations[-1] += remainder

        durations.extend(word_durations)
        phoneme_ids.extend([PHONEME_VOCAB[p] for p in ph_list])
        final_text_parts.append(word)

    durations.append(0)
    phoneme_ids.append(PHONEME_VOCAB["<SIL>"])

    final_transcript = " ".join(final_text_parts)

    return (
        torch.tensor(durations, dtype=torch.long),
        torch.tensor(phoneme_ids, dtype=torch.long),
        final_transcript
    )
