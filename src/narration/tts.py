"""
Piper TTS wrapper — drop-in replacement for coqui/tts.py's speak().
"""

import os
import hashlib
import wave
from piper import PiperVoice

MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "models/en_US-lessac-medium.onnx")
OUTPUT_DIR = "output_audio"

_voice = None


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(MODEL_PATH)
    return _voice


def speak(text: str, out_dir: str = OUTPUT_DIR) -> str:
    if not text.strip():
        return ""

    os.makedirs(out_dir, exist_ok=True)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(out_dir, f"{text_hash}.wav")

    if os.path.exists(out_path):
        return out_path  # cache hit

    voice = _get_voice()

    with wave.open(out_path, "wb") as wav_file:
        # NOTE: voice.synthesize() in current piper-tts is a generator
        # (for streaming raw chunks) — calling it and passing wav_file
        # does nothing but silently create an empty header, since the
        # generator body never runs unless iterated. synthesize_wav()
        # is the correct method for writing directly to an open WAV file.
        voice.synthesize_wav(text, wav_file)

    return out_path