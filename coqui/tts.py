import os 
import hashlib
import wave
from piper import PiperVoice

MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "model/en_US-lessac-medium.onnx")
OUTPUT_DIR = "output_audio"

_voice = None 

def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(MODEL_PATH)
        return _voice

def speak(text: str,out_dir: str = OUTPUT_DIR) -> str:

    if not text.strip():
        return""

    os.makedirs(out_dir, exist_ok = True)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(out_dir, f"{text_hash}.wav")

    if os.path.exists(out_path):
        return out_path

    voice = _get_voice()
    with wave.open(out_path, "wb")as wav_file:
        voice.synthesize(text, wav_file)

        return out_path