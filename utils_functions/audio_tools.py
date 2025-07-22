import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime

def record_audio_to_wav(duration=10, sample_rate=44100, channels=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"

    print(f"[AudioTool] Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    print("[AudioTool] Recording complete.")

    data_int16 = np.int16(recording * 32767)
    write(filename, sample_rate, data_int16)
    print(f"[AudioTool] Saved to {filename}")
    return filename