import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime
import time
import threading


def record_audio_to_wav(duration=10, sample_rate=44100, channels=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"

    print(f"[AudioTool] Recording for {duration} seconds...")

    # Add a countdown for better user feedback
    def countdown_timer():
        for i in range(duration, 0, -1):
            print(f"[AudioTool] {i} seconds remaining...")
            time.sleep(1)
        print("[AudioTool] Recording finished!")

    # Start countdown in a separate thread
    countdown_thread = threading.Thread(target=countdown_timer)
    countdown_thread.daemon = True
    countdown_thread.start()

    # Record audio
    start_time = time.time()
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait for recording to complete
    actual_duration = time.time() - start_time

    print(f"[AudioTool] Actual recording duration: {actual_duration:.2f} seconds")

    # Convert to 16-bit integer format and save
    data_int16 = np.int16(recording * 32767)
    write(filename, sample_rate, data_int16)
    print(f"[AudioTool] Saved to {filename}")
    return filename


# Alternative version with more precise timing
def record_audio_to_wav_precise(duration=10, sample_rate=44100, channels=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"

    print(f"[AudioTool] Starting {duration}-second recording...")
    print("[AudioTool] 3... 2... 1... GO!")

    # More precise recording with exact sample count
    exact_samples = int(duration * sample_rate)

    start_time = time.time()
    recording = sd.rec(exact_samples, samplerate=sample_rate, channels=channels, dtype='float32')

    # Real-time progress indicator
    def show_progress():
        while not recording.flags.writeable:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            if remaining > 0:
                print(f"\r[AudioTool] Recording... {remaining:.1f}s left", end='', flush=True)
                time.sleep(0.1)
            else:
                break
        print()  # New line after progress

    progress_thread = threading.Thread(target=show_progress)
    progress_thread.daemon = True
    progress_thread.start()

    sd.wait()  # Wait for recording to complete
    actual_duration = time.time() - start_time

    print(f"[AudioTool] Recording complete! Actual duration: {actual_duration:.2f}s")

    # Save the recording
    data_int16 = np.int16(recording * 32767)
    write(filename, sample_rate, data_int16)
    print(f"[AudioTool] Saved to {filename}")
    return filename


# Version with configurable feedback
def record_audio_to_wav_enhanced(duration=10, sample_rate=44100, channels=1, show_countdown=True, show_progress=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"

    if show_countdown:
        print(f"[AudioTool] Starting {duration}-second recording in...")
        for i in range(3, 0, -1):
            print(f"[AudioTool] {i}...")
            time.sleep(1)
        print("[AudioTool] Recording NOW!")
    else:
        print(f"[AudioTool] Recording for {duration} seconds...")

    start_time = time.time()
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')

    if show_progress:
        # Show progress every second
        for i in range(duration):
            time.sleep(1)
            remaining = duration - (i + 1)
            if remaining > 0:
                print(f"[AudioTool] {remaining} seconds remaining...")

    sd.wait()
    actual_duration = time.time() - start_time

    print(f"[AudioTool] Recording complete! (Actual: {actual_duration:.2f}s)")

    data_int16 = np.int16(recording * 32767)
    write(filename, sample_rate, data_int16)
    print(f"[AudioTool] Saved to {filename}")
    return filename