import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import time
import sys

# Load Whisper model
model = whisper.load_model("medium")

# Audio config
SAMPLE_RATE = 16000
CHUNK_DURATION = 1  # seconds per chunk
SILENCE_THRESHOLD = 0.01  # adjust this if too sensitive
SILENCE_TIMEOUT = 3  # seconds

def is_silent(audio_chunk, threshold=SILENCE_THRESHOLD):
    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
    energy = np.sqrt(np.mean(audio_chunk**2))
    return energy < threshold

def record_until_silence():
    print("🎙️ Start speaking... (will auto stop after 15s of silence)")
    silent_chunks = 0
    recorded_audio = []

    while True:
        audio_chunk = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        recorded_audio.append(audio_chunk)

        if is_silent(audio_chunk):
            silent_chunks += 1
            print(f"🤫 Silence {silent_chunks}s")
        else:
            silent_chunks = 0
            print("🎤 Voice detected")

        if silent_chunks >= SILENCE_TIMEOUT:
            print("⏱️ No voice for 15 seconds. Stopping execution.")
            return np.concatenate(recorded_audio, axis=0), True

        # Just a safety limit if you want to add one (optional)
        # if len(recorded_audio) > 600:  # 10 minutes max
        #     break

print("🔊 Say something (or say 'stop listening')...\n")

try:
    audio_data, silence_triggered = record_until_silence()
    if silence_triggered:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            normalized_audio = audio_data.astype(np.float32) / 32768.0
            wavfile.write(f.name, SAMPLE_RATE, (normalized_audio * 32768).astype(np.int16))

            # Transcribe
            result = model.transcribe(f.name)
            transcript = result["text"].strip()
            print("📝", transcript)
            
            if "stop listening" in transcript.lower():
                print("👋 'Stop listening' heard. Goodbye!")
            else:
                print("👋 Ending due to silence.")

    # Stop script entirely
    sys.exit(0)

except KeyboardInterrupt:
    print("\n🛑 Manually stopped")
    sys.exit(0)

except Exception as e:
    print(f"⚠️ Error: {e}")
    sys.exit(1)
