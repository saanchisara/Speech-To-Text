import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile

# Load the Whisper model
model = whisper.load_model("medium")  # Try "small" or "medium" for better accuracy

# Audio config
SAMPLE_RATE = 16000
DURATION = 10 # 10-second chunks

def record_audio(duration, fs):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio

print("🔊 Say 'stop listening' to exit.\n")

while True:
    try:
        # Record audio
        audio_data = record_audio(DURATION, SAMPLE_RATE)

        # Save to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)

            # Transcribe using Whisper
            result = model.transcribe(f.name)
            transcript = result["text"].strip()
            print("📝", transcript)

            # Check for exit phrase
            if "stop listening" in transcript.lower():
                print("👋 Heard 'stop listening'. Exiting...")
                break

    except KeyboardInterrupt:
        print("\n🛑 Manually stopped")
        break
