import sounddevice as sd
import numpy as np
import pvporcupine
from faster_whisper import WhisperModel
import requests
from datetime import datetime
import os
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
import platform
import wave
import json

# Optional imports for enhanced features (graceful degradation)
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, some features will be limited")

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("Warning: Resemblyzer not available, speaker identification disabled")

try:
    if TORCH_AVAILABLE:
        # Silero VAD - only load if torch is available
        vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        (get_speech_timestamps, _, _, _, _) = utils
        SILERO_VAD_AVAILABLE = True
    else:
        SILERO_VAD_AVAILABLE = False
except Exception as e:
    print(f"Warning: Silero VAD not available: {e}")
    SILERO_VAD_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
# WAKE_WORD = "picovoice"            # Porcupine custom wake word
SAMPLE_RATE = 16000
DEVICE_INDEX = None                    # None for default mic (better for RPi)
CHUNK_SIZE = 512                       # audio chunk size
STOP_SILENCE_SEC = 2.0                 # reduced from 3.0s to 2.0s for better balance
PRE_BUFFER_SEC = 1.0                   # seconds of audio to keep before wake word
MIN_RECORDING_SEC = 3.0                # minimum recording duration before considering end of speech
MAX_RECORDING_SEC = 45.0               # maximum recording duration regardless of silence
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# Get script directory for resource paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect platform for optimizations
IS_RASPBERRY_PI = platform.machine().lower().startswith('arm') or 'raspi' in platform.uname().node.lower()
IS_LINUX = platform.system().lower() == 'linux'

# Raspberry Pi specific optimizations
if IS_RASPBERRY_PI:
    print("Raspberry Pi detected - applying optimizations")
    CHUNK_SIZE = 1024  # Larger chunks for better performance on RPi
    # Use lighter models
    WHISPER_MODEL_SIZE = "tiny"  # Much faster on RPi
    WHISPER_COMPUTE_TYPE = "int8"  # Lower precision for speed
else:
    WHISPER_MODEL_SIZE = "medium"
    WHISPER_COMPUTE_TYPE = "float16" if TORCH_AVAILABLE else "int8"


# -----------------------------
# Debugging
# -----------------------------
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Is Raspberry Pi: {IS_RASPBERRY_PI}")
print(f"Using device index: {DEVICE_INDEX}")
print(f"Webhook URL: {WEBHOOK_URL}")
print(f"Available audio devices: {sd.query_devices()}")

# -----------------------------
# Initialize models
# -----------------------------
# Initialize Porcupine with platform-specific wake word file
def get_porcupine_keyword_path():
    """Get the appropriate porcupine keyword file for the current platform."""
    resources_dir = os.path.join(SCRIPT_DIR, "resources")
    
    if IS_RASPBERRY_PI or IS_LINUX:
        # Try to find Linux/RPi specific keyword file
        linux_keyword = os.path.join(resources_dir, "Hey-Dude_en_raspberry-pi_v3_0_0.ppn")
        if os.path.exists(linux_keyword):
            return linux_keyword
        # Fallback to a generic Linux keyword if available
        linux_generic = os.path.join(resources_dir, "Hey-Dude_en_linux_v3_0_0.ppn")
        if os.path.exists(linux_generic):
            return linux_generic
    
    # Default to Windows keyword (may work on some platforms)
    windows_keyword = os.path.join(resources_dir, "Hey-Dude_en_windows_v3_0_0.ppn")
    if os.path.exists(windows_keyword):
        print("Warning: Using Windows keyword file on non-Windows platform - this may not work properly")
        return windows_keyword
    
    raise FileNotFoundError("No suitable Porcupine keyword file found for this platform")

try:
    keyword_path = get_porcupine_keyword_path()
    print(f"Using keyword file: {keyword_path}")
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[keyword_path],
    )
    print("Porcupine wake word detection initialized successfully")
except Exception as e:
    print(f"Error initializing Porcupine: {e}")
    print("You may need to download the appropriate .ppn file for your platform")
    raise

# Initialize Whisper model with platform optimizations
print("Initializing Whisper model...")
if TORCH_AVAILABLE and torch.cuda.is_available() and not IS_RASPBERRY_PI:
    # Only try GPU on non-RPi systems
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        print("CUDA test successful")
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
        print(f"Using GPU for Whisper ({WHISPER_MODEL_SIZE} model)")
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        print(f"Falling back to CPU ({WHISPER_MODEL_SIZE} model)")
else:
    # CPU-only for RPi and systems without CUDA
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
    print(f"Using CPU for Whisper ({WHISPER_MODEL_SIZE} model, {WHISPER_COMPUTE_TYPE} precision)")

# Initialize optional components
encoder = None
speaker_db = {}  # {'Alice': embedding, 'Bob': embedding} for known family members

if RESEMBLYZER_AVAILABLE:
    try:
        encoder = VoiceEncoder()
        print("Resemblyzer speaker identification enabled")
    except Exception as e:
        print(f"Error initializing Resemblyzer: {e}")
        RESEMBLYZER_AVAILABLE = False
else:
    print("Resemblyzer not available - speaker identification disabled")

# -----------------------------
# Helper functions
# -----------------------------
def is_speech(audio_chunk):
    """Detect speech in audio chunk using available VAD methods."""
    if SILERO_VAD_AVAILABLE and TORCH_AVAILABLE:
        # Use Silero VAD if available
        tensor = torch.tensor(audio_chunk, dtype=torch.float32)
        speech_segments = get_speech_timestamps(
            tensor, 
            vad_model, 
            sampling_rate=SAMPLE_RATE,
            # Tunable parameters:
            threshold=0.27,              # Adjusted from 0.25 to 0.27 - slightly less sensitive
            min_speech_duration_ms=200,  # Short enough to catch brief utterances
            max_speech_duration_s=float('inf'), 
            min_silence_duration_ms=250, # Adjusted from 300 to 250
            window_size_samples=512,
            speech_pad_ms=50
        )
        return len(speech_segments) > 0
    else:
        # Fallback to simple energy-based VAD
        return simple_energy_vad(audio_chunk)

def simple_energy_vad(audio_chunk, threshold=0.01):
    """Simple energy-based voice activity detection for when Silero VAD is not available."""
    if len(audio_chunk) == 0:
        return False
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    return rms > threshold

def identify_speaker(wav_file):
    """Identify speaker from audio file. Returns 'unknown_speaker' if speaker ID is disabled."""
    if not RESEMBLYZER_AVAILABLE or encoder is None:
        return "unknown_speaker"
    
    try:
        wav = preprocess_wav(wav_file)
        emb = encoder.embed_utterance(wav)
        if not speaker_db:
            return "unknown_speaker"
        
        # Compare embeddings (cosine similarity)
        best_match, best_score = "unknown_speaker", -1
        for name, ref_emb in speaker_db.items():
            score = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
            if score > best_score:
                best_score, best_match = score, name
        return best_match
    except Exception as e:
        print(f"Error in speaker identification: {e}")
        return "unknown_speaker"

def send_to_n8n(speaker, text, wav_file, wake_time=None):
    """Send audio and transcript to webhook endpoint."""
    if not WEBHOOK_URL:
        print("No webhook URL configured - skipping webhook")
        return
        
    try:
        with open(wav_file, "rb") as f:
            audio_data = f.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            'speaker': speaker,
            'text': text,
            'audio': audio_base64,
            'timestamp': datetime.now().isoformat()
        }

        play_sound_file(os.path.join(SCRIPT_DIR, "resources", "success.wav"))
        headers = {'Content-Type': 'application/json'}
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=30)
        
        if wake_time:
            elapsed = (datetime.now() - wake_time).total_seconds()
            print(f"[{elapsed:.2f}s] Webhook response: {response.status_code}")
        else:
            print(f"Webhook response: {response.status_code}")
            
        if response.status_code == 200:
            print("Audio successfully sent to webhook. Checking for audio response...")
            # Check if response contains audio data to play
            response_data = response.json()
            if response_data.get("data"):
                play_audio(response_data.get("data"))
        else:
            print(f"Webhook error: {response.text}")
            play_sound_file(os.path.join(SCRIPT_DIR, "resources", "error.wav"))
    
    except Exception as e:
        if wake_time:
            elapsed = (datetime.now() - wake_time).total_seconds()
            print(f"[{elapsed:.2f}s] Error sending to n8n: {e}")
        else:
            print(f"Error sending to n8n: {e}")
        play_sound_file(os.path.join(SCRIPT_DIR, "resources", "error.wav"))
        raise  # Re-raise to trigger outer error handling

def play_audio(b64_data):
    """Play audio file using system command"""
    if not b64_data:
        return
        
    print("Playing audio response...")
    try:
        audio_data = base64.b64decode(b64_data)
        temp_audio_file = f"temp_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        with open(temp_audio_file, "wb") as f:
            f.write(audio_data)
        
        play_sound_file(temp_audio_file)
        
        # Clean up temp file with retry logic
        def cleanup():
            import time
            max_retries = 5
            retry_delay = 10  # seconds
            
            for attempt in range(max_retries):
                try:
                    time.sleep(retry_delay)
                    if os.path.exists(temp_audio_file):
                        os.remove(temp_audio_file)
                        print(f"Successfully cleaned up {temp_audio_file}")
                        break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"Cleanup attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    else:
                        print(f"Failed to cleanup {temp_audio_file} after {max_retries} attempts: {e}")
                except Exception as e:
                    print(f"Unexpected error during cleanup: {e}")
                    break
        
        threading.Thread(target=cleanup, daemon=True).start()
        
    except Exception as e:
        print(f"Error playing audio: {e}")

def save_audio_wav(audio_data, filename, sample_rate=16000):
    """Save audio data as WAV file using wave module (fallback when torchaudio not available)."""
    try:
        if TORCH_AVAILABLE and torchaudio:
            # Use torchaudio if available
            torchaudio_tensor = torch.tensor(audio_data).unsqueeze(0)
            torchaudio.save(filename, torchaudio_tensor, sample_rate)
        else:
            # Fallback to wave module
            # Convert float32 audio to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
        
        return True
    except Exception as e:
        print(f"Error saving audio file {filename}: {e}")
        return False

def play_sound_file(file_path):
    """Play a sound file using system command"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'afplay "{file_path}"')
        else:  # Linux and others (including Raspberry Pi)
            # Use aplay for Linux/RPi
            os.system(f'aplay "{file_path}" 2>/dev/null')
    except Exception as e:
        print(f"Error playing sound file {file_path}: {e}")

# -----------------------------
# Main loop
# -----------------------------
def main():
    print("Listening for wake word...")

    audio_buffer = []
    pre_buffer = []
    triggered = False
    last_speech_time = datetime.now()
    wake_time = None
    pre_buffer_size = int(PRE_BUFFER_SEC * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                        channels=1, dtype='int16', device=DEVICE_INDEX) as stream:

        while True:
            pcm = stream.read(CHUNK_SIZE)[0].flatten()
            
            # Convert to numpy array immediately for consistency
            pcm_array = np.frombuffer(pcm.tobytes(), dtype=np.int16)
            
            # Maintain pre-buffer for wake word detection
            if not triggered:
                pre_buffer.extend(pcm_array)
                # Keep only the last PRE_BUFFER_SEC seconds
                if len(pre_buffer) > pre_buffer_size:
                    pre_buffer = pre_buffer[-pre_buffer_size:]
            else:
                audio_buffer.extend(pcm_array)

            # -----------------------------
            # Wake word detection
            # -----------------------------
            if not triggered:
                if porcupine.process(pcm_array) >= 0:
                    wake_time = datetime.now()
                    print(f"[0.00s] Wake word detected!")

                    triggered = True
                    # Start with pre-buffer + current chunk
                    audio_buffer = pre_buffer + list(pcm_array)
                    pre_buffer = []  # Clear pre-buffer
                    last_speech_time = datetime.now()
                continue

            # -----------------------------
            # Speech detection via VAD
            # -----------------------------
            if len(audio_buffer) > SAMPLE_RATE // 4:  # Process every 0.25 seconds
                float_audio = np.array(audio_buffer[-SAMPLE_RATE//4:]).astype(np.float32) / 32768.0
                if is_speech(float_audio):
                    last_speech_time = datetime.now()
                else:
                    silence = (datetime.now() - last_speech_time).total_seconds()
                    recording_duration = (datetime.now() - wake_time).total_seconds() if wake_time else 0
                    
                    # Debug output for silence detection
                    if silence > 1.0 and silence < STOP_SILENCE_SEC and recording_duration > MIN_RECORDING_SEC:
                        elapsed = (datetime.now() - wake_time).total_seconds()
                        print(f"[{elapsed:.2f}s] Silence detected for {silence:.1f}s...")
                    
                    # End recording if:
                    # 1. Minimum recording time has passed AND silence threshold is exceeded
                    # 2. OR maximum recording time has been reached
                    if (silence > STOP_SILENCE_SEC and recording_duration > MIN_RECORDING_SEC and len(audio_buffer) > SAMPLE_RATE) or \
                       (recording_duration > MAX_RECORDING_SEC):
                        assert wake_time is not None, "wake_time should not be None when processing audio"
                        elapsed = (datetime.now() - wake_time).total_seconds()
                        reason = "silence detected" if silence > STOP_SILENCE_SEC else "maximum recording time reached"
                        print(f"[{elapsed:.2f}s] Speech ended ({reason}), processing... (includes {PRE_BUFFER_SEC}s pre-buffer)")
                        
                        wav_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

                        try:
                            # Save WAV for Resemblyzer and Whisper
                            full_audio = np.array(audio_buffer).astype(np.float32) / 32768.0
                            if not save_audio_wav(full_audio, wav_filename, SAMPLE_RATE):
                                print("Failed to save audio file")
                                continue

                            # Run speaker identification and transcription in parallel
                            elapsed = (datetime.now() - wake_time).total_seconds()
                            print(f"[{elapsed:.2f}s] Starting speaker identification and STT transcription in parallel...")
                            
                            speaker = None
                            transcript = None
                            
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                # Submit both tasks
                                speaker_future = executor.submit(identify_speaker, wav_filename)
                                transcribe_future = executor.submit(lambda: model.transcribe(wav_filename, beam_size=5))
                                
                                # Wait for both to complete
                                for future in as_completed([speaker_future, transcribe_future]):
                                    if future == speaker_future:
                                        speaker = future.result()
                                        elapsed = (datetime.now() - wake_time).total_seconds()
                                        print(f"[{elapsed:.2f}s] Speaker identification complete: {speaker}")
                                    elif future == transcribe_future:
                                        segments, _ = future.result()
                                        transcript = " ".join([getattr(seg, 'text', str(seg)) for seg in segments]).strip()
                                        elapsed = (datetime.now() - wake_time).total_seconds()
                                        print(f"[{elapsed:.2f}s] STT transcription complete: {transcript}")

                            elapsed = (datetime.now() - wake_time).total_seconds()
                            print(f"[{elapsed:.2f}s] Both tasks completed")

                            # Send to n8n only if we have text
                            if transcript:
                                try:
                                    elapsed = (datetime.now() - wake_time).total_seconds()
                                    print(f"[{elapsed:.2f}s] Sending webhook request...")
                                    send_to_n8n(speaker, transcript, wav_filename, wake_time)
                                    elapsed = (datetime.now() - wake_time).total_seconds()
                                    print(f"[{elapsed:.2f}s] Webhook request complete")
                                except Exception as webhook_error:
                                    elapsed = (datetime.now() - wake_time).total_seconds()
                                    print(f"[{elapsed:.2f}s] Webhook failed, continuing: {webhook_error}")
                                    play_sound_file(os.path.join(SCRIPT_DIR, "resources", "error.wav"))
                            else:
                                elapsed = (datetime.now() - wake_time).total_seconds()
                                print(f"[{elapsed:.2f}s] No transcript generated, skipping webhook")

                        except Exception as e:
                            elapsed = (datetime.now() - wake_time).total_seconds()
                            print(f"[{elapsed:.2f}s] Error processing audio: {e}")
                            play_sound_file(os.path.join(SCRIPT_DIR, "resources", "error.wav"))

                        finally:
                            # Clean up temp file
                            if os.path.exists(wav_filename):
                                try:
                                    os.remove(wav_filename)
                                except Exception as cleanup_error:
                                    print(f"Failed to cleanup temp file: {cleanup_error}")
                            
                            # Reset - ensure this always happens
                            triggered = False
                            audio_buffer = []
                            pre_buffer = []
                            wake_time = None
                            print("Ready for next wake word...")

if __name__ == "__main__":
    main()
