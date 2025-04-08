import pyaudio
import wave
import numpy as np
import tempfile
import os
import threading
import queue
from faster_whisper import WhisperModel

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Adjusted based on your environment
SILENCE_DURATION = 3  # seconds of silence to stop recording

class AudioProcessor:
    """Handles microphone input and speech-to-text conversion using offline Whisper"""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize Whisper model
        # Options: "base", "small", "medium", "large"
        model_size = os.getenv("WHISPER_MODEL", "base.en")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float32")
        print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
        
        # Use CPU by default, but you can set device="cuda" if you have a GPU
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        print("Whisper model loaded successfully")
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Listening... (speak now)")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def _record_audio(self):
        """Record audio until silence is detected"""
        frames = []
        silent_chunks = 0
        silent_threshold = int(SILENCE_DURATION * RATE / CHUNK)
        
        while self.is_recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Detect silence
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > silent_threshold:
                    break
            else:
                silent_chunks = 0
        
        # Save the recorded audio for Whisper processing
        if len(frames) > 0:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self.audio_queue.put(temp_filename)
        
        # Close the stream
        self.stream.stop_stream()
        self.stream.close()
        self.is_recording = False
        print("Finished recording")
        
    def get_transcription(self):
        """Convert recorded audio to text using Whisper"""
        if self.audio_queue.empty():
            return None
            
        audio_file = self.audio_queue.get()
        print("Transcribing audio...")
        
        try:
            # Transcribe with Whisper
            segments, _ = self.model.transcribe(audio_file, beam_size=5)
            
            # Combine all segments into one text
            text = " ".join([segment.text for segment in segments])
            
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)