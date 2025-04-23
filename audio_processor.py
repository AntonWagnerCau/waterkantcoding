import pyaudio
import wave
import numpy as np
import tempfile
import os
import threading
import queue
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024 
SILENCE_THRESHOLD = 100 # Adjusted based on your environment
SNIPPET_SILENCE_DURATION = 0.5  # seconds of silence to detect snippet end
SENTENCE_SILENCE_DURATION = 0.5  # seconds of silence to detect sentence end (Kind of redundant at the moment)
SPEECH_ENERGY_THRESHOLD_MULTIPLIER = 10  # Multiplier over background noise level for speech detection
NOISE_ESTIMATION_SECONDS = 1  # Initial seconds to estimate background noise

class AudioProcessor:
    """Handles microphone input and speech-to-text conversion using offline Whisper"""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.current_sentence = []
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.last_snippet_time = None  # Track time of last snippet transcription
        self.background_noise_level = 0  # Initial background noise level
        
        # Initialize Whisper model
        model_size = os.getenv("WHISPER_MODEL", "base.en")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float32")
        print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
        
        self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        print("Whisper model loaded successfully")
        
        # Start background transcription thread
        self.transcription_thread = threading.Thread(target=self._transcribe_background)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if not self.is_recording:
            self.is_recording = True
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Estimate background noise level at start
            self._estimate_background_noise()
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            print("Recording already in progress")
            
    def _estimate_background_noise(self):
        """Estimate background noise level from initial audio input"""
        print("Estimating background noise... please be silent for a moment.")
        noise_frames = []
        noise_duration_chunks = int(NOISE_ESTIMATION_SECONDS * RATE / CHUNK)
        
        for _ in range(noise_duration_chunks):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            noise_frames.append(data)
        
        # Calculate average energy of noise
        noise_data = np.frombuffer(b''.join(noise_frames), dtype=np.int16)
        if len(noise_data) > 0:
            self.background_noise_level = np.abs(noise_data).mean()
        else:
            self.background_noise_level = SILENCE_THRESHOLD
        print(f"Background noise level estimated: {self.background_noise_level}")

    def _has_speech(self, audio_data):
        """Check if the audio chunk contains speech based on energy level"""
        energy = np.abs(audio_data).mean()
        speech_threshold = self.background_noise_level * SPEECH_ENERGY_THRESHOLD_MULTIPLIER
        return energy > speech_threshold

    def _record_audio(self):
        """Record audio in snippets until explicitly stopped"""
        while self.is_recording:
            frames = []
            silent_chunks_snippet = 0
            snippet_threshold = int(SNIPPET_SILENCE_DURATION * RATE / CHUNK)
            
            # Record until snippet silence is detected
            while self.is_recording:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Detect silence for snippet break
                audio_data = np.frombuffer(data, dtype=np.int16)
                is_silent = np.abs(audio_data).mean() < SILENCE_THRESHOLD
                
                if is_silent:
                    silent_chunks_snippet += 1
                    if silent_chunks_snippet >= snippet_threshold:
                        break
                else:
                    silent_chunks_snippet = 0
            
            # Check if the recorded snippet contains speech
            snippet_audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            if len(snippet_audio_data) > 0 and self._has_speech(snippet_audio_data):
                # Save the recorded snippet for Whisper processing
                if len(frames) > 0:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_filename = temp_audio.name
                    
                    wf = wave.open(temp_filename, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    
                    self.audio_queue.put((temp_filename, False)) 
            
            # Continue recording immediately
            continue

    def _transcribe_background(self):
        """Background thread to transcribe audio snippets and build sentences"""
        import time
        while True:
            if not self.audio_queue.empty():
                audio_file, _ = self.audio_queue.get() 
                
                try:
                    # Transcribe with Whisper
                    segments, _ = self.model.transcribe(audio_file, beam_size=5)
                    
                    # Combine all segments into one text
                    text = " ".join([segment.text for segment in segments]).strip()
                    current_time = time.time()
                    
                    # Check if there's a previous snippet and if the gap indicates a sentence end
                    if self.last_snippet_time is not None and self.current_sentence:
                        time_since_last_snippet = current_time - self.last_snippet_time
                        if time_since_last_snippet >= SENTENCE_SILENCE_DURATION:
                            full_sentence = " ".join(self.current_sentence)
                            self.transcription_queue.put(full_sentence)
                            print(f"Full sentence (time-based): {full_sentence}")
                            self.current_sentence = []  # Reset for next sentence
                            # Restart recording if not already recording
                            if not self.is_recording:
                                self.start_recording()
                    
                    if text and text != "You":
                        self.current_sentence.append(text)
                        self.last_snippet_time = current_time
                except Exception as e:
                    print(f"Error transcribing audio: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
            else:
                # Check if there's a current sentence and a long gap since last snippet
                if self.current_sentence and self.last_snippet_time is not None:
                    current_time = time.time()
                    time_since_last_snippet = current_time - self.last_snippet_time
                    if time_since_last_snippet >= SENTENCE_SILENCE_DURATION:
                        full_sentence = " ".join(self.current_sentence)
                        self.transcription_queue.put(full_sentence)
                        self.current_sentence = []
                        self.last_snippet_time = None
                        # Restart recording if not already recording
                        if not self.is_recording:
                            self.start_recording()
                # Sleep briefly to avoid high CPU usage
                threading.Event().wait(0.1)
                
    def get_transcription(self):
        """Get the latest full sentence transcription"""
        if self.transcription_queue.empty():
            return None
        return self.transcription_queue.get()

    def stop(self):
        """Stop recording and clean up resources"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Audio processing stopped")

if __name__ == "__main__":
    import time
    
    # Create an instance of AudioProcessor
    processor = AudioProcessor()
    
    try:
        # Start recording
        processor.start_recording()
        
        # Continuously check for transcriptions
        print("Testing audio processing. Speak now, and I'll display transcriptions. Press Ctrl+C to stop.")
        while True:
            transcription = processor.get_transcription()
            if transcription:
                print(f"Transcription received: {transcription}")
            time.sleep(0.1)  # Avoid high CPU usage
    except KeyboardInterrupt:
        print("\nStopping audio processing...")
        processor.stop()
        print("Test completed.")