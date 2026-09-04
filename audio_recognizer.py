import pyaudio
import numpy as np
import speech_recognition as sr
import threading

class AudioRecognizer:
    def __init__(self, mic_id=1, rate=16000, channels=2, chunk=1024):
        self.mic_id = mic_id
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        
        self.volume_score = 0.0
        self.is_running = True
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.mic_id,
            frames_per_buffer=self.chunk
        )
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_listening = None

        threading.Thread(target=self._listen_volume_logic, daemon=True).start()

    def start_speech_to_text(self, on_text_detected_callback):
        """バックグラウンドで音声認識（テキスト化）を開始"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        #無音判定、途切れ防止
        self.recognizer.pause_threshold = 1.2
        self.recognizer.phrase_threshold = 0.1
        self.recognizer.non_speaking_duration = 0.8
        
        def callback_wrapper(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio, language='ja-JP')
                if text.strip():
                    on_text_detected_callback(text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"音声認識APIエラー: {e}")

        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone, 
            callback_wrapper,
            phrase_time_limit=15
        )

    def _listen_volume_logic(self):
        noise_threshold = 30.0
        max_volume = 3000.0
        
        while self.is_running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                if self.channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                
                adj_rms = max(0.0, rms - noise_threshold)
                score = min(1.0, adj_rms / (max_volume - noise_threshold))
                self.volume_score = float(round(score, 3))
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()