import io
import pyaudio
import numpy as np
import speech_recognition as sr
import threading
from faster_whisper import WhisperModel

class AudioRecognizer:
    def __init__(self, mic_id=1, rate=16000, channels=2, chunk=1024, model_size="small"):
        self.mic_id = mic_id
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        
        self.volume_score = 0.0
        self.is_running = True

        self.p = pyaudio.PyAudio()
        self.stream = None
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.mic_id,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"[AudioRecognizer]: マイクオープンエラー: {e}")

        try:
            target_model = "turbo" if model_size == "small" else model_size
            print(f"[Whisper]: GPU(CUDA) モードでモデル '{target_model}' を読み込み中...")
            self.whisper_model = WhisperModel(target_model, device="cuda", compute_type="float16")
            print("[Whisper]: CUDA (GPU) 高速モードで起動しました。")
        except Exception:
            print(f"[Whisper]: CPU モードでモデル '{model_size}' を読み込み中...")
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("[Whisper]: CPU モードで起動しました。")

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_listening = None

        threading.Thread(target=self._listen_volume_logic, daemon=True).start()

    def start_speech_to_text(self, on_text_detected_callback):
        """バックグラウンドで Whisper による高精度音声認識を開始"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.1
        self.recognizer.non_speaking_duration = 0.6
        
        def callback_wrapper(recognizer, audio):
            try:
                wav_data = audio.get_wav_data()
                audio_stream = io.BytesIO(wav_data)

                segments, _ = self.whisper_model.transcribe(
                    audio_stream, 
                    language="ja",
                    beam_size=5,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=400,
                        threshold=0.5
                    ),
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    initial_prompt="こんにちは。日本語のリアルタイム音声認識です。自然な日本語の文脈で正確に書き起こしてください。"
                )
                
                text = "".join([segment.text for segment in segments]).strip()
                
                if text:
                    on_text_detected_callback(text)

            except Exception as e:
                print(f"[Whisper エラー]: {e}")

        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone, 
            callback_wrapper,
            phrase_time_limit=15
        )

    def _listen_volume_logic(self):
        noise_threshold = 30.0
        max_volume = 3000.0
        
        while self.is_running and self.stream:
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
            try:
                self.stop_listening(wait_for_stop=False)
            except Exception:
                pass

        try:
            if hasattr(self, 'stream') and self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
        except Exception as e:
            print(f"[AudioRecognizer]: ストリーム終了時処理: {e}")

        try:
            if hasattr(self, 'p') and self.p:
                self.p.terminate()
        except Exception:
            pass
