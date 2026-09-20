# 文字起こし
1.Google Speech API 

**【メリット】**

短い発話に適する 

クラウド型だからGPUがなくても精度が変わらない 

**【デメリット】**

文脈や専門用語で誤認識が多い 

誤変換が一番多い 

2.Whisper「現在使用中」

**【メリット】** 

雑音などの環境音が一切入らない 

句読点入りの日本語文章を生成可能 

モデルによっては一番リアルタイム向き 

GPUもそこまで使わない 

**【デメリット】** 

英語検知はあまり向かない 

3. Qwen　ASR

**【メリット】**

文脈理解の精度が一番良い 

環境音が入りにくい 

トーンを正確に測ることができる 

**【デメリット】**

GPU使用率がダントツで高い 

mediapipeと同時使用するにはあまり向かない 

## 起動に必要なパッケージ

"` pip install opencv-python pillow numpy pandas janome google-genai deepface tf-keras faster-whisper PyAudio SpeechRecognition `"
