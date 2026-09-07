import cv2
import sys
import io
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

from text_analyzer import TextAnalyzer
from audio_recognizer import AudioRecognizer

API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MIC_ID = 1
DETECTOR_BACKEND = 'ssd' 

COLORS = {
    "Joy": (0, 255, 255),
    "Sadness": (255, 105, 180),
    "Anger": (0, 0, 255),
    "Fear": (128, 0, 128),
    "Neutral": (200, 200, 200),
    "Total": (0, 255, 0)
}

EMOTION_JP = {
    "Joy": "喜び",
    "Sadness": "悲しみ",
    "Anger": "怒り",
    "Fear": "恐れ",
    "Neutral": "普通"
}

MODE_NAMES = {
    "presentation": "プレゼン（聴衆）",
    "interview": "面接（採用担当）",
    "casual": "雑談（友人）"
}

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def draw_japanese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("meiryo.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()
        
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    print("ダッシュボードを起動中...")
    
    text_analyzer = TextAnalyzer(api_key=API_KEY, mode="presentation")
    audio_recognizer = AudioRecognizer(mic_id=MIC_ID)
    audio_recognizer.start_speech_to_text(text_analyzer.analyze_async)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    score_history = deque(maxlen=5)
    last_region = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        panel_w = 400
        bottom_h = 70
        
        canvas = cv2.copyMakeBorder(frame, 0, bottom_h, 0, panel_w, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        
        current_face_scores = None

        try:
            results = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=True, 
                silent=True
            )
            if results:
                res = results[0]
                emo = res['emotion']
                current_face_scores = {
                    "Joy": emo['happy']/100, "Sadness": emo['sad']/100,
                    "Anger": emo['angry']/100, "Fear": emo['fear']/100, "Neutral": emo['neutral']/100
                }
                last_region = res['region']
        except Exception:
            pass

        if current_face_scores:
            score_history.append(current_face_scores)
        
        face_scores = {"Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0, "Neutral": 1.0}
        if len(score_history) > 0:
            for key in face_scores:
                face_scores[key] = round(sum(s[key] for s in score_history) / len(score_history), 3)

        if last_region and len(score_history) > 0:
            reg = last_region
            cv2.rectangle(canvas, (reg['x'], reg['y']), (reg['x']+reg['w'], reg['y']+reg['h']), COLORS["Total"], 2)

        txt_scores = text_analyzer.last_res
        txt_only_scores = text_analyzer.text_only_res
        vol_score = audio_recognizer.volume_score
        current_text = text_analyzer.last_text

        #総合スコア計算
        total_scores = {}
        dominant_emo = "Neutral"
        max_s = -1

        for emo in ["Joy", "Sadness", "Anger", "Fear"]:
            base = face_scores[emo] * 0.6 + txt_scores[emo] * 0.4
            boosted = base * (1.0 + vol_score * 0.5)
            total_scores[emo] = min(1.0, round(boosted, 3))
            if total_scores[emo] > max_s:
                max_s = total_scores[emo]
                dominant_emo = emo
        
        total_scores["Neutral"] = round(face_scores["Neutral"] * 0.6 + txt_scores["Neutral"] * 0.4, 3)
        if total_scores["Neutral"] > max_s:
            dominant_emo = "Neutral"

        # UI描画
        px = w + 20
        py = 20

        #評価対象
        mode_label = MODE_NAMES.get(text_analyzer.current_mode, "未設定")
        cv2.rectangle(canvas, (px - 5, py - 5), (px + 375, py + 55), (40, 40, 40), -1)
        cv2.rectangle(canvas, (px - 5, py - 5), (px + 375, py + 55), (0, 255, 255), 1)
        canvas = draw_japanese_text(canvas, f"評価対象: {mode_label}", (px + 5, py), font_size=18, color=(0, 255, 255))
        canvas = draw_japanese_text(canvas, "[1]プレゼン [2]面接 [3]雑談", (px + 5, py + 28), font_size=13, color=(180, 180, 180))

        #総合判定
        g_py = py + 70
        canvas = draw_japanese_text(canvas, "【総合感情】", (px, g_py), font_size=18, color=(255, 255, 255))
        canvas = draw_japanese_text(canvas, EMOTION_JP[dominant_emo], (px, g_py + 30), font_size=30, color=COLORS[dominant_emo])
        
        #声の大きさ
        v_py = g_py + 85
        canvas = draw_japanese_text(canvas, "声の熱量", (px, v_py - 5), font_size=15, color=(200, 200, 200))
        cv2.rectangle(canvas, (px + 90, v_py - 5), (px + 90 + 200, v_py + 15), (50, 50, 50), -1)
        v_w = int(200 * vol_score)
        if v_w > 0:
            cv2.rectangle(canvas, (px + 90, v_py - 5), (px + 90 + v_w, v_py + 15), COLORS["Joy"], -1)
        canvas = draw_japanese_text(canvas, f"{vol_score:.2f}", (px + 300, v_py - 5), font_size=15, color=(255, 255, 255))

        # 詳細
        m_py = v_py + 40
        canvas = draw_japanese_text(canvas, "【感情詳細】(上:表情 / 中:総合 / 下:言葉)", (px, m_py), font_size=14, color=(200, 200, 200))
        
        for i, emo in enumerate(["Joy", "Sadness", "Anger", "Fear", "Neutral"]):
            y = m_py + 30 + i * 45
            color = COLORS[emo]
            
            canvas = draw_japanese_text(canvas, EMOTION_JP[emo], (px, y - 5), font_size=15, color=(220, 220, 220))
            
            gx = px + 75
            gw = 200
            cv2.rectangle(canvas, (gx, y - 5), (gx + gw, y + 18), (30, 30, 30), -1)
            
            f_w = int(gw * face_scores[emo])
            if f_w > 0: cv2.rectangle(canvas, (gx, y - 3), (gx + f_w, y + 1), color, 1)

            t_w = int(gw * total_scores[emo])
            if t_w > 0: cv2.rectangle(canvas, (gx, y + 3), (gx + t_w, y + 10), color, -1)

            x_w = int(gw * txt_only_scores[emo])
            if x_w > 0: cv2.rectangle(canvas, (gx, y + 12), (gx + x_w, y + 16), (0, 255, 255), -1)

            score_str = f"{total_scores[emo]:.2f}({txt_only_scores[emo]:.2f})"
            canvas = draw_japanese_text(canvas, score_str, (gx + gw + 5, y - 5), font_size=12, color=COLORS["Total"])

        #下部UI
        display_txt = current_text if len(current_text) < 25 else current_text[:23] + "..."
        canvas = draw_japanese_text(canvas, f"認識された言葉: {display_txt}", (15, h + 10), font_size=16, color=(255, 255, 255))
        
        score_color = (0, 255, 0) if text_analyzer.score >= 70 else (0, 165, 255)
        canvas = draw_japanese_text(canvas, f"発話スコア: {text_analyzer.score}点", (w + 20, h + 10), font_size=18, color=score_color)

        advice_txt = text_analyzer.advice if len(text_analyzer.advice) < 40 else text_analyzer.advice[:38] + "..."
        canvas = draw_japanese_text(canvas, f"AIアドバイス: {advice_txt}", (15, h + 38), font_size=16, color=(0, 255, 255))

        if text_analyzer.is_processing:
            canvas = draw_japanese_text(canvas, "AI分析中...", (w - 110, h + 10), font_size=16, color=COLORS["Joy"])

        cv2.imshow("リアルタイム感情分析ダッシュボード", canvas)
        
        # キー入力
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            text_analyzer.set_mode("presentation")
        elif key == ord('2'):
            text_analyzer.set_mode("interview")
        elif key == ord('3'):
            text_analyzer.set_mode("casual")

    cap.release()
    audio_recognizer.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
