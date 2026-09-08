import cv2
import sys
import io
import csv
import os
import datetime
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

from text_analyzer import TextAnalyzer
from audio_recognizer import AudioRecognizer

API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MIC_ID = 1
DETECTOR_BACKEND = 'ssd' 
LOG_FILE = "log_history.csv"

COLORS = {
    "Joy": (0, 215, 255),
    "Sadness": (255, 120, 160),
    "Anger": (80, 80, 255),
    "Fear": (200, 100, 200),
    "Neutral": (180, 190, 200),
    "Accent": (255, 190, 0),
    "BgCard": (35, 38, 45),
    "BgMain": (18, 20, 25)
}

EMOTION_JP = {
    "Joy": "喜び", "Sadness": "悲しみ", "Anger": "怒り", "Fear": "恐れ", "Neutral": "普通"
}

MODE_NAMES = {
    "presentation": "プレゼン（聴衆）", "interview": "面接（採用担当）", "casual": "雑談（友人）"
}

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def draw_japanese_text(img, text, position, font_size=18, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = None
    font_paths = ["meiryo.ttc", "msgothic.ttc", "YuGothM.ttc", "hiragino.otf"]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
        
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_card(img, pt1, pt2, color, border_color=None):
    cv2.rectangle(img, pt1, pt2, color, -1)
    if border_color:
        cv2.rectangle(img, pt1, pt2, border_color, 1, cv2.LINE_AA)

def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["日時", "モード", "認識テキスト", "発話スコア", "AIアドバイス", "総合感情", "喜び", "悲しみ", "怒り", "恐れ", "普通"])

def save_log(mode, text, score, advice, dominant_emo, total_scores):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            now, MODE_NAMES.get(mode, mode), text, score, advice, EMOTION_JP[dominant_emo],
            total_scores.get("Joy", 0), total_scores.get("Sadness", 0),
            total_scores.get("Anger", 0), total_scores.get("Fear", 0), total_scores.get("Neutral", 0)
        ])

def main():
    print("ダッシュボードを起動中...")
    init_log_file()
    
    text_analyzer = TextAnalyzer(api_key=API_KEY, mode="presentation")
    audio_recognizer = AudioRecognizer(mic_id=MIC_ID)
    audio_recognizer.start_speech_to_text(text_analyzer.analyze_async)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    score_history = deque(maxlen=5)
    last_region = None

    window_name = "Emotion AI Analytics Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        panel_w = 400
        bottom_h = 100
        
        #キャンバス生成
        canvas = cv2.copyMakeBorder(frame, 0, bottom_h, 0, panel_w, cv2.BORDER_CONSTANT, value=COLORS["BgMain"])
        
        current_face_scores = None
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'], detector_backend=DETECTOR_BACKEND, 
                enforce_detection=True, silent=True
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
            cv2.rectangle(canvas, (reg['x'], reg['y']), (reg['x']+reg['w'], reg['y']+reg['h']), COLORS["Accent"], 2, cv2.LINE_AA)

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

        if text_analyzer.has_new_log:
            save_log(
                text_analyzer.current_mode, text_analyzer.last_text,
                text_analyzer.score, text_analyzer.advice, dominant_emo, total_scores
            )
            text_analyzer.has_new_log = False

        #UI右側
        px = w + 15
        py = 15

        #ヘッダー＆モード
        draw_card(canvas, (px, py), (px + 370, py + 55), COLORS["BgCard"], COLORS["Accent"])
        canvas = draw_japanese_text(canvas, f"TARGET: {MODE_NAMES.get(text_analyzer.current_mode, '')}", (px + 12, py + 8), font_size=16, color=COLORS["Accent"])
        canvas = draw_japanese_text(canvas, "[1]プレゼン  [2]面接  [3]雑談", (px + 12, py + 32), font_size=12, color=(160, 170, 180))

        #総合感情
        g_py = py + 68
        draw_card(canvas, (px, g_py), (px + 370, g_py + 85), COLORS["BgCard"])
        canvas = draw_japanese_text(canvas, "TOTAL EMOTION", (px + 15, g_py + 10), font_size=12, color=(140, 150, 160))
        canvas = draw_japanese_text(canvas, EMOTION_JP[dominant_emo], (px + 15, g_py + 30), font_size=32, color=COLORS[dominant_emo])
        
        canvas = draw_japanese_text(canvas, "熱量 (AUDIO)", (px + 180, g_py + 10), font_size=12, color=(140, 150, 160))
        cv2.rectangle(canvas, (px + 180, g_py + 42), (px + 180 + 140, g_py + 58), (50, 55, 65), -1)
        v_w = int(140 * vol_score)
        if v_w > 0:
            cv2.rectangle(canvas, (px + 180, g_py + 42), (px + 180 + v_w, g_py + 58), COLORS["Joy"], -1)
        canvas = draw_japanese_text(canvas, f"{int(vol_score*100)}%", (px + 330, g_py + 40), font_size=12, color=(255, 255, 255))

        d_py = g_py + 98
        draw_card(canvas, (px, d_py), (px + 370, d_py + 210), COLORS["BgCard"])
        canvas = draw_japanese_text(canvas, "EMOTION BREAKDOWN (Face / Total / Text)", (px + 15, d_py + 8), font_size=12, color=(140, 150, 160))
        
        for i, emo in enumerate(["Joy", "Sadness", "Anger", "Fear", "Neutral"]):
            y = d_py + 32 + i * 35
            color = COLORS[emo]
            
            canvas = draw_japanese_text(canvas, EMOTION_JP[emo], (px + 15, y), font_size=14, color=(220, 225, 230))
            
            gx = px + 75
            gw = 210
            cv2.rectangle(canvas, (gx, y + 2), (gx + gw, y + 18), (50, 55, 65), -1)
        
            f_w = int(gw * face_scores[emo])
            if f_w > 0: cv2.rectangle(canvas, (gx, y + 3), (gx + f_w, y + 6), color, -1)

            t_w = int(gw * total_scores[emo])
            if t_w > 0: cv2.rectangle(canvas, (gx, y + 8), (gx + t_w, y + 13), color, -1)

            x_w = int(gw * txt_only_scores[emo])
            if x_w > 0: cv2.rectangle(canvas, (gx, y + 15), (gx + x_w, y + 17), COLORS["Accent"], -1)

            canvas = draw_japanese_text(canvas, f"{int(total_scores[emo]*100)}%", (gx + gw + 8, y + 1), font_size=12, color=(200, 210, 220))

        #UI
        draw_card(canvas, (10, h + 10), (w + panel_w - 10, h + bottom_h - 10), COLORS["BgCard"])

        display_txt = current_text if len(current_text) < 32 else current_text[:30] + "..."
        canvas = draw_japanese_text(canvas, f"発話内容: {display_txt}", (25, h + 20), font_size=15, color=(255, 255, 255))
        
        score_color = (0, 230, 120) if text_analyzer.score >= 70 else (0, 180, 255)
        canvas = draw_japanese_text(canvas, f"コミュニケーションスコア: {text_analyzer.score}点", (w + 15, h + 20), font_size=16, color=score_color)

        canvas = draw_japanese_text(canvas, f"AI Feedback: {text_analyzer.advice}", (25, h + 52), font_size=15, color=COLORS["Accent"])

        if text_analyzer.is_processing:
            canvas = draw_japanese_text(canvas, "AI Analyzing...", (w - 100, h + 20), font_size=13, color=COLORS["Joy"])

        cv2.imshow(window_name, canvas)
        
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
