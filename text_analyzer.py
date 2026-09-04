import json
import threading
import pandas as pd
from janome.tokenizer import Tokenizer
from google import genai

class TextAnalyzer:
    def __init__(self, api_key, csv_path='JIWC-A_2018.csv'):
        self.t = Tokenizer()
        self.api_key = api_key
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception:
            self.client = None
            
        self.last_res = {"Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0, "Neutral": 1.0}
        self.text_only_res = {"Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0, "Neutral": 1.0} # ★テキスト単体スコア
        self.last_text = "（音声入力を待っています...）"
        self.is_processing = False

        self.emotion_dict = {}
        self.load_csv_dictionary(csv_path)

    def load_csv_dictionary(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Fear'] = df['Anxiety']
            df['Anger_Total'] = df['Anger'] + df['Disgust']
            
            self.emotion_dict = df.set_index('Words')[['Joy', 'Sadness', 'Anger_Total', 'Fear']].rename(columns={'Anger_Total': 'Anger'}).to_dict('index')
            print(f"感情辞書 ({len(self.emotion_dict)}単語) の読み込みが完了しました。")
        except Exception as e:
            print(f"辞書ファイルの読み込みに失敗しました: {e}")

    def analyze_async(self, text):
        if self.is_processing or not text.strip():
            return
        self.last_text = text
        threading.Thread(target=self._analyze_logic, args=(text,), daemon=True).start()

    def _analyze_logic(self, text):
        self.is_processing = True
        try:
            dict_scores = {"Joy": 0.0, "Sadness": 0.0, "Anger": 0.0, "Fear": 0.0, "Neutral": 0.0}
            tokens = self.t.tokenize(text)
            words = [token.surface for token in tokens]
            
            matched_count = 0
            for word in words:
                if word in self.emotion_dict:
                    scores = self.emotion_dict[word]
                    for emo in ["Joy", "Sadness", "Anger", "Fear"]:
                        dict_scores[emo] += scores[emo]
                    matched_count += 1

            if matched_count > 0:
                total_sum = sum(dict_scores.values())
                if total_sum > 0:
                    for emo in ["Joy", "Sadness", "Anger", "Fear"]:
                        dict_scores[emo] = round(dict_scores[emo] / total_sum, 3)
                else:
                    dict_scores["Neutral"] = 1.0
            else:
                dict_scores["Neutral"] = 1.0

            #テキストスコア、結果を更新
            self.text_only_res = dict_scores.copy()
            self.last_res = dict_scores.copy()

            #ログ出力
            print(f"\n[音声認識]: {text}")
            print(f"[言葉の感情スコア]: 喜び={dict_scores['Joy']}, 悲しみ={dict_scores['Sadness']}, 怒り={dict_scores['Anger']}, 恐れ={dict_scores['Fear']}, 普通={dict_scores['Neutral']}")

            # Gemini API補正
            if self.client and self.api_key and "YOUR_GEMINI_API_KEY" not in self.api_key:
                try:
                    prompt = f"分析せよ(JSON出力: Joy, Sadness, Anger, Fear, Neutral): 「{text}」"
                    response = self.client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                    llm_res = json.loads(response.text)

                    final_res = {}
                    for emo in ["Joy", "Sadness", "Anger", "Fear", "Neutral"]:
                        final_res[emo] = round(llm_res.get(emo, 0) * 0.7 + dict_scores[emo] * 0.3, 3)
                    self.last_res = final_res
                except Exception:
                    pass

        except Exception as e:
            print(f"解析処理エラー: {e}")
        finally:
            self.is_processing = False