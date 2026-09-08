import json
import threading
import datetime
import pandas as pd
from janome.tokenizer import Tokenizer
from google import genai

class TextAnalyzer:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash", csv_path='JIWC-A_2018.csv', mode="presentation"):
        self.t = Tokenizer()
        self.api_key = api_key
        self.model_name = model_name if model_name else "gemini-2.0-flash"
        self.client = None

        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Gemini Client 初期化エラー: {e}")
            
        self.last_res = {"Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0, "Neutral": 1.0}
        self.text_only_res = {"Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0, "Neutral": 1.0}
        self.last_text = "（音声入力を待っています...）"
        
        self.score = 80
        self.advice = "AI解析スキップ中（ローカル解析のみ）" if not self.client else "発話を待っています..."
        self.is_processing = False
        self.has_new_log = False
        
        self.current_mode = mode
        self.prompts = {
            "presentation": "あなたはプレゼンのプロです。聞き手に対する論理性の高さ、説得力、わかりやすさを重視して評価してください。",
            "interview": "あなたは採用面接官です。面接での適切な敬語、自信、結論ファーストで話せているかを重視して評価してください。",
            "casual": "あなたは話しやすい友人です。話の面白さ、親しみやすさ、共感度を重視して評価してください。"
        }
        
        self.emotion_dict = {}
        self.load_csv_dictionary(csv_path)

    def set_mode(self, mode):
        if mode in self.prompts:
            self.current_mode = mode

    def load_csv_dictionary(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Fear'] = df['Anxiety']
            df['Anger_Total'] = df['Anger'] + df['Disgust']
            self.emotion_dict = df.set_index('Words')[['Joy', 'Sadness', 'Anger_Total', 'Fear']].rename(columns={'Anger_Total': 'Anger'}).to_dict('index')
        except Exception:
            pass

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

            self.text_only_res = dict_scores.copy()
            self.last_res = dict_scores.copy()

            if self.client:
                try:
                    mode_instruction = self.prompts.get(self.current_mode, self.prompts["presentation"])
                    prompt = (
                        f"{mode_instruction}\n"
                        f"以下の発言を分析し、JSON形式で返してください。\n"
                        f"発言: 「{text}」\n\n"
                        f"出力フォーマット例:\n"
                        f"{{\n"
                        f'  "Joy": 0.1, "Sadness": 0.0, "Anger": 0.0, "Fear": 0.0, "Neutral": 0.9,\n'
                        f'  "score": 85,\n'
                        f'  "advice": "結論から話せていて非常にわかりやすいです。"\n'
                        f"}}\n"
                        f"※scoreは100点満点評価、adviceは30文字以内の簡潔なアドバイス。"
                    )
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                    llm_res = json.loads(response.text)

                    final_res = {}
                    for emo in ["Joy", "Sadness", "Anger", "Fear", "Neutral"]:
                        final_res[emo] = round(llm_res.get(emo, 0) * 0.7 + dict_scores[emo] * 0.3, 3)
                    self.last_res = final_res

                    self.score = llm_res.get("score", 80)
                    self.advice = llm_res.get("advice", "順調なコミュニケーションです。")

                except Exception as e:
                    print(f"Gemini解析エラー: {e}")
                    self.advice = "APIエラー（辞書解析のみ動作中）"
            else:
                self.advice = "AI解析スキップモードで動作中"

            self.has_new_log = True

        except Exception as e:
            print(f"解析エラー: {e}")
        finally:
            self.is_processing = False
