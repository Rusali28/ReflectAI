import os
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

class AnalystAgent:
    def __init__(self, use_local_model=True):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_local = use_local_model
        
        self.model_path = "./roberta/roberta_mixed_model_final"
        
        if self.use_local:
            if os.path.exists(self.model_path):
                print(f"[ANALYST] Loading Mixed RoBERTa Model from {self.model_path}...")
                self.classifier = pipeline(
                    "text-classification", 
                    model=self.model_path, 
                    top_k=None,
                    device=-1 
                )
            else:
                print(f"[ANALYST] ⚠️ Model '{self.model_path}' not found. Falling back to GPT-4o-mini.")
                self.use_local = False

    def analyze_emotions(self, text):
        """
        Returns: A string of emotions (e.g. "Joy, Gratitude")
        """
        if self.use_local:
            # --- PATH A: Mixed RoBERTa ---
            try:
                results = self.classifier(text)[0]
                detected = [res['label'] for res in results if res['score'] > 0.5]
                
                if not detected:
                    top = sorted(results, key=lambda x: x['score'], reverse=True)[0]
                    return top['label']
                
                return ", ".join(detected)
            except Exception as e:
                print(f"[ANALYST ERROR] RoBERTa failed: {e}")
                return "Neutral"

        else:
            ALLOWED_LABELS = [
                "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", 
                "Confusion", "Curiosity", "Desire", "Disappointment", "Disapproval", 
                "Disgust", "Embarrassment", "Excitement", "Fear", "Gratitude", "Grief", 
                "Joy", "Love", "Nervousness", "Optimism", "Pride", "Realization", 
                "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
            ]
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"}, 
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                f"Classify the journal entry into 1-3 emotions from this list: {ALLOWED_LABELS}. "
                                "Return JSON: {'emotions': ['Joy', 'Pride']}"
                            )
                        },
                        {"role": "user", "content": text}
                    ],
                    temperature=0.0
                )
                import json
                data = json.loads(response.choices[0].message.content)
                valid = [e for e in data.get("emotions", []) if e in ALLOWED_LABELS]
                return ", ".join(valid) if valid else "Neutral"
            except Exception as e:
                print(f"[ANALYST ERROR] GPT failed: {e}")
                return "Neutral"

    def extract_triggers(self, text):
        """
        Identifies the 'Why' behind the emotion.
        Returns a comma-separated string of specific nouns/entities.
        """
        system_prompt = (
            "You are an expert Analyst extracting keywords from a journal. "
            "Identify 2-4 specific CAUSES, NOUNS, or ENTITIES that caused the emotion. "
            "Include BOTH the broad category AND specific details mentioned in the text. "
            "Examples:\n"
            "- Input: 'My boss keeps emailing me.' -> Output: 'Work, Boss, Emails'\n"
            "- Input: 'I miss my mom.' -> Output: 'Family, Mom'\n"
            "- Input: 'Traffic was horrible.' -> Output: 'Commute, Traffic'\n"
            "Return ONLY the words separated by commas. No preamble."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except:
            return "General"