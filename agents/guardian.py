import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GuardianAgent:
    def __init__(self):
        # 1. Deterministic Rule-Based Fallback (Fast & Explicit)
        self.risk_keywords = [
            "suicide", "kill myself", "end my life", "hurt myself", 
            "die", "death", "overdose", "cutting myself", "hang myself"
        ]
        # 2. LLM Client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def check_safety_rules(self, text):
        """
        Level 1: Basic keyword scan.
        """
        text_lower = text.lower()
        for keyword in self.risk_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return True, f"Detected high-risk keyword: {keyword}"
        return False, "Safe"

    def check_safety_llm(self, text):
        """
        Level 2: LLM Semantic Analysis (High Sensitivity Mode)
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a Zero-Tolerance Safety Guardian for a mental health app. "
                            "Your ONLY job is to detect potential self-harm or suicide risk. "
                            "\n\n"
                            "RULES:"
                            "1. Flag ANY expression of hopelessness, trapped feelings, or desire to disappear as RISK."
                            "2. Flag subtle metaphors like 'I want to sleep forever' or 'The fog is swallowing me' as RISK."
                            "3. If you are even 1% unsure, choose RISK. Better to be safe than sorry."
                            "4. Ignore clearly positive metaphors (e.g., 'killing it at work' is SAFE)."
                            "\n\n"
                            "Return ONLY the word 'RISK' or 'SAFE'."
                        )
                    },
                    {"role": "user", "content": f"Entry: \"{text}\""}
                ],
                temperature=0.0,
                max_tokens=5
            )
            result = response.choices[0].message.content.strip().upper()
            
            if "RISK" in result:
                return True, "LLM Detected Contextual Risk"
            return False, "Safe"
            
        except Exception as e:
            print(f"Guardian LLM Error: {e}")
            # FAIL SAFE: If the LLM crashes, assume Risk to be safe
            return True, "Error Fallback"

    def analyze(self, text):
        # Step 1: Fast Rule Check
        is_risky, reason = self.check_safety_rules(text)
        
        # Step 2: If Rule Check passed as Safe, Double Check with LLM
        if not is_risky:
            is_risky, reason = self.check_safety_llm(text)
        
        return {
            "is_risk": is_risky,
            "reason": reason
        }