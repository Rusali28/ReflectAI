import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from openai import OpenAI
from dotenv import load_dotenv
from RAG.rag_engine import retrieve_context, initialize_knowledge_base

load_dotenv()

class CoachAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Ensure the knowledge base is ready when the Coach starts
        initialize_knowledge_base()

    def _basic_safety_check(self, text):
        """
        Fallback Layer: Fast keyword scan if the LLM fails.
        """
        banned_phrases = [
            "kill yourself", "suicide is the answer", "stop taking medication",
            "don't call a doctor", "hurt yourself", "end it all", "give up"
        ]
        text_lower = text.lower()
        for phrase in banned_phrases:
            if phrase in text_lower:
                return True # Unsafe
        return False # Safe

    def _intelligent_safety_check(self, response_text):
        """
        Primary Layer: LLM-based Safety Auditor.
        Reads context to distinguish between "Don't stop meds" (Safe) and "Stop meds" (Unsafe).
        """
        try:
            check_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a Safety Auditor for a mental health app. "
                            "Analyze the text below. "
                            "Flag it as UNSAFE if it contains:\n"
                            "1. Encouragement of self-harm or suicide.\n"
                            "2. Instructions to stop prescribed medication without a doctor.\n"
                            "3. Specific medical diagnoses (e.g., 'You have Bipolar Disorder').\n\n"
                            "Reply ONLY with 'SAFE' or 'UNSAFE'."
                        )
                    },
                    {"role": "user", "content": f"Agent Response to Audit: \"{response_text}\""}
                ],
                temperature=0.0
            )
            verdict = check_response.choices[0].message.content.strip().upper()
            
            if "UNSAFE" in verdict:
                return False, "⚠️ **SAFETY ALERT**\n\nThe generated response was blocked by our Safety Monitor because it may contain unsafe medical or crisis-related advice. Please consult a professional."
            
            return True, response_text
            
        except Exception as e:
            print(f"[Safety Audit Error] {e} - Falling back to keywords.")
            # Fallback to keywords
            if self._basic_safety_check(response_text):
                return False, "⚠️ **SAFETY ALERT** (Fallback)\n\nContent blocked due to safety keywords."
            return True, response_text

    def generate_weekly_report(self, entries_df):
        if entries_df.empty:
            return "No entries found for this week."

        # 1. RAG Setup
        combined_text = " ".join(entries_df['content'].tolist())
        retrieved_tips = retrieve_context(combined_text, n_results=2)
        knowledge_block = "\n\n".join(retrieved_tips)

        # 2. The "Deep Insight" Prompt
        # We explicitly tell it to look for masking and contradictions.
        system_prompt = (
            "You are a highly empathetic, clinical-grade Mental Health Coach. "
            "Review the user's journal and provide deep, personalized insights.\n\n"
            
            "VALIDATED STRATEGIES (Reference these):\n"
            "=== START LIBRARY ===\n"
            f"{knowledge_block}\n"
            "=== END LIBRARY ===\n\n"
            
            "INSTRUCTIONS:\n"
            "1. **Deep Analysis**: Do not just summarize events. Analyze the *emotional arc*. "
            "Look for contradictions (e.g., success but feeling empty, socializing but feeling lonely, or 'masking' their true feelings).\n"
            "2. **Pattern Recognition**: Connect specific behaviors (sleep, avoidance, caffeine) to their mood shifts.\n"
            "3. **Actionable Advice**: Suggest specific steps using the Library strategies above. Explain *why* they fit.\n"
            "4. **Tone**: Be warm, validating, and human. Avoid robotic phrases. "
            "Acknowledge how hard the week was, even if it ended well."
        )

        try:
            user_context = ""
            for index, row in entries_df.iterrows():
                user_context += (
                    f"- {row['date']}:\n"
                    f"  Text: {row['content']}\n"
                    f"  Mood: {row['emotions']}\n"
                    f"  Triggers: {row['triggers']}\n"
                    f"  Stats: Sleep {row['sleep_hours']}h, Stress {row['stress_level']}/10\n"
                    "----------------\n"
                )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Entries:\n{user_context}"}
                ],
            )
            raw_output = response.choices[0].message.content
            
            # 3. Guardrail
            is_safe, final_output = self._intelligent_safety_check(raw_output)
            
            return final_output

        except Exception as e:
            return f"Error generating insight: {e}"