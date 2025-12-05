import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
from database import fetch_history
from agents.coach import CoachAgent
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_judge(journal_entries, generated_summary):
    print("[JUDGE] Evaluating Summary Quality with STRICT CRITERIA...")
    
    judge_prompt = f"""
    You are a Senior Clinical Supervisor evaluating a Weekly Progress Note written by an AI trainee.
    Be CRITICAL. Do not give perfect scores easily.
    
    ### CLIENT JOURNAL DATA:
    {journal_entries}
    
    ### AI GENERATED SUMMARY:
    {generated_summary}
    
    ### SCORING RUBRIC (1.0 - 5.0):
    
    1. **Depth of Insight (1-5):** - Does the AI capture the *subtext* (e.g., masking, emotional numbness after success), or just surface-level events?
       - Penalize if it just lists events ("You took an exam").
       - Reward if it connects feelings ("You felt empty despite the success").
       
    2. **Tone Appropriateness (1-5):** - Is it overly cheerful/robotic? 
       - It should be grounded, empathetic, and realistic.
       
    3. **Evidence-Based Advice (1-5):** - Does it give generic advice ("Relax more") or specific clinical tools (CBT/DBT skills)?
       - Penalize vague platitudes.
    
    ### OUTPUT FORMAT:
    Return JSON ONLY:
    {{
      "depth_score": 0.0,
      "depth_reason": "...",
      "tone_score": 0.0,
      "tone_reason": "...",
      "advice_score": 0.0,
      "advice_reason": "..."
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge Error: {e}")
        return None

def run_evaluation():
    df = fetch_history()
    if df.empty:
        print(" Error: Run 'seed_database.py' first.")
        return

    journal_text = ""
    for _, row in df.iterrows():
        journal_text += f"- {row['date']}: {row['content']} (Stress: {row['stress_level']})\n"

    print("[SYSTEM] Generating Weekly Report...")
    coach = CoachAgent()
    summary = coach.generate_weekly_report(df)
    
    print("\n--- SUMMARY GENERATED ---")
    print(summary[:300] + "...\n")

    scores = llm_judge(journal_text, summary)
    
    if scores:
        print(" STRICT EVALUATION REPORT")
        print(f"   • Depth:  {scores['depth_score']}/5  -- {scores['depth_reason']}")
        print(f"   • Tone:   {scores['tone_score']}/5  -- {scores['tone_reason']}")
        print(f"   • Advice: {scores['advice_score']}/5  -- {scores['advice_reason']}")
        
        avg = (scores['depth_score'] + scores['tone_score'] + scores['advice_score']) / 3
        print(f"\n    OVERALL SCORE: {avg:.2f}/5.0")

if __name__ == "__main__":
    run_evaluation()