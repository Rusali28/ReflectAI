import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the "Ground Truth" 
ALEX_ENTRIES = """
- Monday: Terrified about final presentation. Professor ripped apart draft.
- Wednesday: Hit a wall. Skipped class. Cried to mom.
- Friday: Presentation went well! Professor nodded. Huge relief.
- Saturday: Slept 11 hours. Pizza with friends.
"""

# 2. Define Bad Summaries to test the Judge
BAD_SUMMARIES = [
    {
        "type": "Hallucination (Faithfulness Fail)",
        "text": "This week, you went skydiving and adopted a puppy. Your professor failed you, but you didn't care because you won the lottery."
    },
    {
        "type": "Rude/Toxic (Empathy Fail)",
        "text": "You complained a lot this week about a simple presentation. You need to toughen up and stop crying to your mother. It's pathetic."
    },
    {
        "type": "Irrelevant (Actionability Fail)",
        "text": "The mitochondria is the powerhouse of the cell. Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough."
    }
]

def run_judge(summary_text):
    prompt = f"""
    You are a Senior Clinical Supervisor evaluating a Weekly Progress Note.
    
    ### SOURCE DATA:
    {ALEX_ENTRIES}
    
    ### GENERATED SUMMARY:
    {summary_text}
    
    ### SCORING RUBRIC (1-5):
    1. Faithfulness: Accurate to source?
    2. Empathy: Warm and professional?
    3. Actionability: Useful advice?
    
    Return JSON: {{ "faithfulness": 0, "empathy": 0, "actionability": 0, "reason": "..." }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    print("========================================")
    print("   ⚖️  JUDGE STRESS TEST (META-EVAL)   ")
    print("========================================")
    
    pass_count = 0
    total_cases = len(BAD_SUMMARIES)
    
    for case in BAD_SUMMARIES:
        print(f"\n🔹 Testing Case: {case['type']}")
        print(f"   Input: \"{case['text'][:60]}...\"")
        
        score = run_judge(case['text'])
        
        print(f"   scores -> Faithfulness: {score['faithfulness']}/5 | Empathy: {score['empathy']}/5")
        print(f"   Judge's Reason: {score['reason']}")
        
        if score['faithfulness'] < 3 or score['empathy'] < 3:
             print("    RESULT: PASS (Judge correctly identified the failure)")
             pass_count += 1
        else:
             print("    RESULT: FAIL (Judge gave a high score to bad content)")

    accuracy = (pass_count / total_cases) * 100
    print("   FINAL META-EVALUATION REPORT")
    print(f"   Total Cases:      {total_cases}")
    print(f"   Correct Rulings:  {pass_count}")
    print(f"   Judge Reliability:{accuracy:.1f}%")