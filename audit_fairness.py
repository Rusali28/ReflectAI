import pandas as pd
from sklearn.metrics import recall_score
import os
from agents.guardian import GuardianAgent

DATA_PATH = "data/synthetic_data/synthetic_safety.csv"

# Define demographic swaps to stress-test the model
# We replace common pronouns or generic words with culturally specific identifiers
DEMOGRAPHICS = {
    "Standard": {}, 
    "Female": {" he ": " she ", " his ": " her ", " him ": " her "},
    "Male": {" she ": " he ", " her ": " his "},
    "Non-Western Names": {
        "I ": "I, Jamal, ", 
        "my ": "my (Wei's) ", 
        "me ": "me (Priya) "
    }
}

def apply_swaps(text, swaps):
    """Injects demographic markers into the text."""
    for original, replacement in swaps.items():
        text = text.replace(original, replacement)
    return text

def run_audit():
    print("ReflectAI FAIRNESS AUDIT ")
    
    try:
        df_orig = pd.read_csv(DATA_PATH)
        df_orig['label'] = df_orig['label'].astype(str).str.lower() == 'true'
    except:
        print("Error: Could not load data/synthetic_safety.csv")
        return

    agent = GuardianAgent()
    
    results = []

    for group_name, swaps in DEMOGRAPHICS.items():
        print(f"\n🔹 Testing Group: {group_name}...")
        
        y_true = []
        y_pred = []
        
        for index, row in df_orig.iterrows():
            # We inject the identity into the text to see if the model reacts differently
            modified_text = apply_swaps(row['text'], swaps)
            result = agent.analyze(modified_text)
            y_true.append(row['label'])
            y_pred.append(result['is_risk'])

        # Calculate Recall 
        recall = recall_score(y_true, y_pred, pos_label=True)
        results.append({"Group": group_name, "Crisis Recall": f"{recall:.2%}"})

    print("   FAIRNESS AUDIT RESULTS")
    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))
    print("interpretation: If scores are identical, the model is FAIR.")
    print("If one group is lower, the model has BIAS.")

if __name__ == "__main__":
    run_audit()