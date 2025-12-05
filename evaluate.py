import pandas as pd
from sklearn.metrics import recall_score
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score, precision_score, accuracy_score

from agents.guardian import GuardianAgent
from agents.analyst import AnalystAgent

SAFETY_DATA_PATH = os.path.join("data", "synthetic_data/synthetic_safety.csv")
EMOTION_DATA_PATH = os.path.join("data", "synthetic_data/synthetic_emotions.csv")

print("[INFO] Loading Semantic Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def check_semantic_match(expected, predicted, threshold=0.6):
    """
    Returns True if the 'predicted' emotion is semantically close 
    to the 'expected' emotion using Vector Embeddings.
    """
    if expected.lower() in predicted.lower():
        return True
        
    # Semantic Distance
    embedding_expected = embedder.encode(expected, convert_to_tensor=True)
    embedding_predicted = embedder.encode(predicted, convert_to_tensor=True)
    #Calculate Cosine Similarity 
    score = util.cos_sim(embedding_expected, embedding_predicted).item()
    
    return score > threshold

def evaluate_safety():
    print("\n[INFO] RUNNING SAFETY EVALUATION (Risk Detection)...")
    try:
        df = pd.read_csv(SAFETY_DATA_PATH)
        df['label'] = df['label'].astype(str).str.lower() == 'true'
    except FileNotFoundError:
        print(f"[ERROR] Could not find {SAFETY_DATA_PATH}")
        return

    agent = GuardianAgent()
    y_true = []
    y_pred = []
    
    for index, row in df.iterrows():
        result = agent.analyze(row["text"])
        prediction = result["is_risk"]
        
        y_true.append(row["label"])
        y_pred.append(prediction)
        
        if prediction != row["label"]:
            error_type = "MISSED CRISIS" if row["label"] else "FALSE ALARM"
            print(f"   [FAILURE - {error_type}] Input: '{row['text'][:40]}...' | Expected: {row['label']} | Got: {prediction}")

    recall = recall_score(y_true, y_pred, pos_label=True)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    fnr = 1 - recall
    
    print(f"\n   Safety Results:")
    print(f"   - Total Cases: {len(df)}")
    print(f"   - Crisis Recall: {recall:.2%}") 
    print(f"   - False Negative Rate: {fnr:.2%}")
    print(f"   - Precision: {precision:.2%}")
    print(f"   - Overall Accuracy: {accuracy:.2%}")

def evaluate_emotions():
    print("\n[INFO] RUNNING EMOTION ACCURACY TEST (Semantic Similarity)...")
    try:
        df = pd.read_csv(EMOTION_DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {EMOTION_DATA_PATH}")
        return

    agent = AnalystAgent()
    correct = 0
    total = len(df)
    
    for index, row in df.iterrows():
        predicted_str = agent.analyze_emotions(row["text"])
        
        # Use the Embedding Matcher instead of hard-coded dictionary
        if check_semantic_match(row["expected"], predicted_str, threshold=0.55):
            correct += 1
        else:
            print(f"   [MISMATCH] Input: '{row['text'][:40]}...' | Expected: {row['expected']} | Got: {predicted_str}")
            
    accuracy = correct / total
    print(f"\n   Emotion Results:")
    print(f"   - Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    print("   ReflectAI EVALUATION SUITE           ")
    
    evaluate_safety()
    evaluate_emotions()
    
    print("   EVALUATION COMPLETE                  ")