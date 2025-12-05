import pandas as pd
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agents.analyst import AnalystAgent
from sentence_transformers import SentenceTransformer, util

# LOAD EMBEDDING MODEL
embedder = SentenceTransformer('all-MiniLM-L6-v2')

DATASETS = [
    {"name": "GoEmotions (Reddit)", "path": "data/real_data/real_goemotions.csv"},
    {"name": "Vent App (Venting)", "path": "data/real_data/real_vent.csv"},
    {"name": "ISEAR (Journals)",   "path": "data/real_data/real_isear.csv"}
]

def check_match(expected, predicted, threshold=0.55):
    expected = str(expected).lower().strip()
    predicted = str(predicted).lower().strip()
    
    # 1. Exact Substring Match
    if expected in predicted: 
        return True
        
    # 2. Semantic Similarity
    emb1 = embedder.encode(expected, convert_to_tensor=True)
    emb2 = embedder.encode(predicted, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() > threshold

def evaluate_dataset(dataset_name, file_path, roberta_agent, gpt_agent):
    print(f"\n📂 Evaluating on: {dataset_name}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"   ⚠️ File not found: {file_path}. Skipping.")
        return None

    scores = {"roberta": 0, "gpt": 0}
    
    for index, row in df.iterrows():
        text = str(row["text"])
        truth = str(row["expected"])
        
        # Test RoBERTa
        pred_rob_tuple = roberta_agent.analyze_emotions(text)
        pred_rob = pred_rob_tuple[0] if isinstance(pred_rob_tuple, tuple) else pred_rob_tuple
        
        if check_match(truth, pred_rob):
            scores["roberta"] += 1
            
        # Test GPT
        pred_gpt_tuple = gpt_agent.analyze_emotions(text)
        pred_gpt = pred_gpt_tuple[0] if isinstance(pred_gpt_tuple, tuple) else pred_gpt_tuple
        
        if check_match(truth, pred_gpt):
            scores["gpt"] += 1
            
    acc_rob = (scores["roberta"] / len(df)) * 100
    acc_gpt = (scores["gpt"] / len(df)) * 100
    
    return acc_rob, acc_gpt

if __name__ == "__main__":
    print("   ReflectAI GRAND ABLATION STUDY       ")

    print("...Loading Models...")
    roberta_agent = AnalystAgent(use_local_model=True)
    gpt_agent = AnalystAgent(use_local_model=False)

    results = []

    for ds in DATASETS:
        result = evaluate_dataset(ds["name"], ds["path"], roberta_agent, gpt_agent)
        
        if result is None:
            continue
            
        acc_r, acc_g = result
        results.append({
            "Dataset": ds["name"],
            "RoBERTa": f"{acc_r:.1f}%",
            "GPT-4o": f"{acc_g:.1f}%",
            "Gap": f"{acc_g - acc_r:.1f}%"
        })

    print("   FINAL CROSS-DOMAIN RESULTS")
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_markdown(index=False))