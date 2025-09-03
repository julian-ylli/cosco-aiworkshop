import pandas as pd
import numpy as np

def calculate_recall_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    num_relevant_retrieved = len(set(relevant_ids).intersection(retrieved_at_k))
    return num_relevant_retrieved / len(relevant_ids)

def calculate_ndcg_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    idcg = 0.0

    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)

    for i, item_id in enumerate(relevant_ids[:k]): # Assuming relevant_ids are perfectly ranked
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0.0:
        return 0.0

    return dcg / idcg

def calculate_mrr_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_retrieval(experiment_df, k=3):
    recall_scores = []
    ndcg_scores = []
    mrr_scores = []

    for index, row in experiment_df.iterrows():
        relevant_ids = [int(row['id'])] # Assuming 'id' column represents the single relevant document ID
        retrieved_ids = [int(x) for x in row['retrieved_ids_at_3'].split(',')] # Assuming retrieved_ids_at_3 are stored as a comma-separated string of IDs

        recall_scores.append(calculate_recall_at_k(relevant_ids, retrieved_ids, k))
        ndcg_scores.append(calculate_ndcg_at_k(relevant_ids, retrieved_ids, k))
        mrr_scores.append(calculate_mrr_at_k(relevant_ids, retrieved_ids, k))

    return {
        f"recall@{k}": np.mean(recall_scores),
        f"ndcg@{k}": np.mean(ndcg_scores),
        f"mrr@{k}": np.mean(mrr_scores)
    }

if __name__ == "__main__":
    # Load the experiment data
    try:
        experiment_df = pd.read_csv("experiment.csv")
    except FileNotFoundError:
        print("Error: experiment.csv not found. Please ensure the file exists in the same directory.")
        exit()

    # Perform evaluation
    metrics = evaluate_retrieval(experiment_df, k=3)

    # Print results
    print("Retrieval Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
