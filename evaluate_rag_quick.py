"""
Quick RAG Evaluation Demo - Small Sample
"""

import numpy as np
import pandas as pd
from retrieval_bidirectional import load_assets, load_models, search

# Load test data
print("📂 Loading test data...")
df = pd.read_csv("ViMedical_Disease.csv")
df = df.drop_duplicates().dropna()

# Take 50 random samples for quick evaluation
test_sample = df.sample(n=50, random_state=42)
print(f"   Test size: {len(test_sample)}")
print(f"   Unique diseases: {test_sample['Disease'].nunique()}")

# Load models
print("\n🤖 Loading models and index...")
_, documents, index = load_assets()
embed_model, reranker = load_models()

# Metrics
metrics = {
    "top_1_correct": 0,
    "top_3_correct": 0,
    "top_5_correct": 0,
    "mrr_sum": 0.0,
    "avg_score": []
}

print("\n🔍 Running evaluation...")
for idx, row in test_sample.iterrows():
    query = row["Question"]
    ground_truth = row["Disease"]
    
    # Get results
    results = search(query, embed_model, reranker, index, documents, top_k=10)
    retrieved_diseases = [r["disease"] for r in results]
    
    # Calculate metrics
    if ground_truth in retrieved_diseases[:1]:
        metrics["top_1_correct"] += 1
    if ground_truth in retrieved_diseases[:3]:
        metrics["top_3_correct"] += 1
    if ground_truth in retrieved_diseases[:5]:
        metrics["top_5_correct"] += 1
    
    # MRR
    for i, disease in enumerate(retrieved_diseases, 1):
        if disease == ground_truth:
            metrics["mrr_sum"] += 1.0 / i
            break
    
    # Average score of top result
    if results:
        metrics["avg_score"].append(results[0]["score"])
    
    if (idx + 1) % 10 == 0:
        print(f"   ✓ Processed {idx + 1} queries")

# Calculate averages
print("\n" + "="*60)
print("📊 RAG EVALUATION RESULTS (50 queries)")
print("="*60)

total = len(test_sample)

print(f"\n🎯 Accuracy Metrics:")
print(f"   Top-1 Accuracy: {metrics['top_1_correct'] / total:.4f} ({metrics['top_1_correct']}/{total})")
print(f"   Top-3 Accuracy: {metrics['top_3_correct'] / total:.4f} ({metrics['top_3_correct']}/{total})")
print(f"   Top-5 Accuracy: {metrics['top_5_correct'] / total:.4f} ({metrics['top_5_correct']}/{total})")

print(f"\n📍 Retrieval Metrics:")
print(f"   Mean Reciprocal Rank (MRR): {metrics['mrr_sum'] / total:.4f}")
print(f"   Average Top-1 Score: {np.mean(metrics['avg_score']):.4f}")
print(f"   Score Std Dev: {np.std(metrics['avg_score']):.4f}")

print(f"\n💡 Interpretation:")
print(f"   - Top-1 Accuracy: {metrics['top_1_correct'] / total * 100:.1f}% of queries have correct disease at rank 1")
print(f"   - Top-3 Accuracy: {metrics['top_3_correct'] / total * 100:.1f}% of queries have correct disease in top 3")
print(f"   - MRR of {metrics['mrr_sum'] / total:.4f} means on average the correct result appears at position {1/(metrics['mrr_sum']/total):.1f}")

# Show some examples
print("\n" + "="*60)
print("🔬 EXAMPLE QUERIES AND RESULTS")
print("="*60)

for i, (idx, row) in enumerate(test_sample.head(3).iterrows()):
    query = row["Question"]
    ground_truth = row["Disease"]
    
    print(f"\n📌 Example {i+1}:")
    print(f"   Query: {query[:80]}...")
    print(f"   Ground Truth: {ground_truth}")
    
    results = search(query, embed_model, reranker, index, documents, top_k=5)
    print(f"   Retrieved:")
    for j, r in enumerate(results, 1):
        marker = "✅" if r["disease"] == ground_truth else "  "
        print(f"     {marker} {j}. {r['disease']} (score: {r['score']:.3f})")

print("\n" + "="*60)
print("✨ Quick evaluation complete!")
print("="*60)
