"""
Bidirectional Retrieval System for Health Chatbot
Support: Symptoms → Disease AND Disease → Symptoms
"""

import numpy as np
import pickle
import faiss
import torch
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple
from collections import defaultdict

# =========================
# CONFIG
# =========================
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDINGS_PATH = "embeddings.npy"
DOCS_PATH = "documents.pkl"
FAISS_PATH = "faiss.index"
CSV_PATH = "ViMedical_Disease.csv"

TOP_K = 10


# =========================
# Extract symptoms from questions
# =========================
def extract_symptoms_from_question(question: str) -> List[str]:
    """
    Trích xuất triệu chứng từ câu hỏi
    """
    # Remove common question patterns
    text = re.sub(r'Tôi có thể đang bị bệnh gì\??', '', question, flags=re.IGNORECASE)
    text = re.sub(r'Tôi đang bị bệnh gì\??', '', text, flags=re.IGNORECASE)
    text = re.sub(r'là bệnh gì\??', '', text, flags=re.IGNORECASE)
    text = re.sub(r'có phải là\s+\w+\s+không\??', '', text, flags=re.IGNORECASE)
    
    # Extract symptoms patterns
    patterns = [
        r'triệu chứng như ([^.?]+)',
        r'các triệu chứng như ([^.?]+)',
        r'tôi (?:hiện đang có|đang có|đang cảm thấy|cảm thấy|bị|hay) ([^.?]+)',
        r'tôi đang ([^.?]+)',
    ]
    
    symptoms = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        symptoms.extend(matches)
    
    # Clean and split
    all_symptoms = []
    for s in symptoms:
        s = s.strip()
        # Split by comma or "và"
        parts = re.split(r'[,;]|\s+và\s+', s)
        all_symptoms.extend([p.strip() for p in parts if p.strip()])
    
    return list(set(all_symptoms))  # Remove duplicates


# =========================
# Build Disease → Symptoms mapping
# =========================
def build_disease_symptom_mapping(csv_path: str = CSV_PATH) -> Dict[str, Dict]:
    """
    Tạo mapping từ bệnh sang triệu chứng
    Returns: {disease_name: {"symptoms": [...], "sample_questions": [...]}}
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    disease_map = defaultdict(lambda: {"symptoms": set(), "questions": []})
    
    for _, row in df.iterrows():
        disease = row["Disease"]
        question = row["Question"]
        
        # Extract symptoms từ question
        symptoms = extract_symptoms_from_question(question)
        
        disease_map[disease]["symptoms"].update(symptoms)
        disease_map[disease]["questions"].append(question)
    
    # Convert sets to lists and limit questions
    result = {}
    for disease, data in disease_map.items():
        result[disease] = {
            "symptoms": list(data["symptoms"]),
            "sample_questions": data["questions"][:5],  # Keep first 5 as examples
            "symptom_count": len(data["symptoms"]),
            "question_count": len(data["questions"])
        }
    
    return result


# =========================
# Load assets
# =========================
def load_assets():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    index = faiss.read_index(FAISS_PATH)
    
    # Build disease-symptom mapping
    disease_map = build_disease_symptom_mapping(CSV_PATH)
    
    return embeddings, documents, index, disease_map


# =========================
# Load models
# =========================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Device: {device}")

    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    reranker = CrossEncoder(RERANK_MODEL, device=device)

    return embed_model, reranker


# =========================
# Query direction detection
# =========================
def detect_query_direction(query: str) -> str:
    """
    Phát hiện hướng truy vấn: symptoms→disease hoặc disease→symptoms
    Returns: "symptom_to_disease" hoặc "disease_to_symptom"
    """
    query_lower = query.lower()
    
    # Patterns cho disease → symptoms
    disease_to_symptom_patterns = [
        r'triệu chứng của\s+(\w+)',
        r'(\w+)\s+có triệu chứng gì',
        r'bệnh\s+(\w+)\s+(?:có|biểu hiện|triệu chứng)',
        r'các triệu chứng của bệnh\s+(\w+)',
        r'(\w+)\s+biểu hiện như thế nào',
        r'dấu hiệu của\s+(\w+)',
        r'nhận biết\s+(\w+)',
    ]
    
    for pattern in disease_to_symptom_patterns:
        if re.search(pattern, query_lower):
            return "disease_to_symptom"
    
    # Default: symptom → disease
    return "symptom_to_disease"


def detect_query_type(query: str) -> str:
    """
    Alias for detect_query_direction, returns standardized format
    Returns: "symptoms_to_disease" or "disease_to_symptoms"
    """
    direction = detect_query_direction(query)
    # Normalize to match deploy.py expected format
    if direction == "disease_to_symptom":
        return "disease_to_symptoms"
    else:
        return "symptoms_to_disease"


# =========================
# Search: Symptoms → Disease (Original)
# =========================
def search_disease_from_symptoms(query: str, embed_model, reranker, index, documents, top_k=TOP_K):
    """
    Tìm bệnh từ triệu chứng (hướng ban đầu)
    """
    # FAISS retrieval
    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    _, indices = index.search(q_emb, top_k)
    candidates = [documents[idx] for idx in indices[0]]

    # Cross-encoder rerank
    pairs = [(query, doc["text"]) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        {
            "score": float(score),
            "disease": doc["metadata"]["disease"],
            "samples": doc["metadata"]["num_samples"],
            "type": "symptom_to_disease"
        }
        for score, doc in ranked
    ]


# =========================
# Search: Disease → Symptoms (New)
# =========================
def search_symptoms_from_disease(query: str, disease_map: Dict) -> List[Dict]:
    """
    Tìm triệu chứng từ tên bệnh
    """
    query_lower = query.lower()
    
    # Extract disease name from query
    disease_name = None
    for disease in disease_map.keys():
        if disease.lower() in query_lower:
            disease_name = disease
            break
    
    # If not found, fuzzy match
    if not disease_name:
        # Simple fuzzy matching
        for disease in disease_map.keys():
            # Check if any word in query matches disease name
            query_words = set(query_lower.split())
            disease_words = set(disease.lower().split())
            
            if query_words & disease_words:
                disease_name = disease
                break
    
    if disease_name and disease_name in disease_map:
        data = disease_map[disease_name]
        return [{
            "disease": disease_name,
            "symptoms": data["symptoms"],
            "symptom_count": data["symptom_count"],
            "sample_questions": data["sample_questions"],
            "total_questions": data["question_count"],
            "type": "disease_to_symptom"
        }]
    
    # If still not found, return similar diseases
    results = []
    query_words = set(query_lower.split())
    
    for disease, data in disease_map.items():
        disease_words = set(disease.lower().split())
        overlap = len(query_words & disease_words)
        
        if overlap > 0:
            results.append({
                "disease": disease,
                "symptoms": data["symptoms"][:10],  # Limit to 10 symptoms
                "symptom_count": data["symptom_count"],
                "match_score": overlap / max(len(query_words), len(disease_words)),
                "type": "disease_to_symptom_fuzzy"
            })
    
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:5]  # Return top 5 matches


# =========================
# Unified Bidirectional Search
# =========================
def search_bidirectional(query: str, embed_model, reranker, index, documents, 
                        query_type: str = None, disease_map: Dict = None, top_k=TOP_K) -> List[Dict]:
    """
    Tìm kiếm hai chiều
    Args:
        query: Query string
        query_type: "symptoms_to_disease" or "disease_to_symptoms" (optional, will auto-detect if None)
        disease_map: Disease mapping dict (optional, will build if None)
    Returns: List of results
    """
    # Auto-detect direction if not specified
    if query_type is None:
        direction = detect_query_direction(query)
    else:
        # Convert from deploy.py format to internal format
        if query_type == "disease_to_symptoms":
            direction = "disease_to_symptom"
        else:
            direction = "symptom_to_disease"
    
    # Build disease_map if not provided
    if disease_map is None and direction == "disease_to_symptom":
        disease_map = build_disease_symptom_mapping(CSV_PATH)
    
    if direction == "disease_to_symptom":
        results = search_symptoms_from_disease(query, disease_map)
        # Format results to match expected output
        formatted_results = []
        for r in results:
            # Get first disease match
            formatted_results = [
                {"symptom": symptom, "disease": r["disease"], "score": r.get("match_score", 1.0)}
                for symptom in r["symptoms"]
            ]
            break  # Only use first match
        return formatted_results
    else:
        results = search_disease_from_symptoms(query, embed_model, reranker, 
                                              index, documents, top_k)
        return results


def search_bidirectional_legacy(query: str, embed_model, reranker, index, documents, 
                        disease_map: Dict, top_k=TOP_K) -> Tuple[str, List[Dict]]:
    """
    Legacy version that returns (direction, results)
    """
    direction = detect_query_direction(query)
    
    if direction == "disease_to_symptom":
        results = search_symptoms_from_disease(query, disease_map)
    else:
        results = search_disease_from_symptoms(query, embed_model, reranker, 
                                              index, documents, top_k)
    
    return direction, results


# =========================
# Format output
# =========================
def format_results(direction: str, results: List[Dict]) -> str:
    """
    Format kết quả theo hướng truy vấn
    """
    output = []
    
    if direction == "disease_to_symptom":
        if not results:
            return "❌ Không tìm thấy bệnh này trong cơ sở dữ liệu."
        
        for i, r in enumerate(results, 1):
            output.append(f"\n{'='*60}")
            output.append(f"🏥 Bệnh: {r['disease']}")
            output.append(f"📋 Số lượng triệu chứng: {r['symptom_count']}")
            
            if "match_score" in r:
                output.append(f"🎯 Độ khớp: {r['match_score']:.2%}")
            
            output.append(f"\n💊 Các triệu chứng chính:")
            for j, symptom in enumerate(r['symptoms'][:15], 1):
                output.append(f"   {j}. {symptom}")
            
            if len(r['symptoms']) > 15:
                output.append(f"   ... và {len(r['symptoms']) - 15} triệu chứng khác")
            
            if "sample_questions" in r:
                output.append(f"\n📝 Ví dụ câu hỏi từ bệnh nhân:")
                for j, q in enumerate(r['sample_questions'][:3], 1):
                    output.append(f"   {j}. {q[:80]}...")
    
    else:  # symptom_to_disease
        output.append(f"\n{'='*60}")
        output.append("🔍 KẾT QUẢ TÌM KIẾM BỆNH TỪ TRIỆU CHỨNG")
        output.append(f"{'='*60}")
        
        for i, r in enumerate(results[:10], 1):
            output.append(f"\n{i}. 🏥 {r['disease']}")
            output.append(f"   📊 Độ khớp: {r['score']:.4f}")
            output.append(f"   📋 Số mẫu: {r['samples']}")
    
    return "\n".join(output)


# =========================
# DEMO
# =========================
if __name__ == "__main__":
    print("🚀 Loading Bidirectional Retrieval System...")
    
    # Load models and data
    _, documents, index = load_assets()
    embed_model, reranker = load_models()
    
    print("\n📊 Building disease → symptom mapping...")
    disease_map = build_disease_symptom_mapping()
    print(f"   ✅ Loaded {len(disease_map)} diseases")
    
    print("\n" + "="*70)
    print("💬 BIDIRECTIONAL HEALTH CHATBOT")
    print("="*70)
    
    # Example 1: Symptoms → Disease
    print("\n" + "="*70)
    print("📝 EXAMPLE 1: TÌM BỆNH TỪ TRIỆU CHỨNG")
    print("="*70)
    
    query1 = """
    Tôi hiện đang có các triệu chứng như rụng tóc,
    da sạm màu và kinh nguyệt thưa dần.
    Tôi có thể đang bị bệnh gì?
    """
    
    print(f"\n❓ Query: {query1.strip()}")
    direction, results = search_bidirectional(query1, embed_model, reranker, 
                                             index, documents, disease_map)
    print(f"\n🎯 Detected direction: {direction}")
    print(format_results(direction, results))
    
    # Example 2: Disease → Symptoms  
    print("\n\n" + "="*70)
    print("📝 EXAMPLE 2: TÌM TRIỆU CHỨNG TỪ BỆNH")
    print("="*70)
    
    query2 = "Bệnh Alzheimer có triệu chứng gì?"
    
    print(f"\n❓ Query: {query2}")
    direction, results = search_bidirectional(query2, embed_model, reranker, 
                                             index, documents, disease_map)
    print(f"\n🎯 Detected direction: {direction}")
    print(format_results(direction, results))
    
    # Example 3: Another disease query
    print("\n\n" + "="*70)
    print("📝 EXAMPLE 3: TÌM TRIỆU CHỨNG TỪ BỆNH KHÁC")
    print("="*70)
    
    query3 = "Triệu chứng của bệnh tiểu đường"
    
    print(f"\n❓ Query: {query3}")
    direction, results = search_bidirectional(query3, embed_model, reranker, 
                                             index, documents, disease_map)
    print(f"\n🎯 Detected direction: {direction}")
    print(format_results(direction, results))
    
    print("\n" + "="*70)
    print("✨ Demo complete! System supports both directions.")
    print("="*70)
