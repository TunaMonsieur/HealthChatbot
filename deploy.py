from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from retrieval_bidirectional import load_assets, load_models, search_bidirectional, detect_query_type

# Load environment variables
load_dotenv()

app = FastAPI(title="Health Chatbot API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Config constants
# =========================
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH = "faiss.index"
DOCUMENTS_PATH = "documents.pkl"

# =========================
# Global loaded objects
# =========================
embed_model = None
reranker = None
faiss_index = None
documents = None
disease_map = None  # Added for bidirectional support

# =========================
# Configure Gemini
# =========================
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB9N-bcCq8NADKKfA6Hena90txX790ewjU")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

if not API_KEY or API_KEY == "your_api_key_here":
    print("⚠️  WARNING: Please set GEMINI_API_KEY in .env file")
    print("   Get your API key at: https://aistudio.google.com/apikey")


# =========================
# Request/Response Models
# =========================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    evidence: list


# =========================
# Startup: Load models once
# =========================
@app.on_event("startup")
async def startup_event():
    global embed_model, reranker, faiss_index, documents, disease_map
    print("🚀 Loading models and indexes...")
    _, documents, faiss_index, disease_map = load_assets()
    embed_model, reranker = load_models()
    print("✅ Models loaded successfully!")


# =========================
# Helper functions
# =========================
def detect_query_intent(user_message: str) -> str:
    """Detect if user is asking symptoms->disease or disease->symptoms"""
    return detect_query_type(user_message)


def extract_symptoms_llm(user_message: str) -> list:
    """Extract symptoms from user message using Gemini"""
    prompt = f"""
    Bạn là trợ lý y tế. Hãy đọc câu hỏi của người dùng và liệt kê các triệu chứng sức khỏe họ đang gặp.
    Chỉ liệt kê triệu chứng, mỗi dòng một triệu chứng, ngắn gọn.
    
    Câu hỏi: {user_message}
    
    Các triệu chứng:
    """
    
    response = model.generate_content(prompt)
    symptoms = [s.strip() for s in response.text.strip().split('\n') if s.strip()]
    return symptoms


def extract_disease_llm(user_message: str) -> str:
    """Extract disease name from user message using Gemini"""
    prompt = f"""
    Bạn là trợ lý y tế. Hãy đọc câu hỏi của người dùng và trích xuất tên bệnh họ đang hỏi.
    Chỉ trả lời tên bệnh, ngắn gọn, không giải thích.
    
    Câu hỏi: {user_message}
    
    Tên bệnh:
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()


def retrieve_bidirectional(query_text: str, query_type: str) -> list:
    """Retrieve results using bidirectional system"""
    results = search_bidirectional(
        query_text, 
        embed_model, 
        reranker, 
        faiss_index, 
        documents, 
        query_type=query_type,
        top_k=5
    )
    return results


def generate_medical_answer(user_message: str, query_type: str, retrieved_info: list, extracted_info: any) -> str:
    """Generate final answer using Gemini with retrieved context"""
    
    if query_type == "symptoms_to_disease":
        evidence_text = "\n".join([
            f"- {d['disease']}"
            for d in retrieved_info[:3]
        ])
        
        prompt = f"""
        Bạn là trợ lý sức khỏe thân thiện, nói chuyện tự nhiên như một người bạn quan tâm.
        
        Người dùng hỏi: {user_message}
        
        Triệu chứng: {', '.join(extracted_info)}
        
        Theo dữ liệu y tế, các bệnh có thể liên quan:
        {evidence_text}
        
        Hãy trả lời theo phong cách:
        - Nói chuyện tự nhiên, thân thiện như đang tư vấn trực tiếp
        - Dùng "mình", "bạn" thay vì "tôi", "bệnh nhân"
        - Giải thích đơn giản, dễ hiểu
        - Thể hiện sự quan tâm chân thành
        - Nhắc nhở nên đi khám bác sĩ để chẩn đoán chính xác
        - Không liệt kê quá nhiều bệnh, chỉ tập trung vào 1-2 khả năng chính
        - Độ dài vừa phải (3-5 câu)
        """
    else:  # disease_to_symptoms
        evidence_text = "\n".join([
            f"- {s['symptom']}"
            for s in retrieved_info[:5]
        ])
        
        prompt = f"""
        Bạn là trợ lý sức khỏe thân thiện, nói chuyện tự nhiên như một người bạn quan tâm.
        
        Người dùng hỏi về bệnh: {extracted_info}
        Câu hỏi gốc: {user_message}
        
        Các triệu chứng phổ biến của bệnh {extracted_info}:
        {evidence_text}
        
        Hãy trả lời theo phong cách:
        - Nói chuyện tự nhiên, thân thiện
        - Giải thích các triệu chứng đơn giản, dễ hiểu
        - Nhóm các triệu chứng liên quan lại với nhau
        - Thể hiện sự quan tâm
        - Khuyên nên đi khám nếu có các triệu chứng này
        - Độ dài vừa phải (4-6 câu)
        """
    
    response = model.generate_content(prompt)
    return response.text


# =========================
# API Endpoints
# =========================
@app.get("/")
async def root():
    return {
        "message": "Health Chatbot API",
        "status": "running",
        "endpoints": {
            "POST /chat": "Send a health query",
            "GET /health": "Check API health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": embed_model is not None and reranker is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with bidirectional support"""
    try:
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Detect query type
        query_type = detect_query_intent(user_message)
        print(f"🔍 Query type: {query_type}")
        
        if query_type == "symptoms_to_disease":
            # Extract symptoms
            symptoms = extract_symptoms_llm(user_message)
            print(f"📋 Extracted symptoms: {symptoms}")
            
            if len(symptoms) < 1:
                return ChatResponse(
                    reply="Chào bạn! Mình muốn hỗ trợ bạn nhưng cần biết thêm về triệu chứng bạn đang gặp phải. Bạn có thể kể cụ thể hơn về cảm giác khó chịu hay những dấu hiệu bất thường nào không?",
                    evidence=[]
                )
            
            # Retrieve diseases
            symptoms_query = " ".join(symptoms)
            results = retrieve_bidirectional(symptoms_query, "symptoms_to_disease")
            print(f"🔍 Found {len(results)} diseases")
            
            # Generate answer
            reply = generate_medical_answer(user_message, query_type, results, symptoms)
            
            return ChatResponse(
                reply=reply,
                evidence=results[:5]
            )
        
        else:  # disease_to_symptoms
            # Extract disease name
            disease_name = extract_disease_llm(user_message)
            print(f"🏥 Extracted disease: {disease_name}")
            
            if not disease_name:
                return ChatResponse(
                    reply="Bạn muốn hỏi về triệu chứng của bệnh nào? Hãy cho mình biết tên bệnh để mình có thể giúp bạn nhé!",
                    evidence=[]
                )
            
            # Retrieve symptoms
            results = retrieve_bidirectional(disease_name, "disease_to_symptoms")
            print(f"🔍 Found {len(results)} symptoms")
            
            # Generate answer
            reply = generate_medical_answer(user_message, query_type, results, disease_name)
            
            return ChatResponse(
                reply=reply,
                evidence=results[:5]
            )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Run server
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
