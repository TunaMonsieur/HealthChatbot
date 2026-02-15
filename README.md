# 🏥 Health Chatbot - Trợ Lý Sức Khỏe AI

Hệ thống chatbot y tế thông minh sử dụng RAG (Retrieval Augmented Generation) với khả năng tìm kiếm hai chiều: từ triệu chứng dự đoán bệnh và từ bệnh tra cứu triệu chứng.

## ✨ Tính năng chính

- **🔍 Tìm kiếm hai chiều (Bidirectional Search)**
  - **Triệu chứng → Bệnh**: "Tôi bị đau đầu, sốt cao, có thể bị bệnh gì?"
  - **Bệnh → Triệu chứng**: "Bệnh cúm có những triệu chứng gì?"

- **🤖 RAG System (Retrieval Augmented Generation)**
  - Vector search với FAISS
  - Semantic embedding (Sentence Transformers)
  - Reranking với Cross-Encoder
  - LLM generation với Google Gemini

- **💬 Giao diện thân thiện**
  - Web UI responsive
  - RESTful API với FastAPI
  - Câu trả lời tự nhiên, dễ hiểu

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Query Type Detection   │  ← Gemini LLM
│ (Symptoms/Disease)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Extract Information    │  ← Gemini LLM
│  (Symptoms/Disease)     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Embed Query           │  ← Sentence Transformers
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  FAISS Search (Top-K)   │  ← Vector Database
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Rerank Results         │  ← Cross-Encoder
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Generate Answer        │  ← Gemini LLM
│  (với context)          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

## 📋 Yêu cầu hệ thống

- Python 3.8+
- 4GB RAM trở lên
- GPU (tùy chọn, cho tốc độ tốt hơn)

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd HealthChatbot
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Đảm bảo có file `ViMedical_Disease.csv` với format:
```csv
Disease,Question
Cúm,"Tôi bị sốt cao và đau đầu, có thể bị bệnh gì?"
```

### 4. Build FAISS index

```bash
python build.py
```

Script này sẽ:
- Load dữ liệu từ CSV
- Build documents (1 disease = 1 document)
- Tạo embeddings với Sentence Transformers
- Build FAISS index
- Lưu outputs:
  - `embeddings.npy`: Vector embeddings
  - `documents.pkl`: Document metadata
  - `faiss.index`: FAISS index

### 5. Cấu hình API Key

Tạo file `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Lấy API key tại: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 6. Chạy server

```bash
python deploy.py
```

hoặc

```bash
uvicorn deploy:app --reload --port 8000
```

Server sẽ chạy tại: `http://localhost:8000`

### 7. Mở giao diện web

Mở file `index.html` trong trình duyệt, hoặc serve với:

```bash
python -m http.server 3000
```

Truy cập: `http://localhost:3000`

## � Sử dụng Docker (Khuyên dùng)

Docker giúp deploy dễ dàng hơn, không cần cài đặt Python hay dependencies.

### Yêu cầu

- Docker Desktop (Windows/Mac) hoặc Docker Engine (Linux)
- Docker Compose

### Cách 1: Chỉ dùng Docker (Backend API)

#### Bước 1: Build FAISS index (chỉ cần làm 1 lần)

```bash
# Cài dependencies và build index
pip install -r requirements.txt
python build.py
```

#### Bước 2: Tạo file .env

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Bước 3: Build và chạy container

```bash
# Build Docker image
docker build -t health-chatbot .

# Chạy container
docker run -d \
  --name health-chatbot \
  -p 8000:8000 \
  --env-file .env \
  -v ${PWD}/embeddings.npy:/app/embeddings.npy \
  -v ${PWD}/faiss.index:/app/faiss.index \
  -v ${PWD}/documents.pkl:/app/documents.pkl \
  health-chatbot
```

#### Bước 4: Kiểm tra

```bash
# Xem logs
docker logs health-chatbot

# Kiểm tra health
curl http://localhost:8000/health
```

API sẽ chạy tại: `http://localhost:8000`

### Cách 2: Dùng Docker Compose (Full Stack - Khuyên dùng)

Docker Compose sẽ chạy cả API backend + Nginx web server.

#### Bước 1: Chuẩn bị file cần thiết

```bash
# Build FAISS index (nếu chưa có)
pip install -r requirements.txt
python build.py

# Tạo file .env
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

#### Bước 2: Chạy Docker Compose

```bash
# Chạy tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Kiểm tra status
docker-compose ps
```

#### Bước 3: Truy cập

- **Web UI**: `http://localhost` (port 80)
- **API**: `http://localhost/api/`
- **Health Check**: `http://localhost/health`

#### Quản lý containers

```bash
# Dừng services
docker-compose down

# Dừng và xóa volumes
docker-compose down -v

# Rebuild sau khi thay đổi code
docker-compose up -d --build

# Restart một service
docker-compose restart healthchatbot
```

### Cách 3: Build index trong Docker (Không cần Python local)

Nếu không muốn cài Python trên máy local:

```bash
# Build container tạm để build index
docker run --rm \
  -v ${PWD}:/app \
  -w /app \
  python:3.10-slim \
  bash -c "pip install -r requirements.txt && python build.py"

# Sau đó chạy docker-compose bình thường
docker-compose up -d
```

### Production Deployment

Để deploy lên production server:

```bash
# 1. Copy files lên server
scp -r . user@server:/path/to/app

# 2. SSH vào server
ssh user@server

# 3. Chạy docker compose
cd /path/to/app
docker-compose up -d

# 4. Setup auto-restart on reboot
docker update --restart unless-stopped health-chatbot
```

### Troubleshooting Docker

**Container không start:**
```bash
docker logs health-chatbot
```

**Port đã được sử dụng:**
```bash
# Thay đổi port trong docker-compose.yml
ports:
  - "8080:8000"  # Thay 8000 thành port khác
```

**Update code:**
```bash
docker-compose down
docker-compose up -d --build
```

**Xóa tất cả và start lại:**
```bash
docker-compose down -v
docker system prune -af
docker-compose up -d --build
```

## �📖 Sử dụng

### Web UI

1. Mở `index.html` trong trình duyệt
2. Nhập câu hỏi sức khỏe
3. Nhận câu trả lời từ AI

**Ví dụ câu hỏi:**
- "Tôi bị đau đầu, sốt cao, có thể bị bệnh gì?"
- "Bệnh cúm có những triệu chứng gì?"
- "Triệu chứng của viêm phổi là gì?"

### API Endpoints

#### POST /chat

**Request:**
```json
{
  "message": "Tôi bị đau đầu và sốt cao"
}
```

**Response:**
```json
{
  "reply": "Dựa vào triệu chứng đau đầu và sốt cao...",
  "evidence": [
    {
      "disease": "Cúm",
      "score": 0.89
    }
  ]
}
```

#### GET /health

Kiểm tra trạng thái server:
```bash
curl http://localhost:8000/health
```

### Python API

```python
from retrieval_bidirectional import load_assets, load_models, search_bidirectional

# Load models
_, documents, faiss_index, disease_map = load_assets()
embed_model, reranker = load_models()

# Tìm kiếm: Triệu chứng → Bệnh
results = search_bidirectional(
    query_text="đau đầu sốt cao",
    embed_model=embed_model,
    reranker=reranker,
    faiss_index=faiss_index,
    documents=documents,
    query_type="symptoms_to_disease",
    top_k=5
)

# Tìm kiếm: Bệnh → Triệu chứng
results = search_bidirectional(
    query_text="Cúm",
    embed_model=embed_model,
    reranker=reranker,
    faiss_index=faiss_index,
    documents=documents,
    query_type="disease_to_symptoms",
    top_k=5
)
```

## 🧪 Đánh giá hệ thống

Chạy evaluation script:

```bash
python evaluate_rag_quick.py
```

Metrics được đo:
- **Top-1 Accuracy**: % queries có bệnh đúng ở vị trí #1
- **Top-3 Accuracy**: % queries có bệnh đúng trong top 3
- **MRR (Mean Reciprocal Rank)**: Vị trí trung bình của kết quả đúng
- **Average Score**: Điểm confidence trung bình

## 📁 Cấu trúc project

```
HealthChatbot/
├── build.py                      # Build FAISS index từ CSV
├── deploy.py                     # FastAPI server
├── retrieval_bidirectional.py   # Core retrieval logic
├── evaluate_rag_quick.py        # Evaluation script
├── index.html                    # Web UI
├── requirements.txt              # Dependencies
├── .env                          # API keys (tạo thủ công)
├── .env.example                  # Template cho .env
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Production multi-container
├── docker-compose.dev.yml        # Development với hot reload
├── .dockerignore                 # Docker build exclusions
├── nginx.conf                    # Nginx configuration
├── README.md                     # Documentation chính
├── DOCKER_QUICKSTART.md          # Quick start guide cho Docker
├── embeddings.npy               # Vector embeddings (generated)
├── documents.pkl                # Document metadata (generated)
└── faiss.index                  # FAISS index (generated)
```

## 🔧 Configuration

### Models

Có thể thay đổi models trong code:

**Embedding Model:**
```python
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

**Reranker:**
```python
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**LLM:**
```python
model = genai.GenerativeModel("gemini-2.5-flash")
```

### Retrieval Parameters

```python
TOP_K = 10  # Số kết quả retrieve ban đầu
RERANK_TOP = 5  # Số kết quả sau rerank
```

## 🤝 Contributing

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📝 License

Dự án này được phát hành dưới MIT License.

## ⚠️ Lưu ý quan trọng

- Đây là **hệ thống hỗ trợ tham khảo**, không thay thế chẩn đoán y tế chuyên nghiệp
- Luôn khuyên người dùng đi khám bác sĩ để được chẩn đoán chính xác
- Không sử dụng cho mục đích chẩn đoán bệnh chính thức

## 🐛 Troubleshooting

### Lỗi: Models không load được
```bash
# Thử cài đặt lại sentence-transformers
pip install --upgrade sentence-transformers
```

### Lỗi: FAISS không tìm thấy
```bash
# Đảm bảo đã chạy build.py trước
python build.py
```

### Lỗi: Gemini API
```bash
# Kiểm tra API key trong .env
# Đảm bảo đã enable Gemini API
```

### Lỗi: CORS khi dùng Web UI
```bash
# Serve index.html qua HTTP server
python -m http.server 3000
```

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub.

---

**Made with ❤️ using Python, FastAPI, FAISS, Sentence Transformers, and Google Gemini**
