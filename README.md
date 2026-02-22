# 🏥 Health Chatbot

AI-powered medical chatbot using RAG (Retrieval Augmented Generation) with bidirectional search: symptoms → disease and disease → symptoms.

## Features

- **Bidirectional Search**: Query by symptoms or disease name
- **RAG Pipeline**: FAISS vector search + Cross-Encoder reranking + Gemini LLM
- **Web Interface**: Clean, responsive UI with REST API

## Quick Start

### Using Docker (Recommended)

```bash
# 1. Setup environment
echo "GEMINI_API_KEY=your_api_key" > .env

# 2. Build index (one-time)
pip install -r requirements.txt && python build.py

# 3. Run
docker-compose up -d
```

Access: `http://localhost`

### Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build FAISS index
python build.py

# 3. Configure API key
echo "GEMINI_API_KEY=your_key" > .env

# 4. Start server
python deploy.py

# 5. Open index.html in browser
```

## API Usage

**Endpoint:** `POST http://localhost:8000/chat`

```json
{
  "message": "Tôi bị đau đầu và sốt cao"
}
```

**Response:**
```json
{
  "reply": "Based on symptoms...",
  "evidence": [{"disease": "Cúm", "score": 0.89}]
}
```

## Architecture

```
User Query → LLM (classify) → Extract Info → Embedding 
→ FAISS Search → Rerank → LLM (generate) → Response
```

**Tech Stack:** FastAPI, FAISS, Sentence Transformers, Google Gemini

## Project Structure

```
HealthChatbot/
├── build.py                    # Build FAISS index
├── deploy.py                   # FastAPI server
├── retrieval_bidirectional.py # Core retrieval
├── index.html                  # Web UI
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Docker Commands

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | `pip install --upgrade sentence-transformers` |
| FAISS not found | Run `python build.py` |
| API errors | Check `.env` and Gemini API key |
| CORS errors | Serve via `python -m http.server 3000` |

## Important

⚠️ This is a **reference tool only** - not for medical diagnosis. Always consult healthcare professionals.

---

**Stack:** Python • FastAPI • FAISS • Sentence Transformers • Google Gemini
