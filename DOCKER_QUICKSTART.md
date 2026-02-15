# 🚀 Quick Start với Docker

Cách nhanh nhất để chạy Health Chatbot!

## Bước 1: Chuẩn bị

```bash
# Clone repository
git clone <repository-url>
cd HealthChatbot

# Copy .env template
cp .env.example .env

# Sửa .env và thêm GEMINI_API_KEY
# Lấy API key tại: https://aistudio.google.com/apikey
```

## Bước 2: Build FAISS Index

### Option A: Build trên máy local (cần Python)

```bash
pip install -r requirements.txt
python build.py
```

### Option B: Build trong Docker (không cần Python)

**Windows PowerShell:**
```powershell
docker run --rm `
  -v ${PWD}:/app `
  -w /app `
  python:3.10-slim `
  bash -c "pip install -r requirements.txt && python build.py"
```

**Linux/Mac:**
```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  python:3.10-slim \
  bash -c "pip install -r requirements.txt && python build.py"
```

## Bước 3: Chạy với Docker Compose

```bash
docker-compose up -d
```

## Bước 4: Truy cập

- 🌐 Web UI: http://localhost
- 🔌 API: http://localhost/api/
- ✅ Health Check: http://localhost/health

## Các lệnh hữu ích

```bash
# Xem logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Update code và rebuild
docker-compose up -d --build

# Xem status
docker-compose ps
```

## Development Mode (Hot Reload)

```bash
# Sử dụng dev compose file
docker-compose -f docker-compose.dev.yml up

# Code changes sẽ tự động reload!
```

## Troubleshooting

**Lỗi "port already in use":**
```bash
# Thay port trong docker-compose.yml
ports:
  - "3000:80"  # Thay 80 thành port khác
```

**Models không load:**
```bash
# Kiểm tra file FAISS đã được tạo chưa
ls -la *.npy *.pkl *.index

# Xem logs để debug
docker-compose logs healthchatbot
```

**API key không đúng:**
```bash
# Kiểm tra .env file
cat .env

# Restart sau khi sửa .env
docker-compose restart
```

---

✅ Xong! Bạn đã có Health Chatbot chạy với Docker!
