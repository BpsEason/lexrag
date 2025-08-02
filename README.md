# LexRAG – 法律文件智慧檢索樣板

LexRAG 是一套開源的 RAG（Retrieval-Augmented Generation）樣板，專為法律領域設計。使用者可透過自然語言查詢法規、判決書等文件，獲得可信、含出處的回應。

## 🔧 技術架構
- **後端**: FastAPI RESTful API + SSE 流式回應
- **核心**: LangChain + Chroma 向量庫
- **模型**: Mistral-7b 模型（支援 GGUF 量化）
- **部署**: Docker Compose 一鍵部署
- **監控**: Prometheus + Grafana

## 🚀 快速啟動
1. 確保 Docker 服務正在運行。
2. 將法律文件 (PDF, DOCX, Markdown) 放入 `documents/` 目錄中。
3. 執行 `./start.sh` 腳本以啟動所有服務。

## 📚 使用案例
- 詢問法條：「請問勞動基準法第14條的內容？」
- 檢索判決書：「有關性騷擾的最高法院判決？」
