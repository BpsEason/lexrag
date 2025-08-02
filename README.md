# LexRAG – 法律文件智慧檢索樣板

LexRAG 是一套專為法律領域打造的 Retrieval-Augmented Generation (RAG) 範本。它將散落於 PDF、DOCX、Markdown 等非結構化法律文件向量化並編入 ChromaDB，結合大型語言模型（LLM）與檢索引擎，讓使用者能以自然語言發問，快速獲得含有明確出處的高可信度法律問答。

## 主要特色

  * **一鍵攝取多格式文件**
    支援 PDF、DOCX、Markdown，利用 LangChain 載入與拆段，輕鬆整合各渠道法規與判決書。

  * **高度可信溯源**
    回應內標註來源檔名、段落編號，確保每個答案都能追溯回原始文件。

  * **即時流式回應**
    採用 FastAPI + SSE，使用者在 CLI 或前端介面中可邊生成邊閱讀結果。

  * **容器化部署**
    使用 Docker Compose 一鍵啟動五大服務（API、向量庫、LLM、Prometheus、Grafana），減輕部署複雜度。

  * **完整 CI/CD 與測試**
    提供 GitHub Actions 範例流程與 pytest 測試套件，確保專案品質可長期維護。

## 技術架構

這張圖展示了 LexRAG 的資料流與各服務間的交互關係。我特別加入了「使用者/系統管理者」節點，清楚標示出資料攝取的發起者。

```mermaid
graph TD
    subgraph "資料攝取與檢索"
        direction LR
        A[documents<br>PDF/MD/DOCX] -- 載入 --> B(LangChain<br>loaders / splitter)
        B -- 向量化與寫入 --> C[ChromaDB<br>向量庫]
    end
    
    subgraph "RAG 檢索流程"
        direction LR
        D[FastAPI<br>+ SSE] -- 查詢請求 --> E{Retrieval<br>+ LLM}
        E -- 檢索 --> C
        E -- 生成答案 --> G[LLM Service<br>vLLM]
        G -- 回應 --> D
    end

    subgraph "使用者介面與監控"
        direction LR
        H[Frontend<br>/ cURL] -- 查詢 --> D
        D -- 監控指標 --> I[Prometheus]
        G -- 監控指標 --> I
        I -- 視覺化 --> J[Grafana]
    end
    
    K[使用者/系統管理者] -- 執行攝取指令 --> B
    E --> C
    
    style K fill:#FFC107,stroke:#333,stroke-width:2px,color:#000
    style A fill:#D4E7F4,stroke:#333,stroke-width:2px,color:#1A237E
    style B fill:#E0F7FA,stroke:#333,stroke-width:2px,color:#1A237E
    style C fill:#90CAF9,stroke:#333,stroke-width:2px,color:#1A237E
    style D fill:#C8E6C9,stroke:#333,stroke-width:2px,color:#1A237E
    style E fill:#FFF9C4,stroke:#333,stroke-width:2px,color:#1A237E
    style G fill:#B39DDB,stroke:#333,stroke-width:2px,color:#1A237E
    style H fill:#E1F5FE,stroke:#333,stroke-width:2px,color:#1A237E
    style I fill:#FFCCBC,stroke:#333,stroke-width:2px,color:#1A237E
    style J fill:#FFECB3,stroke:#333,stroke-width:2px,color:#1A237E
```

  * **文件攝取 (手動觸發)**：使用者或系統管理者透過執行指令（例如 `./start.sh` 或手動呼叫 `ingest_documents()` 函數），啟動資料攝取流程。此時，系統會讀取 `documents` 目錄中的文件，經由 `LangChain document loaders` 載入，並將其內容及 `metadata` 轉換成向量後，**寫入** `ChromaDB`。
  * **RAG 檢索 (自動化)**：當使用者透過前端或 cURL 提交查詢時，`FastAPI` 會自動啟動 RAG 檢索流程。它會透過 `ChromaDB Retriever` **從 `ChromaDB` 讀取**與使用者問題最相關的文件區塊，再將這些上下文與原始問題一同傳送給 `LLM Service` 進行問答生成。
  * **API 與部署**：`FastAPI` 提供主要的 RESTful API，並使用 `Docker Compose` 進行容器化部署。`Prometheus` 和 `Grafana` 負責監控 API 與 LLM 服務的效能指標。

## 核心程式碼片段與註解

### `app/main.py` – 注入並回傳片段 metadata

以下程式碼展示了如何將文件來源資訊（`source`）與區塊編號（`chunk`）存入 ChromaDB，並在查詢結果中返回。

```python
# 文件攝取函數，包含 metadata 處理
def ingest_documents():
    # ... (載入文件與分割的代碼) ...

    # 使用 LangChain 將文件區塊轉換為向量
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 關鍵點：在存入 ChromaDB 時，將 'source' 與 'chunk' 資訊一併存入 metadata
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db",
        metadatas=[{"source": doc.metadata.get("source"), "chunk": idx} for idx, doc in enumerate(splits)]
    )
    vectorstore.persist()
    print("文件攝取完成並儲存到 ChromaDB。")
    return vectorstore, embeddings

# 查詢端點，執行完整的 RAG 檢索與生成
@app.post("/query")
async def query_documents(q: Query):
    if not retrieval_qa_chain:
        return {"error": "RAG pipeline not initialized. Please ensure documents are ingested."}
    
    # 呼叫 RetrievalQA 鏈來執行檢索與生成
    result = retrieval_qa_chain.invoke({"query": q.query})
    
    # 提取生成的答案
    answer = result["result"]
    
    # 提取所有來源文件名稱，並去除重複
    sources = [doc.metadata.get("source") for doc in result["source_documents"]]
    
    return {
        "query": q.query, 
        "answer": answer, 
        "sources": list(set(sources)) # 移除重複來源
    }
```

### `app/main.py` – 實作 RetrievalQA pipeline

這段程式碼展示了如何將向量檢索器與 LLM 結合，並定義一個專業的 Prompt 模板，以提高回答的品質與可信度。

```python
# 在應用程式啟動時，初始化 RetrievalQA 管道
@app.on_event("startup")
async def startup_event():
    global vectorstore, embeddings, retrieval_qa_chain
    # ... (文件攝取與 ChromaDB 載入邏輯) ...

    # 關鍵點：從配置檔案讀取 LLM 參數，而非硬編碼
    llm_config = yaml.safe_load(open("./config/llm.yaml", "r"))
    llm_model_name = llm_config["model_name"]
    llm_api_base = llm_config["api_base"]
    llm_api_key = os.getenv("OPENAI_API_KEY", "sk-xxxx") 

    # 建立 LLM 實例，這裡使用 OpenAI 服務的相容 API
    llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_base=llm_api_base, openai_api_key=llm_api_key)
    
    # 定義 RAG 專用的 Prompt 模板，引導 LLM 回答
    prompt_template = """你是一個專業的法律文件檢索助手，請根據提供的上下文來回答問題。
如果上下文沒有足夠資訊，請回答「根據現有文件，我無法回答這個問題。」
請在答案中註明相關的來源文件。

上下文：{context}
問題：{question}
答案："""
    
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 建立 RetrievalQA 鏈，將 LLM 與 Retriever 結合
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True, # 必須為 True 才能回傳來源文件
        chain_type_kwargs={"prompt": qa_prompt}
    )
```

## 快速開始

### 前置需求

  * **Docker & Docker Compose**
  * **具網路的 LLM 服務**（或自建 vLLM 容器）
  * **本機具備適當的硬碟**存放文件與向量庫

### 1\. 取得專案

從 GitHub 儲存庫克隆程式碼：

```bash
git clone https://github.com/BpsEason/lexrag.git
cd lexrag
```

### 2\. 配置環境變數

建立 `.env` 檔案以儲存敏感資訊，例如 Hugging Face token 或 OpenAI API Key。

````bash
# 建立 .env 檔案
touch .env
```.env` 範例：

````

# .env 檔案範例

APP\_ENV=development
APP\_PORT=8000

CHROMA\_DB\_DIR=./chroma\_db

# 如果需要 HuggingFace 模型，請在此填寫

HF\_TOKEN=your\_huggingface\_token

# 如果使用 vLLM，此處為假 API Key

OPENAI\_API\_KEY=sk-your-vllm-key

PROMETHEUS\_PORT=9090

````

### 3. 啟動服務

```bash
./start.sh
````

或者手動啟動：

```bash
docker-compose up -d --build
```

所有服務啟動完畢後：

  * **API 文件 (Swagger UI)**：`http://localhost:8000/docs`
  * **ReDoc 文件**：`http://localhost:8000/redoc`
  * **Prometheus**：`http://localhost:9090`
  * **Grafana**：`http://localhost:3000`

### 使用範例

1.  **放置法律文件**
    將欲檢索的 PDF、DOCX 或 Markdown 檔放入 `documents/` 目錄。

2.  **查詢**
    使用 cURL 發送請求：

    ```bash
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"query": "請問刑法第九十條的內容是什麼？"}'
    ```

    範例回應：

    ```json
    {
      "query": "請問刑法第九十條的內容是什麼？",
      "answer": "刑法第九十條規定…",
      "sources": ["民法全書.pdf"]
    }
    ```

### 設定說明

  * `config/embeddings.yaml`：向量化模型參數
  * `config/llm.yaml`：LLM 連線與生成參數
  * `monitoring/`：Prometheus 與 Grafana dashboard 配置
  * `tests/`：pytest 測試範例

### 測試與 CI/CD

  * **本地測試**：

    ```bash
    pytest --maxfail=1 --disable-warnings -v
    ```

  * **GitHub Actions**：
    `ci.yml` 會在每次 push 與 pull request 自動執行 lint、安裝相依、執行測試。
    可延伸新增部署步驟或對應 staging/production 分支策略。

### 社群與貢獻

歡迎提出 issue、PR 或 feature request。建議流程：

  * Fork 本專案
  * 建立 feature 分支：`git checkout -b feature/your-feature`
  * 提交程式：`git commit -m "新增 XXX 功能"`
  * Push 並發起 Pull Request

## 授權

本專案採用 **MIT License** 條款。詳見 `LICENSE`。
