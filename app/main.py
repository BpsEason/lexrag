from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os
import yaml

app = FastAPI(
    title="LexRAG: Legal RAG Template",
    description="A template for legal document retrieval and generation.",
    version="1.0.0"
)

class Query(BaseModel):
    query: str

def ingest_documents():
    """載入文件、分割並存入向量資料庫。"""
    documents = []
    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        print("沒有找到 documents 目錄，跳過文件攝取流程。")
        return None, None
    
    for file in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())
    
    if not documents:
        print("documents 目錄中沒有找到支援的文件，無法進行攝取。")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 優化：儲存帶有 metadata 的向量
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db",
        metadatas=[{"source": doc.metadata.get("source"), "chunk": idx} for idx, doc in enumerate(splits)]
    )
    vectorstore.persist()
    print("文件攝取完成並儲存到 ChromaDB。")
    return vectorstore, embeddings

# 全局變數，用於在啟動時儲存 vectorstore
vectorstore = None
embeddings = None
retrieval_qa_chain = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, embeddings, retrieval_qa_chain

    # 優化：從 config/llm.yaml 讀取 LLM 配置
    with open("./config/llm.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
        llm_model_name = llm_config["model_name"]
        llm_api_base = llm_config["api_base"]
        # 注意：這裡的 API Key 應從環境變數讀取，增加安全性
        llm_api_key = os.getenv("OPENAI_API_KEY", "sk-xxxx") 
    
    # 在啟動時檢查是否已存在 ChromaDB，若無則執行文件攝取
    if not os.path.exists("./chroma_db"):
        os.makedirs("documents", exist_ok=True)
        vectorstore, embeddings = ingest_documents()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 優化：實作 RetrievalQA pipeline
    llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_base=llm_api_base, openai_api_key=llm_api_key)
    
    prompt_template = """你是一個專業的法律文件檢索助手，請根據提供的上下文來回答問題。
如果上下文沒有足夠資訊，請回答「根據現有文件，我無法回答這個問題。」
請在答案中註明相關的來源文件。

上下文：{context}
問題：{question}
答案："""
    
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

@app.get("/")
def read_root():
    """根路由，返回歡迎訊息。"""
    return {"message": "Welcome to LexRAG: Legal Document RAG Template!"}

@app.post("/query")
async def query_documents(q: Query):
    """查詢端點，執行完整的 RAG 檢索與生成。"""
    if not retrieval_qa_chain:
        return {"error": "RAG pipeline not initialized. Please ensure documents are ingested."}
    
    result = retrieval_qa_chain.invoke({"query": q.query})
    
    answer = result["result"]
    sources = [doc.metadata.get("source") for doc in result["source_documents"]]
    
    return {
        "query": q.query, 
        "answer": answer, 
        "sources": list(set(sources)) # 移除重複來源
    }
