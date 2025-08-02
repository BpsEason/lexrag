import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to LexRAG: Legal Document RAG Template!"}

def test_query_endpoint_placeholder():
    # 由於 RAG 鏈需要啟動時載入，測試時可能未初始化，這裡簡單測試端點是否回應
    response = client.post("/query", json={"query": "Test query"})
    assert response.status_code == 200
    assert "query" in response.json()
    assert "answer" in response.json()
    assert "sources" in response.json()
