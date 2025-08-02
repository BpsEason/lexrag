#!/bin/bash
PROJECT_NAME="lexrag"
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
echo -e "${GREEN}### Starting $PROJECT_NAME service... ###${NC}"
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker service is not running. Please start it.${NC}"
  exit 1
fi
docker-compose up --build -d
if [ $? -eq 0 ]; then
  echo -e "${GREEN}### Service started successfully! ###${NC}"
  echo ""
  echo "---"
  echo "你可以透過瀏覽器或 cURL 測試 API："
  echo ""
  echo -e "${GREEN}API 文件 (Swagger UI):${NC} http://localhost:8000/docs"
  echo -e "${GREEN}API 文件 (ReDoc):${NC} http://localhost:8000/redoc"
  echo ""
  echo -e "${GREEN}cURL 測試範例:${NC}"
  echo -e "${GREEN}根路由:${NC} curl http://localhost:8000"
  echo -e "${GREEN}查詢路由:${NC} curl -X POST \"http://localhost:8000/query\" -H \"Content-Type: application/json\" -d '{\"query\": \"你好，Docker！\"}'"
  echo ""
  echo -e "${GREEN}Grafana dashboard:${NC} http://localhost:3000"
  echo -e "${GREEN}Prometheus metrics:${NC} http://localhost:9090"
  echo "---"
else
  echo -e "${RED}Error: Failed to start services. Please check Docker logs.${NC}"
  exit 1
fi
