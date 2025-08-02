from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
import time
from datetime import datetime

from config.settings import settings, check_environment
from models.kanana_model import KananaModel
from services.vector_service import VectorService
from services.gpt_service import GPTService
from utils.memory_manager import MessageQueue

# FastAPI 앱 초기화
app = FastAPI(title="RAG Chat API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
            allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수들
kanana_model = None
vector_service = None
gpt_service = None
memory = MessageQueue(cnt=settings.memory_max_count)
stats = {
    "total_queries": 0,
    "total_search_time": 0,
    "total_gpt_time": 0,
    "total_embedding_time": 0
}

# Pydantic 모델들
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    search_results: List[dict]
    memory_content: str
    timing: dict
    stats: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_db_connected: bool
    document_count: int
    uptime: str

# 시작 시간 기록
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델들 초기화"""
    global kanana_model, vector_service, gpt_service
    
    print("🚀 RAG Chat 백엔드 서버 초기화 중...")
    
    # 환경 설정 확인
    settings.print_settings_summary()
    if not check_environment():
        print("❌ 환경 설정에 문제가 있습니다. 서버를 종료합니다.")
        exit(1)
    
    try:
        # 1. Kanana 모델 로딩
        print("📦 Kanana 모델 로딩...")
        kanana_model = KananaModel(settings.get_model_config())
        await asyncio.to_thread(kanana_model.load_model)
        
        # 2. 벡터 서비스 초기화
        print("🗄️ Vector 서비스 초기화...")
        vector_service = VectorService(kanana_model, settings.get_chroma_config())
        await asyncio.to_thread(vector_service.initialize)
        
        # 3. GPT 서비스 초기화
        print("🧠 GPT 서비스 초기화...")
        gpt_service = GPTService(settings.get_openai_config())
        
        print("✅ 모든 서비스 초기화 완료!")
        
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    uptime_seconds = time.time() - start_time
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
    
    doc_count = 0
    vector_connected = False
    
    if vector_service:
        try:
            doc_count = vector_service.get_document_count()
            vector_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy" if kanana_model and vector_service and gpt_service else "initializing",
        model_loaded=kanana_model is not None,
        vector_db_connected=vector_connected,
        document_count=doc_count,
        uptime=uptime_str
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """채팅 메시지 처리"""
    if not all([kanana_model, vector_service, gpt_service]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    query_start_time = time.time()
    
    try:
        # 통계 업데이트
        stats["total_queries"] += 1
        
        # 1. 벡터 검색
        search_start = time.time()
        search_results = await asyncio.to_thread(
            vector_service.search_similar,
            message.message,
            n_results=settings.search_default_results
        )
        search_time = (time.time() - search_start) * 1000
        stats["total_search_time"] += search_time
        
        # 2. 단기 기억 가져오기
        memory_content = memory.view()
        
        # 3. GPT 응답 생성
        gpt_start = time.time()
        response = await gpt_service.generate_response(
            user_message=message.message,
            search_results=search_results,
            memory_content=memory_content
        )
        gpt_time = (time.time() - gpt_start) * 1000
        stats["total_gpt_time"] += gpt_time
        
        # 4. 벡터화 및 저장
        embedding_start = time.time()
        doc = f"USER : {message.message}<\\n>ASSISTANT : {response}"
        await asyncio.to_thread(vector_service.add_document, doc)
        embedding_time = (time.time() - embedding_start) * 1000
        stats["total_embedding_time"] += embedding_time
        
        # 5. 메모리 업데이트
        memory.append({"role": "user", "content": message.message})
        memory.append({"role": "assistant", "content": response})
        
        # 타이밍 정보
        total_time = (time.time() - query_start_time) * 1000
        timing = {
            "total": f"{total_time:.2f}ms",
            "search": f"{search_time:.2f}ms",
            "gpt": f"{gpt_time:.2f}ms",
            "embedding": f"{embedding_time:.2f}ms"
        }
        
        # 평균 통계 계산
        avg_stats = {}
        if stats["total_queries"] > 0:
            avg_stats = {
                "total_queries": stats["total_queries"],
                "avg_search": f"{stats['total_search_time'] / stats['total_queries']:.2f}ms",
                "avg_gpt": f"{stats['total_gpt_time'] / stats['total_queries']:.2f}ms",
                "avg_embedding": f"{stats['total_embedding_time'] / stats['total_queries']:.2f}ms"
            }
        
        return ChatResponse(
            response=response,
            search_results=search_results,
            memory_content=memory_content,
            timing=timing,
            stats=avg_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.get("/memory")
async def get_memory():
    """현재 메모리 상태 조회"""
    return {
        "messages": memory.messages,
        "count": len(memory.messages),
        "max_count": memory.max_count
    }

@app.delete("/memory")
async def clear_memory():
    """메모리 초기화"""
    memory.clear()
    return {"message": "Memory cleared successfully"}

@app.get("/stats")
async def get_stats():
    """성능 통계 조회"""
    avg_stats = {}
    if stats["total_queries"] > 0:
        avg_stats = {
            "total_queries": stats["total_queries"],
            "avg_search_time": f"{stats['total_search_time'] / stats['total_queries']:.2f}ms",
            "avg_gpt_time": f"{stats['total_gpt_time'] / stats['total_queries']:.2f}ms",
            "avg_embedding_time": f"{stats['total_embedding_time'] / stats['total_queries']:.2f}ms"
        }
    
    return {
        "raw_stats": stats,
        "averages": avg_stats,
        "document_count": vector_service.get_document_count() if vector_service else 0
    }

if __name__ == "__main__":
    print("🚀 RAG Chat 백엔드 서버 시작!")
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )