from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
import time
from datetime import datetime
import atexit

from config.settings import settings, check_environment
from models.kanana_model import KananaModel
from services.vector_service import VectorService
from services.gpt_service import GPTService
from utils.memory_manager import MessageQueue
from utils.mlops_manager import MLOpsManager  # 🚀 새로운 통합 MLOps 매니저

# FastAPI 앱 초기화
app = FastAPI(title="RAG Chat API with Advanced MLOps", version="3.0.0")

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
mlops_manager = None  # 🚀 통합 MLOps 매니저

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
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    search_results: List[dict]
    memory_content: str
    timing: dict
    stats: dict
    mlops_info: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_db_connected: bool
    document_count: int
    uptime: str
    mlops_status: dict

# 🚀 새로운 MLOps 관련 모델들
class MLOpsStatusResponse(BaseModel):
    collector_stats: dict
    training_status: dict
    models_info: dict
    events_summary: dict
    performance_metrics: dict

class FinetuneRequest(BaseModel):
    force: bool = False
    backup_existing: bool = True

class FinetuneResponse(BaseModel):
    success: bool
    message: str
    version: Optional[str] = None
    training_data_count: int
    training_time: Optional[float] = None
    estimated_time: Optional[str] = None

class MLOpsSettingsRequest(BaseModel):
    batch_size: Optional[int] = None
    auto_trigger: Optional[bool] = None
    collection_enabled: Optional[bool] = None
    monitoring_enabled: Optional[bool] = None

# 시작 시간 기록
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델들 초기화"""
    global kanana_model, vector_service, gpt_service, mlops_manager
    
    print("🚀 RAG Chat 백엔드 서버 (Advanced MLOps) 초기화 중...")
    
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
        
        # 🚀 4. 통합 MLOps 매니저 초기화
        print("🤖 통합 MLOps 매니저 초기화...")
        mlops_manager = MLOpsManager(
            finetune_config=settings.get_finetune_config(),
            conversation_config=settings.get_conversation_config()
        )
        
        print("✅ 모든 서비스 초기화 완료!")
        
        # 🚀 서버 시작 이벤트 로깅
        if mlops_manager:
            mlops_manager._log_event("server_started", {
                "model_loaded": True,
                "vector_db_connected": True,
                "gpt_service_ready": True
            }, "RAG Chat 서버가 성공적으로 시작되었습니다.")
        
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    print("🛑 서버 종료 중...")
    
    if mlops_manager:
        mlops_manager.shutdown()
    
    print("✅ 정리 작업 완료")

# 애플리케이션 종료 시 정리
atexit.register(lambda: mlops_manager.shutdown() if mlops_manager else None)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인 (MLOps 포함)"""
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
    
    # 🚀 MLOps 상태 정보
    mlops_status = {}
    if mlops_manager:
        try:
            mlops_status = mlops_manager.get_status()
        except Exception as e:
            mlops_status = {"error": str(e)}
    
    return HealthResponse(
        status="healthy" if kanana_model and vector_service and gpt_service else "initializing",
        model_loaded=kanana_model is not None,
        vector_db_connected=vector_connected,
        document_count=doc_count,
        uptime=uptime_str,
        mlops_status=mlops_status
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """채팅 메시지 처리 (MLOps 대화 수집 포함)"""
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
        
        # 🚀 4. MLOps 대화 수집 및 파인튜닝 트리거 확인
        mlops_info = {"collection_enabled": False, "training_triggered": False}
        if mlops_manager:
            try:
                mlops_result = mlops_manager.process_conversation(
                    user_message=message.message,
                    assistant_response=response,
                    user_id=message.user_id,
                    session_id=message.session_id,
                    metadata={
                        "search_results_count": len(search_results),
                        "memory_length": len(memory_content),
                        "timing": {
                            "search_ms": search_time,
                            "gpt_ms": gpt_time
                        }
                    }
                )
                mlops_info = mlops_result
            except Exception as e:
                print(f"⚠️ MLOps 처리 오류: {e}")
                mlops_info = {"error": str(e)}
        
        # 5. 벡터화 및 저장
        embedding_start = time.time()
        doc = f"USER : {message.message}<\\n>ASSISTANT : {response}"
        await asyncio.to_thread(vector_service.add_document, doc)
        embedding_time = (time.time() - embedding_start) * 1000
        stats["total_embedding_time"] += embedding_time
        
        # 6. 메모리 업데이트
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
            stats=avg_stats,
            mlops_info=mlops_info
        )
        
    except Exception as e:
        # 🚀 오류 이벤트 로깅
        if mlops_manager:
            mlops_manager._log_event("chat_error", {
                "error": str(e),
                "user_id": message.user_id,
                "message_length": len(message.message)
            }, f"채팅 처리 오류: {str(e)}")
        
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

# 🚀 강화된 MLOps 엔드포인트들

@app.get("/mlops/status", response_model=MLOpsStatusResponse)
async def get_mlops_status():
    """통합 MLOps 상태 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        status = mlops_manager.get_status()
        performance = mlops_manager.get_performance_metrics()
        
        return MLOpsStatusResponse(
            collector_stats=status["collector"],
            training_status=status["training"],
            models_info=status["models"],
            events_summary=status["events"],
            performance_metrics=performance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLOps status error: {str(e)}")

@app.get("/mlops/conversations")
async def get_conversations(limit: Optional[int] = 50):
    """수집된 대화 목록 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        conversations = mlops_manager.collector.get_collected_conversations(limit)
        return {
            "conversations": [conv.to_dict() for conv in conversations],
            "total_count": mlops_manager.collector.stats["total_collected"],
            "stats": mlops_manager.collector.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversations retrieval error: {str(e)}")

@app.post("/mlops/finetune", response_model=FinetuneResponse)
async def trigger_finetuning(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """수동 파인튜닝 트리거"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    if mlops_manager.training_in_progress:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    try:
        # 수집된 대화 수 확인
        total_conversations = mlops_manager.collector.stats["total_collected"]
        
        if total_conversations == 0:
            return FinetuneResponse(
                success=False,
                message="수집된 대화가 없습니다.",
                training_data_count=0
            )
        
        if not request.force and total_conversations < mlops_manager.batch_size:
            return FinetuneResponse(
                success=False,
                message=f"배치 크기({mlops_manager.batch_size})에 도달하지 않았습니다. force=true로 강제 실행 가능.",
                training_data_count=total_conversations
            )
        
        # 백그라운드에서 파인튜닝 실행
        def run_training():
            try:
                mlops_manager._log_event("training_triggered", {
                    "trigger_type": "manual",
                    "total_conversations": total_conversations,
                    "force": request.force
                }, "수동 파인튜닝 트리거")
                
                result = mlops_manager.start_finetuning(force=request.force)
                
                if result["success"]:
                    mlops_manager.current_model_version = result["version"]
                    mlops_manager.last_training_count = total_conversations
                    
                    mlops_manager._log_event("training_completed", {
                        "version": result["version"],
                        "training_time": result["training_time"],
                        "training_samples": result["training_samples"]
                    }, f"수동 파인튜닝 완료! 버전: {result['version']}")
                
            except Exception as e:
                mlops_manager._log_event("training_failed", {
                    "error": str(e),
                    "trigger_type": "manual"
                }, f"수동 파인튜닝 실패: {str(e)}")
        
        # 백그라운드 태스크로 실행
        background_tasks.add_task(run_training)
        
        return FinetuneResponse(
            success=True,
            message="파인튜닝이 백그라운드에서 시작되었습니다.",
            training_data_count=total_conversations,
            estimated_time="약 5-10분"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finetuning error: {str(e)}")

@app.delete("/mlops/conversations")
async def clear_conversations(backup: bool = True):
    """수집된 대화 데이터 초기화"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        old_count = mlops_manager.collector.stats["total_collected"]
        success = mlops_manager.collector.clear_conversations(backup_first=backup)
        
        if success:
            mlops_manager._log_event("conversations_cleared", {
                "cleared_count": old_count,
                "backup_created": backup
            }, f"대화 데이터 초기화: {old_count}개 제거")
        
        return {
            "success": success,
            "message": "대화 데이터가 초기화되었습니다." if success else "초기화에 실패했습니다.",
            "backup_created": backup and success,
            "cleared_count": old_count if success else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear conversations error: {str(e)}")

@app.post("/mlops/export")
async def export_training_data():
    """파인튜닝용 데이터셋 생성 및 내보내기"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        dataset_path = mlops_manager.collector.export_for_finetuning()
        
        if dataset_path:
            mlops_manager._log_event("dataset_exported", {
                "dataset_path": dataset_path,
                "sample_count": mlops_manager.collector.stats["total_collected"]
            }, "학습 데이터셋이 내보내졌습니다.")
            
            return {
                "success": True,
                "message": "데이터셋이 성공적으로 생성되었습니다.",
                "dataset_path": dataset_path,
                "training_samples": mlops_manager.collector.stats["total_collected"]
            }
        else:
            return {
                "success": False,
                "message": "데이터셋 생성에 실패했습니다.",
                "dataset_path": None,
                "training_samples": 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/mlops/training-progress")
async def get_training_progress():
    """파인튜닝 진행 상황 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    current_count = mlops_manager.collector.stats["total_collected"]
    batch_size = mlops_manager.batch_size
    
    return {
        "current_conversations": current_count,
        "batch_size": batch_size,
        "progress_percentage": (current_count % batch_size) / batch_size * 100 if batch_size > 0 else 0,
        "conversations_until_training": mlops_manager.collector.get_pending_training_count(batch_size),
        "training_in_progress": mlops_manager.training_in_progress,
        "auto_trigger_enabled": mlops_manager.auto_trigger,
        "current_version": mlops_manager.current_model_version,
        "last_training_count": mlops_manager.last_training_count
    }

@app.post("/mlops/settings")
async def update_mlops_settings(settings_request: MLOpsSettingsRequest):
    """MLOps 설정 동적 업데이트"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        # 설정 업데이트
        updates = {}
        if settings_request.batch_size is not None:
            updates["batch_size"] = settings_request.batch_size
        if settings_request.auto_trigger is not None:
            updates["auto_trigger"] = settings_request.auto_trigger
        if settings_request.collection_enabled is not None:
            updates["collection_enabled"] = settings_request.collection_enabled
        if settings_request.monitoring_enabled is not None:
            updates["monitoring_enabled"] = settings_request.monitoring_enabled
        
        updated = mlops_manager.update_settings(**updates)
        
        return {
            "success": True,
            "message": "설정이 업데이트되었습니다.",
            "updated_settings": updated
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Settings update error: {str(e)}")

@app.get("/mlops/events")
async def get_mlops_events(
    event_type: Optional[str] = None,
    limit: int = 100
):
    """MLOps 이벤트 로그 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        events = mlops_manager.get_events_log(event_type=event_type, limit=limit)
        
        return {
            "events": events,
            "total_count": len(mlops_manager.events_log),
            "filtered_count": len(events),
            "filter": event_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Events retrieval error: {str(e)}")

@app.get("/mlops/training-history")
async def get_training_history():
    """파인튜닝 히스토리 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        history = mlops_manager.get_training_history()
        
        return {
            "training_history": history,
            "total_trainings": len(history),
            "successful_trainings": len([h for h in history if h.get("success", False)])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training history error: {str(e)}")

@app.get("/mlops/models")
async def get_model_versions():
    """모델 버전 목록 조회"""
    if not mlops_manager or not mlops_manager.finetuner:
        raise HTTPException(status_code=503, detail="Finetuner not available")
    
    try:
        versions = mlops_manager.finetuner.get_model_versions()
        
        return {
            "model_versions": versions,
            "total_versions": len(versions),
            "current_version": mlops_manager.current_model_version,
            "latest_version": versions[0]["version"] if versions else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model versions error: {str(e)}")

@app.get("/mlops/performance")
async def get_performance_metrics():
    """성능 메트릭 조회"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        metrics = mlops_manager.get_performance_metrics()
        
        return {
            "performance_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metrics error: {str(e)}")

@app.post("/mlops/cleanup")
async def cleanup_old_data(keep_days: int = 30):
    """오래된 데이터 정리"""
    if not mlops_manager:
        raise HTTPException(status_code=503, detail="MLOps manager not initialized")
    
    try:
        result = mlops_manager.cleanup_old_data(keep_days=keep_days)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

# 기존 엔드포인트들 유지

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
    
    # 🚀 이벤트 로깅
    if mlops_manager:
        mlops_manager._log_event("memory_cleared", {
            "previous_count": len(memory.messages)
        }, "대화 메모리가 초기화되었습니다.")
    
    return {"message": "Memory cleared successfully"}

@app.get("/stats")
async def get_stats():
    """성능 통계 조회 (MLOps 포함)"""
    avg_stats = {}
    if stats["total_queries"] > 0:
        avg_stats = {
            "total_queries": stats["total_queries"],
            "avg_search_time": f"{stats['total_search_time'] / stats['total_queries']:.2f}ms",
            "avg_gpt_time": f"{stats['total_gpt_time'] / stats['total_queries']:.2f}ms",
            "avg_embedding_time": f"{stats['total_embedding_time'] / stats['total_queries']:.2f}ms"
        }
    
    # 🚀 MLOps 통계 추가
    mlops_stats = {}
    if mlops_manager:
        try:
            mlops_stats = mlops_manager.get_status()
        except Exception as e:
            mlops_stats = {"error": str(e)}
    
    return {
        "performance": {
            "raw_stats": stats,
            "averages": avg_stats,
            "document_count": vector_service.get_document_count() if vector_service else 0
        },
        "mlops": mlops_stats,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 RAG Chat 백엔드 서버 (Advanced MLOps) 시작!")
    print("=" * 80)
    print("🆕 새로운 Advanced MLOps 엔드포인트:")
    print("- GET  /mlops/status            : 통합 MLOps 상태 (성능 메트릭 포함)")
    print("- GET  /mlops/conversations     : 수집된 대화 조회")
    print("- POST /mlops/finetune          : 수동 파인튜닝 트리거 (백그라운드)")
    print("- DELETE /mlops/conversations   : 대화 데이터 초기화")
    print("- POST /mlops/export           : 학습 데이터 내보내기")
    print("- GET  /mlops/training-progress : 학습 진행 상황")
    print("- POST /mlops/settings         : 설정 동적 업데이트")
    print("- GET  /mlops/events           : 이벤트 로그 조회")
    print("- GET  /mlops/training-history : 파인튜닝 히스토리")
    print("- GET  /mlops/models           : 모델 버전 목록")
    print("- GET  /mlops/performance      : 성능 메트릭")
    print("- POST /mlops/cleanup          : 오래된 데이터 정리")
    print("=" * 80)
    print("🎯 핵심 기능:")
    print("- 🤖 자동 대화 수집 및 파인튜닝")
    print("- 📊 실시간 성능 모니터링")
    print("- 🔄 점진적 모델 개선 (v1 → v2 → v3)")
    print("- 📈 이벤트 로깅 및 웹훅 알림")
    print("- ⚡ 백그라운드 파인튜닝")
    print("=" * 80)
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )