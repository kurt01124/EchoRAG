import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from utils.conversation_collector import ConversationCollector
from utils.automated_finetuning import AutomatedFinetuner

@dataclass
class MLOpsEvent:
    """MLOps 이벤트 데이터 클래스"""
    event_type: str  # "conversation_collected", "training_triggered", "training_completed", "error"
    timestamp: str
    data: Dict[str, Any]
    message: str

class MLOpsManager:
    """통합 MLOps 관리 시스템 (개선된 배치 처리)"""
    
    def __init__(self, finetune_config: Dict[str, Any], conversation_config: Dict[str, Any]):
        self.finetune_config = finetune_config
        self.conversation_config = conversation_config
        
        # 핵심 컴포넌트 초기화
        self.collector = ConversationCollector(conversation_config)
        self.finetuner = AutomatedFinetuner(finetune_config) if finetune_config.get("enabled", True) else None
        
        # 설정
        self.batch_size = finetune_config.get("batch_size", 50)
        self.auto_trigger = finetune_config.get("auto_trigger", True)
        self.monitoring_enabled = finetune_config.get("monitoring_enabled", True)
        self.webhook_url = finetune_config.get("webhook_url")
        
        # 🚀 개선된 상태 관리
        self.training_in_progress = False
        self.last_training_count = 0  # 마지막으로 학습한 시점의 총 대화 수
        self.current_model_version = None
        self.events_log = []
        self.pending_training_request = False  # 🆕 대기 중인 학습 요청
        
        # 스레드 관리
        self.training_thread = None
        self.lock = threading.Lock()
        
        # 이벤트 로그 파일 경로
        self.events_log_path = Path(finetune_config.get("data_path", "./data/finetune")) / "mlops_events.json"
        
        # 기존 이벤트 로그 로드
        self._load_events_log()
        
        # 초기화 이벤트
        self._log_event("system_initialized", {
            "batch_size": self.batch_size,
            "auto_trigger": self.auto_trigger,
            "finetuner_enabled": self.finetuner is not None
        }, "MLOps 시스템이 초기화되었습니다.")
        
        print(f"🤖 MLOps 매니저 초기화 완료")
        print(f"📊 배치 크기: {self.batch_size}개 대화")
        print(f"⚡ 자동 트리거: {'ON' if self.auto_trigger else 'OFF'}")
        print(f"🔧 파인튜너: {'활성화' if self.finetuner else '비활성화'}")
        print(f"📈 모니터링: {'활성화' if self.monitoring_enabled else '비활성화'}")
    
    def _load_events_log(self):
        """이벤트 로그 로드"""
        if self.events_log_path.exists():
            try:
                with open(self.events_log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    self.events_log = [MLOpsEvent(**event) for event in log_data]
                print(f"📂 이벤트 로그 로드: {len(self.events_log)}개 이벤트")
            except Exception as e:
                print(f"⚠️ 이벤트 로그 로드 실패: {e}")
                self.events_log = []
    
    def _save_events_log(self):
        """이벤트 로그 저장"""
        try:
            self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_data = []
            for event in self.events_log[-1000:]:  # 최근 1000개만 저장
                log_data.append({
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "data": event.data,
                    "message": event.message
                })
            
            with open(self.events_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ 이벤트 로그 저장 실패: {e}")
    
    def _log_event(self, event_type: str, data: Dict[str, Any], message: str):
        """이벤트 로깅"""
        event = MLOpsEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
            message=message
        )
        
        self.events_log.append(event)
        
        # 이벤트 출력
        if self.monitoring_enabled:
            print(f"📋 [{event_type}] {message}")
        
        # 로그 저장 (주기적으로)
        if len(self.events_log) % 10 == 0:
            self._save_events_log()
        
        # 웹훅 알림 (중요 이벤트만)
        if event_type in ["training_completed", "training_failed", "error"] and self.webhook_url:
            self._send_webhook_notification(event)
    
    def _send_webhook_notification(self, event: MLOpsEvent):
        """웹훅 알림 전송 (비동기)"""
        def send_webhook():
            try:
                import requests
                
                payload = {
                    "text": f"🤖 MLOps 알림: {event.message}",
                    "attachments": [{
                        "color": "good" if "completed" in event.event_type else "danger",
                        "fields": [
                            {"title": "이벤트 타입", "value": event.event_type, "short": True},
                            {"title": "시간", "value": event.timestamp, "short": True},
                            {"title": "데이터", "value": json.dumps(event.data, ensure_ascii=False), "short": False}
                        ]
                    }]
                }
                
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
            except Exception as e:
                print(f"⚠️ 웹훅 전송 실패: {e}")
        
        # 백그라운드에서 실행
        webhook_thread = threading.Thread(target=send_webhook, daemon=True)
        webhook_thread.start()
    
    def _should_trigger_training(self) -> bool:
        """🚀 개선된 트리거 조건 판단"""
        current_count = self.collector.stats["total_collected"]
        new_data_count = current_count - self.last_training_count
        
        # 최소 배치 크기만큼 새 데이터가 쌓였는지 확인
        return new_data_count >= self.batch_size
    
    def _get_new_data_count(self) -> int:
        """🚀 새로운 데이터 개수 반환"""
        current_count = self.collector.stats["total_collected"]
        return current_count - self.last_training_count
    
    def process_conversation(
        self, 
        user_message: str, 
        assistant_response: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """🚀 개선된 대화 처리 및 자동 파인튜닝 트리거"""
        
        # 1. 대화 수집
        collected = self.collector.collect_conversation(
            user_message=user_message,
            assistant_response=assistant_response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        current_count = self.collector.stats["total_collected"]
        new_data_count = self._get_new_data_count()
        
        result = {
            "collected": collected,
            "total_collected": current_count,
            "new_data_count": new_data_count,  # 🆕 새 데이터 개수
            "should_train": False,
            "pending_count": 0,
            "training_triggered": False,
            "training_queued": False,  # 🆕 대기열 추가 여부
            "current_version": self.current_model_version
        }
        
        if collected:
            # 이벤트 로깅
            self._log_event("conversation_collected", {
                "user_id": user_id,
                "session_id": session_id,
                "total_count": current_count,
                "new_data_count": new_data_count,
                "message_length": len(user_message) + len(assistant_response)
            }, f"새 대화 수집됨 (총 {current_count}개, 신규 {new_data_count}개)")
            
            if self.auto_trigger and self.finetuner:
                # 🚀 개선된 트리거 로직
                should_train = self._should_trigger_training()
                result["should_train"] = should_train
                result["pending_count"] = max(0, self.batch_size - new_data_count)
                
                if should_train:
                    if not self.training_in_progress:
                        # 즉시 파인튜닝 시작
                        result["training_triggered"] = self._trigger_async_training()
                    else:
                        # 🆕 진행 중이면 대기 요청 설정
                        self.pending_training_request = True
                        result["training_queued"] = True
                        
                        self._log_event("training_queued", {
                            "current_count": current_count,
                            "new_data_count": new_data_count,
                            "batch_size": self.batch_size
                        }, f"파인튜닝 대기 설정 (신규 데이터 {new_data_count}개)")
                        
                        print(f"📋 파인튜닝 대기 설정: 현재 진행 중이므로 완료 후 실행 예정 (신규 {new_data_count}개)")
        
        return result
    
    def _trigger_async_training(self) -> bool:
        """🚀 개선된 비동기 파인튜닝 트리거"""
        try:
            with self.lock:
                if self.training_in_progress:
                    print("⚠️ 이미 파인튜닝이 진행 중입니다.")
                    return False
                
                self.training_in_progress = True
            
            def training_worker():
                try:
                    current_count = self.collector.stats["total_collected"]
                    new_data_count = current_count - self.last_training_count
                    
                    self._log_event("training_triggered", {
                        "batch_size": self.batch_size,
                        "total_conversations": current_count,
                        "new_data_count": new_data_count,
                        "trigger_type": "automatic"
                    }, f"자동 파인튜닝 시작 (총 {current_count}개, 신규 {new_data_count}개)")
                    
                    print(f"🚀 자동 파인튜닝 시작: 총 {current_count}개 대화 (신규 {new_data_count}개)")
                    success_result = self.start_finetuning()
                    
                    if success_result["success"]:
                        # 🚀 성공 시 last_training_count 업데이트
                        self.current_model_version = success_result["version"]
                        self.last_training_count = current_count  # 현재 시점으로 업데이트
                        
                        self._log_event("training_completed", {
                            "version": success_result["version"],
                            "training_time": success_result["training_time"],
                            "training_samples": success_result["training_samples"],
                            "output_path": success_result["output_path"],
                            "total_conversations": current_count,
                            "new_data_processed": new_data_count
                        }, f"파인튜닝 완료! 버전: {success_result['version']} (신규 {new_data_count}개 처리)")
                        
                        print(f"✅ 자동 파인튜닝 완료! 버전: {success_result['version']}")
                        
                        # 🚀 대기 중인 요청이 있으면 다시 트리거
                        self._check_pending_training()
                        
                    else:
                        self._log_event("training_failed", {
                            "error": "Unknown error during training",
                            "total_conversations": current_count,
                            "new_data_count": new_data_count
                        }, "파인튜닝 실패")
                        
                        print("❌ 자동 파인튜닝 실패!")
                        
                except Exception as e:
                    self._log_event("training_failed", {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "total_conversations": self.collector.stats["total_collected"]
                    }, f"파인튜닝 오류: {str(e)}")
                    
                    print(f"❌ 자동 파인튜닝 오류: {e}")
                    
                finally:
                    with self.lock:
                        self.training_in_progress = False
            
            # 백그라운드 스레드에서 실행
            self.training_thread = threading.Thread(target=training_worker, daemon=True)
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            with self.lock:
                self.training_in_progress = False
            
            self._log_event("error", {
                "error": str(e),
                "context": "trigger_async_training"
            }, f"파인튜닝 트리거 실패: {str(e)}")
            
            print(f"❌ 파인튜닝 트리거 실패: {e}")
            return False
    
    def _check_pending_training(self):
        """🚀 대기 중인 파인튜닝 요청 확인 및 실행"""
        if self.pending_training_request:
            # 대기 요청 재확인
            new_data_count = self._get_new_data_count()
            
            if new_data_count >= self.batch_size:
                self.pending_training_request = False
                
                print(f"🔄 대기 중이던 파인튜닝 시작: 신규 {new_data_count}개 데이터")
                
                # 약간의 지연 후 다시 트리거 (이전 작업 완전 완료 보장)
                def delayed_trigger():
                    time.sleep(1)  # 1초 대기
                    self._trigger_async_training()
                
                delayed_thread = threading.Thread(target=delayed_trigger, daemon=True)
                delayed_thread.start()
                
            else:
                # 아직 배치 크기에 도달하지 않음
                self.pending_training_request = False
                print(f"📋 대기 요청 해제: 신규 데이터 {new_data_count}개로 배치 크기 미달")
    
    def start_finetuning(self, force: bool = False) -> Dict[str, Any]:
        """파인튜닝 시작 (동기 실행)"""
        if not self.finetuner:
            return {
                "success": False,
                "error": "파인튜너가 비활성화되어 있습니다.",
                "version": None,
                "training_time": 0,
                "training_samples": 0
            }
        
        try:
            # 1. 데이터셋 생성 (전체 대화 사용)
            dataset_path = self.collector.export_for_finetuning()
            if not dataset_path:
                return {
                    "success": False,
                    "error": "학습 데이터셋 생성 실패",
                    "version": None,
                    "training_time": 0,
                    "training_samples": 0
                }
            
            # 2. 파인튜닝 실행
            result = self.finetuner.run_automated_finetuning(dataset_path)
            
            if result["success"]:
                return {
                    "success": True,
                    "version": result["version"],
                    "output_path": result["output_path"],
                    "training_time": result["training_time"],
                    "training_samples": result["training_samples"]
                }
            else:
                return {
                    "success": False,
                    "error": "파인튜닝 실행 실패",
                    "version": None,
                    "training_time": 0,
                    "training_samples": 0
                }
                
        except Exception as e:
            self._log_event("error", {
                "error": str(e),
                "context": "start_finetuning"
            }, f"파인튜닝 실행 오류: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "version": None,
                "training_time": 0,
                "training_samples": 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """🚀 개선된 전체 MLOps 상태 반환"""
        collector_stats = self.collector.get_stats()
        new_data_count = self._get_new_data_count()
        
        # 파인튜닝 관련 상태
        training_status = {
            "batch_size": self.batch_size,
            "auto_trigger": self.auto_trigger,
            "in_progress": self.training_in_progress,
            "pending_request": self.pending_training_request,  # 🆕
            "last_training_count": self.last_training_count,
            "new_data_count": new_data_count,  # 🆕
            "pending_count": max(0, self.batch_size - new_data_count),
            "should_train": self._should_trigger_training(),
            "current_version": self.current_model_version,
            "finetuner_enabled": self.finetuner is not None
        }
        
        # 모델 정보
        models_info = {
            "base_path": str(self.finetuner.base_output_dir) if self.finetuner else None,
            "backup_count": self.finetuner.backup_count if self.finetuner else 0,
            "available_versions": self.finetuner.get_model_versions() if self.finetuner else []
        }
        
        # 최근 이벤트
        recent_events = []
        for event in self.events_log[-10:]:  # 최근 10개
            recent_events.append({
                "type": event.event_type,
                "timestamp": event.timestamp,
                "message": event.message,
                "data": event.data
            })
        
        return {
            "collector": collector_stats,
            "training": training_status,
            "models": models_info,
            "events": {
                "total_count": len(self.events_log),
                "recent": recent_events
            },
            "monitoring": {
                "enabled": self.monitoring_enabled,
                "webhook_configured": self.webhook_url is not None
            }
        }
    
    # 나머지 메서드들은 기존과 동일...
    def get_training_history(self) -> List[Dict[str, Any]]:
        """파인튜닝 히스토리 반환"""
        if not self.finetuner:
            return []
        
        return self.finetuner.get_training_history()
    
    def get_events_log(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """이벤트 로그 조회"""
        events = self.events_log
        
        # 타입 필터링
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # 최신 순으로 제한
        events = events[-limit:]
        
        return [{
            "type": event.event_type,
            "timestamp": event.timestamp,
            "message": event.message,
            "data": event.data
        } for event in events]
    
    def update_settings(self, **kwargs) -> Dict[str, Any]:
        """설정 동적 업데이트"""
        updated = {}
        
        if "batch_size" in kwargs and kwargs["batch_size"] > 0:
            old_batch = self.batch_size
            self.batch_size = kwargs["batch_size"]
            updated["batch_size"] = {"old": old_batch, "new": self.batch_size}
        
        if "auto_trigger" in kwargs:
            old_trigger = self.auto_trigger
            self.auto_trigger = kwargs["auto_trigger"]
            updated["auto_trigger"] = {"old": old_trigger, "new": self.auto_trigger}
        
        if "collection_enabled" in kwargs:
            old_enabled = self.collector.enabled
            self.collector.enabled = kwargs["collection_enabled"]
            updated["collection_enabled"] = {"old": old_enabled, "new": self.collector.enabled}
        
        if "monitoring_enabled" in kwargs:
            old_monitoring = self.monitoring_enabled
            self.monitoring_enabled = kwargs["monitoring_enabled"]
            updated["monitoring_enabled"] = {"old": old_monitoring, "new": self.monitoring_enabled}
        
        # 설정 변경 이벤트 로깅
        if updated:
            self._log_event("settings_updated", updated, f"설정이 업데이트되었습니다: {list(updated.keys())}")
        
        return updated
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        # 대화 수집 성능
        collector_stats = self.collector.get_stats()
        
        # 파인튜닝 성능
        training_history = self.get_training_history() if self.finetuner else []
        
        avg_training_time = 0
        total_trainings = len([h for h in training_history if h.get("success", False)])
        
        if total_trainings > 0:
            successful_trainings = [h for h in training_history if h.get("success", False)]
            avg_training_time = sum(h.get("training_time_seconds", 0) for h in successful_trainings) / len(successful_trainings)
        
        # 이벤트 통계
        event_stats = {}
        for event in self.events_log:
            event_type = event.event_type
            event_stats[event_type] = event_stats.get(event_type, 0) + 1
        
        return {
            "collection": {
                "total_collected": collector_stats["total_collected"],
                "filtered_out": collector_stats["filtered_out"],
                "collection_rate": collector_stats["total_collected"] / (collector_stats["total_collected"] + collector_stats["filtered_out"]) * 100 if (collector_stats["total_collected"] + collector_stats["filtered_out"]) > 0 else 0,
                "file_size_kb": collector_stats["file_size_kb"]
            },
            "training": {
                "total_trainings": total_trainings,
                "failed_trainings": len(training_history) - total_trainings,
                "avg_training_time_seconds": avg_training_time,
                "success_rate": total_trainings / len(training_history) * 100 if training_history else 0
            },
            "events": event_stats,
            "system": {
                "uptime_hours": (datetime.now() - datetime.fromisoformat(self.events_log[0].timestamp)).total_seconds() / 3600 if self.events_log else 0,
                "training_in_progress": self.training_in_progress
            }
        }
    
    def cleanup_old_data(self, keep_days: int = 30) -> Dict[str, Any]:
        """오래된 데이터 정리"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            # 오래된 이벤트 로그 정리
            original_count = len(self.events_log)
            self.events_log = [
                event for event in self.events_log 
                if datetime.fromisoformat(event.timestamp) > cutoff_date
            ]
            removed_events = original_count - len(self.events_log)
            
            # 로그 저장
            self._save_events_log()
            
            # 정리 이벤트 로깅
            self._log_event("cleanup_completed", {
                "keep_days": keep_days,
                "removed_events": removed_events
            }, f"데이터 정리 완료: {removed_events}개 이벤트 제거")
            
            return {
                "success": True,
                "removed_events": removed_events,
                "remaining_events": len(self.events_log),
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            self._log_event("error", {
                "error": str(e),
                "context": "cleanup_old_data"
            }, f"데이터 정리 오류: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def shutdown(self):
        """MLOps 매니저 종료"""
        try:
            # 진행 중인 훈련 대기
            if self.training_in_progress and self.training_thread:
                print("🔄 진행 중인 파인튜닝 완료 대기 중...")
                self.training_thread.join(timeout=300)  # 5분 대기
            
            # 최종 이벤트 로그 저장
            self._save_events_log()
            
            self._log_event("system_shutdown", {}, "MLOps 시스템이 종료되었습니다.")
            
            print("🛑 MLOps 매니저 종료 완료")
            
        except Exception as e:
            print(f"⚠️ MLOps 매니저 종료 중 오류: {e}")