import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import portalocker as fcntl
import threading
from dataclasses import dataclass, asdict

@dataclass
class ConversationEntry:
    """대화 엔트리 데이터 클래스"""
    user_message: str
    assistant_response: str
    timestamp: str
    user_id: str = "default"
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_training_format(self) -> str:
        """파인튜닝용 형태로 변환"""
        return f"USER : {self.user_message}<\\n>ASSISTANT : {self.assistant_response}"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

class ConversationCollector:
    """실시간 대화 수집 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 2000)
        self.filter_system = config.get("filter_system", True)
        self.data_path = config.get("data_path", "./data/finetune")
        self.file_name = config.get("file_name", "conversations.jsonl")
        
        # 파일 경로 설정
        self.file_path = Path(self.data_path) / self.file_name
        self.backup_path = Path(self.data_path) / f"backup_{self.file_name}"
        
        # 통계
        self.stats = {
            "total_collected": 0,
            "filtered_out": 0,
            "last_collection": None,
            "file_size_kb": 0
        }
        
        # 스레드 안전성을 위한 락
        self.lock = threading.Lock()
        
        # 초기화
        self._ensure_directory()
        self._load_stats()
        
        print(f"📚 대화 수집기 초기화: {'활성화' if self.enabled else '비활성화'}")
        if self.enabled:
            print(f"📁 저장 경로: {self.file_path}")
            print(f"📏 수집 조건: {self.min_length}~{self.max_length}자")
            print(f"📊 현재 통계: {self.stats['total_collected']}개 수집됨")
    
    def _ensure_directory(self):
        """디렉터리 생성"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_stats(self):
        """통계 로드"""
        if self.file_path.exists():
            try:
                # 파일 크기 계산
                self.stats["file_size_kb"] = round(self.file_path.stat().st_size / 1024, 2)
                
                # 라인 수 계산 (총 수집 개수)
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                    self.stats["total_collected"] = lines
                    
            except Exception as e:
                print(f"⚠️ 통계 로드 실패: {e}")
    
    def is_valid_conversation(self, user_message: str, assistant_response: str) -> tuple[bool, str]:
        """대화 유효성 검증"""
        if not self.enabled:
            return False, "수집 비활성화됨"
        
        # 길이 검증
        user_len = len(user_message.strip())
        assistant_len = len(assistant_response.strip())
        
        if user_len < self.min_length:
            return False, f"사용자 메시지 너무 짧음 ({user_len}자 < {self.min_length}자)"
        
        if assistant_len < self.min_length:
            return False, f"응답 메시지 너무 짧음 ({assistant_len}자 < {self.min_length}자)"
        
        if user_len > self.max_length:
            return False, f"사용자 메시지 너무 김 ({user_len}자 > {self.max_length}자)"
        
        if assistant_len > self.max_length:
            return False, f"응답 메시지 너무 김 ({assistant_len}자 > {self.max_length}자)"
        
        # 시스템 메시지 필터링
        if self.filter_system:
            system_keywords = [
                "초기화", "설정", "오류", "서버", "모델", "로딩", "API", 
                "시스템", "에러", "debug", "test", "health", "status"
            ]
            
            combined_text = (user_message + " " + assistant_response).lower()
            for keyword in system_keywords:
                if keyword in combined_text:
                    return False, f"시스템 메시지 필터링됨 (키워드: {keyword})"
        
        return True, "유효함"
    
    def collect_conversation(
        self, 
        user_message: str, 
        assistant_response: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """대화 수집"""
        
        # 유효성 검증
        is_valid, reason = self.is_valid_conversation(user_message, assistant_response)
        
        if not is_valid:
            self.stats["filtered_out"] += 1
            print(f"🚫 대화 필터링: {reason}")
            return False
        
        # 대화 엔트리 생성
        entry = ConversationEntry(
            user_message=user_message.strip(),
            assistant_response=assistant_response.strip(),
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # 파일에 저장 (스레드 안전)
        success = self._save_to_file(entry)
        
        if success:
            self.stats["total_collected"] += 1
            self.stats["last_collection"] = entry.timestamp
            print(f"💾 대화 수집 완료: {self.stats['total_collected']}번째")
            return True
        else:
            print(f"❌ 대화 저장 실패")
            return False
    
    def _save_to_file(self, entry: ConversationEntry) -> bool:
        """파일에 안전하게 저장"""
        try:
            with self.lock:
                # JSONL 형태로 저장
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    # 파일 락킹 (다중 프로세스 환경에서 안전)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(entry.to_dict(), f, ensure_ascii=False)
                        f.write('\n')
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # 파일 크기 업데이트
                self.stats["file_size_kb"] = round(self.file_path.stat().st_size / 1024, 2)
            
            return True
            
        except Exception as e:
            print(f"❌ 파일 저장 오류: {e}")
            return False
    
    def get_collected_conversations(self, limit: Optional[int] = None) -> List[ConversationEntry]:
        """수집된 대화 목록 반환"""
        conversations = []
        
        if not self.file_path.exists():
            return conversations
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 최신 순으로 제한
                if limit:
                    lines = lines[-limit:]
                
                for line in lines:
                    if line.strip():
                        data = json.loads(line.strip())
                        conversations.append(ConversationEntry(**data))
            
            return conversations
            
        except Exception as e:
            print(f"❌ 대화 로드 오류: {e}")
            return []
    
    def get_training_data(self, limit: Optional[int] = None) -> List[str]:
        """파인튜닝용 형태로 대화 반환"""
        conversations = self.get_collected_conversations(limit)
        return [conv.to_training_format() for conv in conversations]
    
    def get_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        # 실시간 파일 크기 업데이트
        if self.file_path.exists():
            self.stats["file_size_kb"] = round(self.file_path.stat().st_size / 1024, 2)
        
        return {
            **self.stats,
            "enabled": self.enabled,
            "file_path": str(self.file_path),
            "collection_criteria": {
                "min_length": self.min_length,
                "max_length": self.max_length,
                "filter_system": self.filter_system
            }
        }
    
    def backup_conversations(self) -> bool:
        """대화 데이터 백업"""
        if not self.file_path.exists():
            print("⚠️ 백업할 파일이 없습니다.")
            return False
        
        try:
            # 타임스탬프 포함 백업 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"conversations_backup_{timestamp}.jsonl"
            backup_full_path = Path(self.data_path) / backup_name
            
            # 파일 복사
            import shutil
            shutil.copy2(self.file_path, backup_full_path)
            
            print(f"💾 대화 백업 완료: {backup_full_path}")
            return True
            
        except Exception as e:
            print(f"❌ 백업 실패: {e}")
            return False
    
    def clear_conversations(self, backup_first: bool = True) -> bool:
        """대화 데이터 초기화"""
        if backup_first:
            self.backup_conversations()
        
        try:
            if self.file_path.exists():
                self.file_path.unlink()
            
            # 통계 초기화
            self.stats = {
                "total_collected": 0,
                "filtered_out": 0,
                "last_collection": None,
                "file_size_kb": 0
            }
            
            print("🗑️ 대화 데이터 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            return False
    
    def export_for_finetuning(self, output_path: Optional[str] = None) -> Optional[str]:
        """파인튜닝용 데이터셋 생성"""
        if output_path is None:
            output_path = Path(self.data_path) / "training_dataset.json"
        else:
            output_path = Path(output_path)
        
        conversations = self.get_collected_conversations()
        
        if not conversations:
            print("⚠️ 내보낼 대화가 없습니다.")
            return None
        
        try:
            # 파인튜닝용 데이터 형태로 변환
            training_data = []
            
            for conv in conversations:
                # TARGET 태그를 사용한 벡터 복원 학습 형태
                conversation_text = conv.to_training_format()
                
                training_sample = {
                    "input": f"<TARGET>{conversation_text}</TARGET>TARGET 태그 안의 내용만 출력하세요. 추가 설명은 필요 없습니다.",
                    "output": conversation_text,
                    "metadata": {
                        "timestamp": conv.timestamp,
                        "user_id": conv.user_id,
                        "session_id": conv.session_id,
                        "source": "conversation_collector",
                        "text_length": len(conversation_text)
                    }
                }
                
                training_data.append(training_sample)
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"📊 파인튜닝 데이터셋 생성 완료: {len(training_data)}개 샘플")
            print(f"💾 저장 경로: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 데이터셋 생성 실패: {e}")
            return None
    
    def should_trigger_training(self, batch_size: int) -> bool:
        """파인튜닝 트리거 조건 확인"""
        return self.stats["total_collected"] >= batch_size and self.stats["total_collected"] % batch_size == 0
    
    def get_pending_training_count(self, batch_size: int) -> int:
        """다음 학습까지 남은 대화 수"""
        current = self.stats["total_collected"]
        return batch_size - (current % batch_size) if current % batch_size != 0 else 0


class MLOpsDataManager:
    """MLOps를 위한 통합 데이터 관리자"""
    
    def __init__(self, finetune_config: Dict[str, Any], conversation_config: Dict[str, Any]):
        self.finetune_config = finetune_config
        self.conversation_config = conversation_config
        
        # 대화 수집기 초기화
        self.collector = ConversationCollector(conversation_config)
        
        # 파인튜닝 관련 설정
        self.batch_size = finetune_config.get("batch_size", 50)
        self.auto_trigger = finetune_config.get("auto_trigger", True)
        self.models_path = Path(finetune_config.get("models_path", "./models"))
        self.backup_count = finetune_config.get("backup_count", 3)
        
        # 상태 추적
        self.last_training_count = 0
        self.training_in_progress = False
        
        print(f"🤖 MLOps 데이터 매니저 초기화")
        print(f"📊 배치 크기: {self.batch_size}개 대화")
        print(f"⚡ 자동 트리거: {'ON' if self.auto_trigger else 'OFF'}")
    
    def process_conversation(
        self, 
        user_message: str, 
        assistant_response: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """대화 처리 및 파인튜닝 트리거 확인"""
        
        # 대화 수집
        collected = self.collector.collect_conversation(
            user_message=user_message,
            assistant_response=assistant_response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        result = {
            "collected": collected,
            "total_collected": self.collector.stats["total_collected"],
            "should_train": False,
            "pending_count": 0,
            "training_triggered": False
        }
        
        if collected and self.auto_trigger:
            # 파인튜닝 트리거 확인
            should_train = self.collector.should_trigger_training(self.batch_size)
            result["should_train"] = should_train
            result["pending_count"] = self.collector.get_pending_training_count(self.batch_size)
            
            if should_train and not self.training_in_progress:
                # 자동 파인튜닝 트리거 (비동기 실행)
                result["training_triggered"] = self._trigger_async_training()
        
        return result
    
    def _trigger_async_training(self) -> bool:
        """비동기 파인튜닝 트리거"""
        try:
            import threading
            
            def training_worker():
                self.training_in_progress = True
                try:
                    print("🚀 자동 파인튜닝 시작...")
                    success = self.start_finetuning()
                    if success:
                        print("✅ 자동 파인튜닝 완료!")
                    else:
                        print("❌ 자동 파인튜닝 실패!")
                finally:
                    self.training_in_progress = False
            
            # 백그라운드 스레드에서 실행
            training_thread = threading.Thread(target=training_worker, daemon=True)
            training_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ 파인튜닝 트리거 실패: {e}")
            self.training_in_progress = False
            return False
    
    def start_finetuning(self) -> bool:
        """파인튜닝 시작"""
        try:
            # 1. 데이터셋 생성
            dataset_path = self.collector.export_for_finetuning()
            if not dataset_path:
                print("❌ 데이터셋 생성 실패")
                return False
            
            # 2. 모델 백업 (기존 어댑터가 있다면)
            self._backup_existing_model()
            
            # 3. 파인튜닝 실행 (여기서는 스크립트 호출 시뮬레이션)
            print("🧠 파인튜닝 실행 중...")
            success = self._run_finetuning_script(dataset_path)
            
            if success:
                # 4. 모델 버전 업데이트
                self._update_model_version()
                self.last_training_count = self.collector.stats["total_collected"]
                return True
            else:
                print("❌ 파인튜닝 실행 실패")
                return False
                
        except Exception as e:
            print(f"❌ 파인튜닝 오류: {e}")
            return False
    
    def _backup_existing_model(self):
        """기존 모델 백업"""
        # 구현 예정: 기존 어댑터 백업 로직
        print("💾 기존 모델 백업 중...")
    
    def _run_finetuning_script(self, dataset_path: str) -> bool:
        """파인튜닝 스크립트 실행"""
        # 실제로는 finetuning.py를 subprocess로 실행
        print(f"🚀 파인튜닝 스크립트 실행: {dataset_path}")
        # 시뮬레이션
        import time
        time.sleep(2)  # 실제로는 더 오래 걸림
        return True
    
    def _update_model_version(self):
        """모델 버전 업데이트"""
        # 구현 예정: 버전 관리 로직
        print("🔄 모델 버전 업데이트 중...")
    
    def get_status(self) -> Dict[str, Any]:
        """전체 상태 반환"""
        collector_stats = self.collector.get_stats()
        
        return {
            "collector": collector_stats,
            "training": {
                "batch_size": self.batch_size,
                "auto_trigger": self.auto_trigger,
                "in_progress": self.training_in_progress,
                "last_training_count": self.last_training_count,
                "pending_count": self.collector.get_pending_training_count(self.batch_size),
                "should_train": self.collector.should_trigger_training(self.batch_size)
            },
            "models": {
                "path": str(self.models_path),
                "backup_count": self.backup_count
            }
        }