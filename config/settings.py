import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # === API 설정 ===
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    
    # === 서버 설정 ===
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")
    reload: bool = Field(default=True, env="RELOAD")
    
    # === 모델 설정 ===
    kanana_model_name: str = Field(
        default="kakaocorp/kanana-1.5-2.1b-instruct-2505", 
        env="KANANA_MODEL_NAME"
    )
    kanana_finetuned_path: Optional[str] = Field(
        default="./kanana-vector-restoration", 
        env="KANANA_FINETUNED_PATH"
    )
    device: str = Field(default="auto", env="DEVICE")  # auto, cuda, cpu
    dtype: str = Field(default="bfloat16", env="MODEL_DTYPE")  # 이름 변경
    
    # === 데이터베이스 설정 ===
    chroma_data_path: str = Field(default="./data/chroma_data", env="CHROMA_DATA_PATH")
    chroma_collection_name: str = Field(
        default="kanana-docs-optimized", 
        env="CHROMA_COLLECTION_NAME"
    )
    
    # === 메모리 설정 ===
    memory_max_count: int = Field(default=30, env="MEMORY_MAX_COUNT")
    memory_auto_save: bool = Field(default=True, env="MEMORY_AUTO_SAVE")
    memory_save_path: str = Field(default="./data/memory", env="MEMORY_SAVE_PATH")
    
    # === 검색 설정 ===
    search_default_results: int = Field(default=3, env="SEARCH_DEFAULT_RESULTS")
    search_max_results: int = Field(default=10, env="SEARCH_MAX_RESULTS")
    
    # === 로깅 설정 ===
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_path: str = Field(default="./logs/app.log", env="LOG_FILE_PATH")
    log_file_rotation: str = Field(default="10 MB", env="LOG_FILE_ROTATION")
    
    # === CORS 설정 ===
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # === 성능 설정 ===
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 초
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ('settings_',),  # Pydantic 충돌 해결
        "env_nested_delimiter": "__",
        "extra": "ignore"  # 추가 필드 무시
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
        self._create_directories()
    
    def _validate_settings(self):
        """설정 값 유효성 검사"""
        if not self.openai_api_key:
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        
        if self.device not in ["auto", "cuda", "cpu"]:
            print(f"⚠️ 알 수 없는 디바이스 설정: {self.device}. 'auto'로 변경합니다.")
            self.device = "auto"
        
        if self.memory_max_count <= 0:
            raise ValueError("MEMORY_MAX_COUNT는 0보다 커야 합니다.")
    
    def _create_directories(self):
        """필요한 디렉터리 생성"""
        directories = [
            self.chroma_data_path,
            self.memory_save_path,
            os.path.dirname(self.log_file_path) if self.log_file_path else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> dict:
        """모델 설정 딕셔너리 반환"""
        return {
            "model_name": self.kanana_model_name,
            "finetuned_path": self.kanana_finetuned_path,
            "device": self.device,
            "dtype": self.dtype
        }
    
    def get_chroma_config(self) -> dict:
        """ChromaDB 설정 딕셔너리 반환"""
        return {
            "path": self.chroma_data_path,
            "collection_name": self.chroma_collection_name
        }
    
    def get_openai_config(self) -> dict:
        """OpenAI 설정 딕셔너리 반환"""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "max_tokens": self.openai_max_tokens
        }
    
    def print_settings_summary(self):
        """설정 요약 출력"""
        print("\n🔧 === 시스템 설정 ===")
        print(f"🌐 서버: {self.host}:{self.port}")
        print(f"🤖 모델: {self.kanana_model_name}")
        print(f"🔥 디바이스: {self.device}")
        print(f"🗄️ ChromaDB: {self.chroma_data_path}")
        print(f"🧠 메모리: {self.memory_max_count}개 메시지")
        print(f"🔍 검색: 기본 {self.search_default_results}개 결과")
        print(f"📝 로그: {self.log_level} → {self.log_file_path}")
        print(f"🌍 CORS: {', '.join(self.allowed_origins)}")
        print("🔧 ====================\n")

# 전역 설정 인스턴스
settings = Settings()

# 설정 로드 확인 함수
def check_environment():
    """환경 설정 상태 확인"""
    issues = []
    
    # API 키 확인
    if not settings.openai_api_key:
        issues.append("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
    else:
        print("✅ OpenAI API 키 로드됨")
    
    # 모델 경로 확인
    if settings.kanana_finetuned_path and not Path(settings.kanana_finetuned_path).exists():
        issues.append(f"⚠️ 파인튜닝 모델 경로를 찾을 수 없습니다: {settings.kanana_finetuned_path}")
    
    # 디렉터리 권한 확인
    for path in [settings.chroma_data_path, settings.memory_save_path]:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            test_file = Path(path) / "test_write"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"❌ 디렉터리 쓰기 권한 없음: {path} ({e})")
    
    if issues:
        print("\n🚨 === 환경 설정 문제 ===")
        for issue in issues:
            print(issue)
        print("🚨 ========================\n")
        return False
    else:
        print("✅ 모든 환경 설정이 올바릅니다!")
        return True