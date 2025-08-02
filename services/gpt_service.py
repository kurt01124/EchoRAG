import os
from openai import OpenAI
from typing import List, Dict, Any

class GPTService:
    """GPT API 관리 서비스"""
    
    def __init__(self, config: dict):
        # 설정에서 API 키 및 옵션 가져오기
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # 시스템 프롬프트
        self.system_prompt = {
            "role": "system",
            "content": "당신은 친절하고 정확한 AI 비서입니다. 가능한 한 자세하고 명확하게 답변해주세요."
        }
        
        print(f"🧠 GPT 서비스 초기화 완료! (모델: {self.model})")
    
    async def generate_response(
        self,
        user_message: str,
        search_results: List[Dict[str, Any]] = None,
        memory_content: str = "",
        model: str = None,
        temperature: float = None
    ) -> str:
        """GPT를 사용해 응답 생성"""
        
        # 매개변수 기본값 설정
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        
        # 검색 결과 포맷팅
        search_text = ""
        if search_results:
            search_text = "\n".join([
                f"[score: {result['score']:.4f}] {result['document']}"
                for result in search_results
            ])
        
        # 프롬프트 구성
        prompt = f"""당신은 유저의 맥락을 이해하고, 과거 대화 및 특징을 기반으로 지능적인 답변을 생성하는 AI 비서입니다.

[검색 결과]
{search_text if search_text else "❌ 검색 결과 없음"}

[단기 기억]
{memory_content if memory_content else "❌ 없음"}

--- 유저가 입력한 메시지에 응답하세요:
유저입력 : {user_message}
너 : 
"""
        
        # GPT API 호출용 메시지 구성
        messages = [
            self.system_prompt,
            {"role": "user", "content": prompt}
        ]
        
        try:
            # GPT API 호출
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            reply = response.choices[0].message.content.strip()
            return reply
            
        except Exception as e:
            error_msg = f"GPT API 오류: {str(e)}"
            print(f"❌ {error_msg}")
            return f"죄송합니다. 현재 응답을 생성할 수 없습니다. ({error_msg})"
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id.lower()]
        except Exception as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]  # 기본 모델들
    
    def validate_api_key(self) -> bool:
        """API 키 유효성 검사"""
        try:
            # 간단한 요청으로 API 키 검증
            self.client.models.list()
            return True
        except Exception as e:
            print(f"❌ API 키 검증 실패: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 반환"""
        return {
            "status": "initialized",
            "api_key_valid": self.validate_api_key(),
            "available_models": self.get_available_models(),
            "default_model": "gpt-4o-mini",
            "system_prompt": self.system_prompt["content"]
        }