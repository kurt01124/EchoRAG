from typing import List, Dict, Any
import json
import os
from datetime import datetime

class MessageQueue:
    """대화 메모리 관리 클래스"""
    
    def __init__(self, cnt=30):
        self.max_count = cnt
        self.messages = []
        self.created_at = datetime.now()
    
    def append(self, message: Dict[str, str]):
        """메시지 추가"""
        if len(self.messages) >= self.max_count:
            self.messages.pop(0)  # 가장 오래된 메시지 제거
        
        # 타임스탬프 추가
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.messages.append(message_with_timestamp)
    
    def view(self) -> str:
        """메모리 내용을 문자열로 반환"""
        if not self.messages:
            return ""
        
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages
        ])
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """전체 메시지 리스트 반환"""
        return self.messages.copy()
    
    def get_recent_messages(self, count: int) -> List[Dict[str, Any]]:
        """최근 N개 메시지 반환"""
        return self.messages[-count:] if count > 0 else []
    
    def clear(self):
        """메모리 초기화"""
        self.messages.clear()
        print("🧠 메모리 초기화 완료")
    
    def save_to_file(self, filepath: str):
        """메모리를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data = {
                "created_at": self.created_at.isoformat(),
                "max_count": self.max_count,
                "message_count": len(self.messages),
                "messages": self.messages
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 메모리 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 메모리 저장 실패: {e}")
            return False
    
    def load_from_file(self, filepath: str):
        """파일에서 메모리 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.messages = data.get("messages", [])
            self.max_count = data.get("max_count", 30)
            
            if "created_at" in data:
                self.created_at = datetime.fromisoformat(data["created_at"])
            
            print(f"📂 메모리 로드 완료: {len(self.messages)}개 메시지")
            return True
            
        except Exception as e:
            print(f"❌ 메모리 로드 실패: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        if not self.messages:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "average_length": 0,
                "memory_usage": "0%"
            }
        
        user_count = sum(1 for msg in self.messages if msg.get("role") == "user")
        assistant_count = sum(1 for msg in self.messages if msg.get("role") == "assistant")
        total_length = sum(len(msg.get("content", "")) for msg in self.messages)
        avg_length = total_length / len(self.messages) if self.messages else 0
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "average_length": round(avg_length, 1),
            "memory_usage": f"{len(self.messages) / self.max_count * 100:.1f}%",
            "max_capacity": self.max_count,
            "created_at": self.created_at.isoformat()
        }
    
    def search_messages(self, keyword: str) -> List[Dict[str, Any]]:
        """키워드로 메시지 검색"""
        matching_messages = []
        
        for i, msg in enumerate(self.messages):
            if keyword.lower() in msg.get("content", "").lower():
                matching_messages.append({
                    **msg,
                    "index": i,
                    "relative_position": f"{i + 1}/{len(self.messages)}"
                })
        
        return matching_messages
    
    def get_conversation_context(self, max_tokens: int = 2000) -> str:
        """토큰 제한을 고려한 대화 컨텍스트 반환"""
        if not self.messages:
            return ""
        
        # 대략적인 토큰 계산 (1토큰 ≈ 4글자)
        context = ""
        token_count = 0
        
        # 최신 메시지부터 역순으로 추가
        for msg in reversed(self.messages):
            content = f"{msg['role']}: {msg['content']}\n"
            content_tokens = len(content) // 4  # 대략적 계산
            
            if token_count + content_tokens > max_tokens:
                break
            
            context = content + context
            token_count += content_tokens
        
        return context.strip()
    
    def __str__(self) -> str:
        """문자열 표현"""
        return self.view()
    
    def __len__(self) -> int:
        """메시지 개수 반환"""
        return len(self.messages)


class ConversationManager:
    """다중 대화 세션 관리"""
    
    def __init__(self, default_memory_size: int = 30):
        self.sessions = {}
        self.default_memory_size = default_memory_size
        self.active_session = None
    
    def create_session(self, session_id: str, memory_size: int = None) -> MessageQueue:
        """새 대화 세션 생성"""
        memory_size = memory_size or self.default_memory_size
        
        if session_id in self.sessions:
            print(f"⚠️ 세션 '{session_id}'가 이미 존재합니다.")
        
        self.sessions[session_id] = MessageQueue(cnt=memory_size)
        self.active_session = session_id
        
        print(f"✅ 새 세션 생성: {session_id}")
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> MessageQueue:
        """세션 가져오기"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            if self.active_session == session_id:
                self.active_session = None
            
            print(f"🗑️ 세션 삭제: {session_id}")
            return True
        
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록 반환"""
        sessions_info = []
        
        for session_id, memory in self.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "message_count": len(memory),
                "created_at": memory.created_at.isoformat(),
                "is_active": session_id == self.active_session,
                **memory.get_statistics()
            })
        
        return sessions_info
    
    def switch_session(self, session_id: str) -> bool:
        """활성 세션 변경"""
        if session_id in self.sessions:
            self.active_session = session_id
            print(f"🔄 활성 세션 변경: {session_id}")
            return True
        
        return False