import torch
import time
import base64
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class KananaModel:
    """Kanana 모델 관리 클래스"""
    
    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.get("device") == "cpu":
            self.device = "cpu"
        elif config.get("device") == "cuda":
            self.device = "cuda"
        
        self.model_name = config.get("model_name", "kakaocorp/kanana-1.5-2.1b-instruct-2505")
        self.finetuned_path = config.get("finetuned_path", "./kanana-vector-restoration")
        self.dtype_str = config.get("dtype", "bfloat16")
        
        # dtype 설정
        if self.dtype_str == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.dtype_str == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.embedding_layer = None
        
        print(f"🔥 디바이스: {self.device}, 데이터 타입: {self.dtype_str}")
    
    def load_model(self):
        """모델과 토크나이저 로딩"""
        print("🚀 Kanana 모델 로딩 시작...")
        start_time = time.time()
        
        try:
            # 1. 토크나이저 로딩
            print("📝 토크나이저 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. 베이스 모델 로딩
            print("🧠 베이스 모델 로딩...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True
            ).eval().to(self.device)
            
            # 3. 파인튜닝된 모델 로딩 (있는 경우)
            try:
                print("🎯 파인튜닝된 모델 로딩...")
                self.model = PeftModel.from_pretrained(self.base_model, self.finetuned_path)
                print("✅ 파인튜닝된 모델 로딩 성공")
            except Exception as e:
                print(f"⚠️ 파인튜닝된 모델 로딩 실패: {e}")
                print("📦 베이스 모델로 진행...")
                self.model = self.base_model
            
            # 4. 임베딩 레이어 설정
            self.embedding_layer = self.model.get_input_embeddings()
            
            load_time = time.time() - start_time
            print(f"✅ Kanana 모델 로딩 완료: {load_time:.2f}초")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise e
    
    def embed_optimized(self, text: str):
        """bfloat16을 유지하는 최적화된 임베딩 함수"""
        if not self.tokenizer or not self.embedding_layer:
            raise RuntimeError("모델이 로딩되지 않았습니다.")
        
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        vectors = self.embedding_layer(ids).squeeze(0)
        result_bfloat16 = vectors.mean(dim=0).detach()  # bfloat16 유지
        return result_bfloat16
    
    def embed(self, text: str):
        """기존 호환성을 위한 float32 함수 (ChromaDB 검색용)"""
        bfloat16_result = self.embed_optimized(text)
        float32_list = self.bfloat16_to_float32_list(bfloat16_result)
        return float32_list
    
    def embed_text(self, text: str):
        """텍스트 임베딩"""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        result = self.embedding_layer(ids)
        return result
    
    def embed_token(self, token: str):
        """토큰 임베딩"""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        result = self.embedding_layer(torch.tensor([[token_id]]).to(self.device))
        return result
    
    def embed_sentence(self, sentence_text: str):
        """문장 전체 임베딩 (채팅 형식)"""
        bos               = self.embed_token("<|begin_of_text|>")
        sys_head          = self.embed_token("<|start_header_id|>")
        sys_role          = self.embed_text("system")
        sys_tail          = self.embed_token("<|end_header_id|>")
        sys_content       = self.embed_text("당신은 문장을 그대로 읽어주는 친절한 AI 비서입니다.")
        sys_eot           = self.embed_token("<|eot_id|>")
        user_head         = self.embed_token("<|start_header_id|>")
        user_role         = self.embed_text("user")
        user_tail         = self.embed_token("<|end_header_id|>")
        sentence_embed    = self.embed_text(sentence_text)
        suffix_embed      = self.embed_text("이 문장을 최대한 원본과 똑같이 읽어주세요.")
        user_eot          = self.embed_token("<|eot_id|>")
        asst_head         = self.embed_token("<|start_header_id|>")
        asst_role         = self.embed_text("assistant")
        asst_tail         = self.embed_token("<|end_header_id|>")
        
        result = torch.cat([
            bos,
            sys_head, sys_role, sys_tail,
            sys_content, sys_eot,
            user_head, user_role, user_tail,
            sentence_embed, suffix_embed, user_eot,
            asst_head, asst_role, asst_tail
        ], dim=1)
        
        return result
    
    def generate_text(self, input_text: str, max_new_tokens: int = 100):
        """텍스트 생성"""
        if not self.tokenizer or not self.model:
            raise RuntimeError("모델이 로딩되지 않았습니다.")
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 결과 추출
        if "assistant" in response:
            result = response.split("assistant")[-1].strip()
        else:
            result = response.strip()
        
        return result
    
    # 🔧 유틸리티 함수들
    @staticmethod
    def serialize_bfloat16_vector(tensor):
        """bfloat16 텐서를 ChromaDB 저장용 문자열로 변환"""
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        
        bytes_data = tensor.cpu().numpy().tobytes()
        return base64.b64encode(bytes_data).decode('ascii')
    
    @staticmethod
    def deserialize_bfloat16_vector(encoded_str, shape):
        """ChromaDB에서 가져온 문자열을 bfloat16 텐서로 복원"""
        bytes_data = base64.b64decode(encoded_str.encode('ascii'))
        numpy_array = np.frombuffer(bytes_data, dtype=np.uint16)
        
        tensor = torch.from_numpy(numpy_array.view(np.dtype('>u2'))).view(torch.bfloat16)
        return tensor.reshape(shape)
    
    @staticmethod
    def bfloat16_to_float32_list(tensor):
        """bfloat16 텐서를 ChromaDB 검색용 float32 리스트로 변환"""
        return tensor.to(torch.float32).cpu().numpy().tolist()
    
    def get_model_info(self):
        """모델 정보 반환"""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "finetuned_path": self.finetuned_path,
            "has_finetuned": isinstance(self.model, PeftModel),
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "unknown",
            "vocab_size": len(self.tokenizer) if self.tokenizer else "unknown"
        }