import os
from chromadb import PersistentClient
from typing import List, Dict, Any
import time

class VectorService:
    """벡터 검색 및 저장 서비스"""
    
    def __init__(self, kanana_model, config: dict):
        self.kanana_model = kanana_model
        self.chroma_path = config.get("path", "./data/chroma_data")
        self.collection_name = config.get("collection_name", "kanana-docs-optimized")
        self.client = None
        self.collection = None
        
    def initialize(self):
        """ChromaDB 초기화"""
        print("🗄️ ChromaDB 초기화...")
        
        # ChromaDB 클라이언트 생성
        os.makedirs(self.chroma_path, exist_ok=True)
        
        self.client = PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # 기존 문서 수 확인
        doc_count = self.collection.count()
        
        print(f"📚 기존 저장된 문서 수: {doc_count}개")
        print(f"🗄️ ChromaDB 경로: {self.chroma_path}")
        print(f"📦 컬렉션 이름: {self.collection_name}")
        print("✅ ChromaDB 초기화 완료!")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """유사 문서 검색 (시간순 정렬)"""
        if not self.collection:
            raise RuntimeError("VectorService가 초기화되지 않았습니다.")
        
        # 쿼리 임베딩
        query_embedding = self.kanana_model.embed(query)
        
        # 검색 실행 (ID도 함께 가져오기)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )
        
        # 결과 포맷팅
        formatted_results = []
        if results["documents"][0]:
            for i, (doc, score, metadata) in enumerate(zip(
                results["documents"][0],
                results["distances"][0],
                results.get("metadatas", [[{}] * len(results["documents"][0])])[0]
            )):
                # ID에서 타임스탬프 추출: "doc-1736123456.789" -> 1736123456.789
                doc_id = results["ids"][0][i] if "ids" in results else f"doc-{time.time()}"
                timestamp = float(doc_id.split("-")[1]) if "-" in doc_id else time.time()
                
                formatted_results.append({
                    "document": doc,
                    "score": float(score),
                    "metadata": metadata or {},
                    "rank": i + 1,
                    "timestamp": timestamp
                })
        
        # 🎯 시간순 정렬 (오래된 것부터)
        formatted_results.sort(key=lambda x: x["timestamp"])
        
        # timestamp는 내부 용도이므로 제거
        for result in formatted_results:
            result.pop("timestamp")
        
        return formatted_results
    
    def add_document(self, document: str, metadata: Dict[str, Any] = None) -> str:
        """문서 추가"""
        if not self.collection:
            raise RuntimeError("VectorService가 초기화되지 않았습니다.")
        
        # 벡터 생성
        vec_bfloat16 = self.kanana_model.embed_optimized(document)
        vec_float32 = self.kanana_model.bfloat16_to_float32_list(vec_bfloat16)
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        
        # 메모리 사용량 계산
        vector_memory_bfloat16 = vec_bfloat16.numel() * 2  # bfloat16
        vector_memory_float32 = len(vec_float32) * 4  # float32
        
        metadata.update({
            "vector_type": "bfloat16_optimized",
            "original_dtype": "bfloat16", 
            "memory_saved_kb": f"{(vector_memory_float32 - vector_memory_bfloat16)/1024:.2f}",
            "doc_length": len(document)
        })
        
        # 🎯 시간 기반 ID 생성
        doc_id = f"doc-{time.time()}"
        
        # ChromaDB에 저장
        self.collection.add(
            documents=[document],
            embeddings=[vec_float32],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        print(f"💾 문서 저장: {doc_id} (길이: {len(document)}자)")
        return doc_id
    
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        if not self.collection:
            return 0
        return self.collection.count()
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        if not self.collection:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "name": self.collection_name,
            "path": self.chroma_path,
            "count": self.collection.count()
        }
    
    def search_by_id(self, doc_id: str) -> Dict[str, Any]:
        """ID로 문서 검색"""
        if not self.collection:
            raise RuntimeError("VectorService가 초기화되지 않았습니다.")
        
        results = self.collection.get(ids=[doc_id])
        
        if results["documents"]:
            return {
                "id": doc_id,
                "document": results["documents"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
        
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """문서 삭제"""
        if not self.collection:
            raise RuntimeError("VectorService가 초기화되지 않았습니다.")
        
        try:
            self.collection.delete(ids=[doc_id])
            print(f"🗑️ 문서 삭제: {doc_id}")
            return True
        except Exception as e:
            print(f"❌ 문서 삭제 실패: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """모든 문서 삭제"""
        if not self.collection:
            raise RuntimeError("VectorService가 초기화되지 않았습니다.")
        
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print("🗑️ 모든 문서 삭제 완료")
            return True
        except Exception as e:
            print(f"❌ 문서 삭제 실패: {e}")
            return False