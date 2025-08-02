import torch
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

class CustomDataCollator:
    """커스텀 데이터 콜레이터"""
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            padded_lbls = lbls + [-100] * padding_length
            padded_labels.append(padded_lbls)
            
            attention_mask = [1] * len(ids) + [0] * padding_length
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

class AutomatedFinetuner:
    """자동화된 파인튜닝 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 기본 설정
        self.model_name = config.get("model_name", "kakaocorp/kanana-1.5-2.1b-instruct-2505")
        self.base_output_dir = Path(config.get("models_path", "./models"))
        self.data_path = Path(config.get("data_path", "./data/finetune"))
        self.backup_count = config.get("backup_count", 3)
        self.version_prefix = config.get("version_prefix", "v")
        
        # 하이퍼파라미터
        self.hyperparams = config.get("hyperparameters", {})
        self.epochs = self.hyperparams.get("epochs", 1)
        self.learning_rate = self.hyperparams.get("learning_rate", 2e-5)
        self.lora_r = self.hyperparams.get("lora_r", 8)
        self.lora_alpha = self.hyperparams.get("lora_alpha", 16)
        self.lora_dropout = self.hyperparams.get("lora_dropout", 0.1)
        
        # 상태 추적
        self.current_version = self._get_next_version()
        self.training_log = []
        
        # 디렉터리 생성
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 자동화된 파인튜너 초기화")
        print(f"📁 모델 저장 경로: {self.base_output_dir}")
        print(f"🏷️ 다음 버전: {self.current_version}")
        print(f"🎛️ 하이퍼파라미터: epochs={self.epochs}, lr={self.learning_rate}, r={self.lora_r}")
    
    def _get_next_version(self) -> str:
        """다음 버전 번호 계산"""
        existing_versions = []
        
        # 기존 모델 디렉터리에서 버전 번호 찾기
        if self.base_output_dir.exists():
            for item in self.base_output_dir.iterdir():
                if item.is_dir() and item.name.startswith(self.version_prefix):
                    try:
                        version_num = int(item.name[len(self.version_prefix):])
                        existing_versions.append(version_num)
                    except ValueError:
                        continue
        
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
        
        return f"{self.version_prefix}{next_version}"
    
    def _get_latest_model_path(self) -> Optional[Path]:
        """최신 모델 경로 반환"""
        if not self.base_output_dir.exists():
            return None
        
        latest_version = 0
        latest_path = None
        
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.name.startswith(self.version_prefix):
                try:
                    version_num = int(item.name[len(self.version_prefix):])
                    if version_num > latest_version:
                        latest_version = version_num
                        latest_path = item
                except ValueError:
                    continue
        
        return latest_path
    
    def _backup_existing_models(self):
        """기존 모델들 백업 관리"""
        models = []
        
        # 모든 버전 모델 찾기
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.name.startswith(self.version_prefix):
                try:
                    version_num = int(item.name[len(self.version_prefix):])
                    models.append((version_num, item))
                except ValueError:
                    continue
        
        # 버전 순으로 정렬
        models.sort(key=lambda x: x[0])
        
        # 백업 개수 초과 시 오래된 모델 삭제
        while len(models) >= self.backup_count:
            old_version, old_path = models.pop(0)
            if old_path.exists():
                print(f"🗑️ 오래된 모델 삭제: {old_path.name}")
                shutil.rmtree(old_path)
    
    def _convert_conversations_to_training_format(self, dataset_path: str) -> str:
        """대화 데이터를 파인튜닝 형태로 변환"""
        print(f"🔄 학습 데이터 변환: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미 올바른 형태라면 그대로 사용
        if isinstance(data, list) and len(data) > 0:
            if 'input' in data[0] and 'output' in data[0]:
                print("✅ 이미 올바른 학습 데이터 형태입니다.")
                return dataset_path
        
        # 대화 형태라면 변환
        training_data = []
        
        for item in data:
            if isinstance(item, dict):
                if 'to_training_format' in item:
                    # ConversationEntry 형태
                    conversation_text = item['to_training_format']()
                elif 'user_message' in item and 'assistant_response' in item:
                    # 직접 대화 형태
                    conversation_text = f"USER : {item['user_message']}<\\n>ASSISTANT : {item['assistant_response']}"
                else:
                    continue
                
                # 올바른 대화 형식으로 변환
                training_sample = {
                    "text": self._create_chat_format(conversation_text),
                    "metadata": item.get('metadata', {})
                }
                
                training_data.append(training_sample)
        
        # 변환된 데이터 저장
        converted_path = self.data_path / f"converted_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(converted_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 변환 완료: {len(training_data)}개 샘플 → {converted_path}")
        return str(converted_path)
    
    def _create_chat_format(self, conversation_text: str) -> str:
        """대화를 올바른 채팅 형태로 변환"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 문장을 그대로 읽어주는 친절한 AI 비서입니다.<|eot_id|><|start_header_id|>user<|end_header_id|>

<TARGET>{conversation_text}</TARGET>TARGET 태그 안의 내용만 출력하세요.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{conversation_text}<|eot_id|>"""
    
    def _setup_model_and_tokenizer(self, existing_adapter_path: Optional[str] = None):
        """모델과 토크나이저 설정 (기존 어댑터 로드 포함)"""
        print(f"🚀 모델 로딩: {self.model_name}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            use_cache=False,
        )
        
        # 기존 어댑터가 있다면 로드
        if existing_adapter_path and Path(existing_adapter_path).exists():
            print(f"🔄 기존 어댑터 로드: {existing_adapter_path}")
            try:
                model = PeftModel.from_pretrained(base_model, existing_adapter_path)
                print("✅ 기존 어댑터 로드 성공 - 점진적 학습 진행")
            except Exception as e:
                print(f"⚠️ 기존 어댑터 로드 실패: {e}")
                print("📦 베이스 모델로 새로 시작")
                model = base_model
        else:
            print("📦 베이스 모델로 새로 시작")
            model = base_model
        
        print(f"💾 모델 파라미터 수: {model.num_parameters():,}")
        return model, tokenizer
    
    def _setup_lora_config(self):
        """LoRA 설정"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        
        print("🎯 LoRA 설정:")
        print(f"- Rank (r): {lora_config.r}")
        print(f"- Alpha: {lora_config.lora_alpha}")
        print(f"- Dropout: {lora_config.lora_dropout}")
        
        return lora_config
    
    def _load_and_prepare_dataset(self, tokenizer, dataset_file: str):
        """데이터셋 로드 및 전처리 - 완전히 올바른 라벨 마스킹"""
        print(f"📂 데이터셋 로딩: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 총 {len(data)}개의 학습 샘플")
        
        dataset = Dataset.from_list(data)
        
        def tokenize_function(examples):
            """🔥 완전히 올바른 토크나이징 및 라벨 마스킹"""
            model_inputs = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
                return_overflowing_tokens=False,
            )
            
            labels = []
            
            for i, text in enumerate(examples["text"]):
                input_ids = model_inputs["input_ids"][i]
                
                # 🎯 핵심: 문자열 기반으로 assistant 부분 찾기
                text_str = text
                
                # assistant 헤더 부분과 응답 부분 분리
                assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                
                if assistant_header in text_str:
                    # assistant 응답 시작 위치 찾기
                    assistant_start_pos = text_str.find(assistant_header) + len(assistant_header)
                    
                    # 마지막 <|eot_id|> 찾기
                    last_eot_pos = text_str.rfind("<|eot_id|>")
                    
                    if last_eot_pos > assistant_start_pos:
                        # assistant 응답 부분만 추출
                        assistant_response = text_str[assistant_start_pos:last_eot_pos]
                        
                        # 🔥 핵심: 전체 텍스트에서 assistant 응답 부분의 토큰 위치 찾기
                        before_assistant = text_str[:assistant_start_pos]
                        
                        # before_assistant를 토크나이징해서 길이 확인
                        before_tokens = tokenizer(before_assistant, add_special_tokens=False)["input_ids"]
                        assistant_tokens = tokenizer(assistant_response, add_special_tokens=False)["input_ids"]
                        
                        # labels 초기화
                        label = [-100] * len(input_ids)
                        
                        # 🎯 중요: 실제 토큰 인덱스에서 assistant 부분 찾기
                        if len(assistant_tokens) > 0:
                            # assistant 토큰들이 전체 input_ids에서 시작하는 위치 찾기
                            for start_idx in range(len(input_ids) - len(assistant_tokens) + 1):
                                # assistant_tokens와 일치하는 부분 찾기
                                if input_ids[start_idx:start_idx + len(assistant_tokens)] == assistant_tokens:
                                    # 찾았으면 해당 부분만 학습 대상으로 설정
                                    for j in range(start_idx, start_idx + len(assistant_tokens)):
                                        if j < len(input_ids):
                                            label[j] = input_ids[j]
                                    break
                            else:
                                # 매칭 실패 시 대안: 뒤에서부터 찾기
                                estimated_start = len(input_ids) - len(assistant_tokens) - 2
                                if estimated_start > 0:
                                    for j in range(estimated_start, len(input_ids) - 1):
                                        if j >= 0:
                                            label[j] = input_ids[j]
                    else:
                        print(f"Warning: assistant 응답 부분을 찾지 못했습니다")
                        label = [-100] * len(input_ids)
                else:
                    print(f"Warning: assistant 헤더가 없습니다")
                    label = [-100] * len(input_ids)
                
                labels.append(label)
            
            model_inputs["labels"] = labels
            return model_inputs
        
        # 배치 토크나이징
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 훈련/검증 분할
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"✅ 데이터 전처리 완료:")
        print(f"- 훈련 샘플: {len(split_dataset['train'])}개")
        print(f"- 검증 샘플: {len(split_dataset['test'])}개")
        
        return split_dataset['train'], split_dataset['test']
    
    def _setup_training_arguments(self, output_dir: str):
        """훈련 설정"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # 설정 가능한 하이퍼파라미터
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            
            # 학습률
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=5,
            
            # 저장/평가
            save_strategy="epoch",
            save_total_limit=1,
            eval_strategy="epoch",
            
            # 로깅
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            
            # 기타
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],
            
            # 정밀도
            fp16=True,
            bf16=False,
            
            # 메모리 최적화
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            max_grad_norm=0.5,
            
            load_best_model_at_end=False,
        )
        
        print("🎛️ 훈련 설정:")
        print(f"- 에폭: {training_args.num_train_epochs}")
        print(f"- 학습률: {training_args.learning_rate}")
        print(f"- 출력 경로: {output_dir}")
        
        return training_args
    
    def run_automated_finetuning(self, dataset_path: str) -> Dict[str, Any]:
        """자동화된 파인튜닝 실행"""
        start_time = datetime.now()
        output_dir = self.base_output_dir / self.current_version
        
        try:
            print("🚀 자동화된 파인튜닝 시작!")
            print("=" * 70)
            
            # 1. 기존 모델 백업 관리
            print("💾 기존 모델 백업 관리...")
            self._backup_existing_models()
            
            # 2. 데이터 변환
            print("🔄 학습 데이터 준비...")
            converted_dataset_path = self._convert_conversations_to_training_format(dataset_path)
            
            # 3. 기존 최신 어댑터 찾기
            latest_model_path = self._get_latest_model_path()
            existing_adapter_path = str(latest_model_path) if latest_model_path else None
            
            # 4. 모델과 토크나이저 설정
            model, tokenizer = self._setup_model_and_tokenizer(existing_adapter_path)
            
            # 5. LoRA 설정 (기존 어댑터가 없을 때만)
            if not isinstance(model, PeftModel):
                lora_config = self._setup_lora_config()
                model = get_peft_model(model, lora_config)
            
            model.train()
            
            # 훈련 가능한 파라미터 확인
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            
            # 6. 데이터셋 준비
            train_dataset, eval_dataset = self._load_and_prepare_dataset(tokenizer, converted_dataset_path)
            
            # 7. 데이터 콜레이터
            data_collator = CustomDataCollator(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
            )
            
            # 8. 훈련 설정
            training_args = self._setup_training_arguments(str(output_dir))
            
            # 9. 트레이너 설정
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # 10. 훈련 시작!
            print(f"\n🎯 훈련 시작 (버전: {self.current_version})...")
            training_output = trainer.train()
            
            # 11. 모델 저장
            print("\n💾 모델 저장 중...")
            trainer.save_model()
            tokenizer.save_pretrained(str(output_dir))
            
            # 12. 훈련 로그 기록
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            log_entry = {
                "version": self.current_version,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "training_time_seconds": training_time,
                "dataset_path": dataset_path,
                "training_samples": len(train_dataset),
                "hyperparameters": {
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                },
                "output_path": str(output_dir),
                "success": True
            }
            
            self.training_log.append(log_entry)
            self._save_training_log()
            
            print(f"✅ 파인튜닝 완료! (버전: {self.current_version})")
            print(f"📁 저장 경로: {output_dir}")
            print(f"⏱️ 소요 시간: {training_time:.2f}초")
            
            # 다음 버전 번호 업데이트
            self.current_version = self._get_next_version()
            
            return {
                "success": True,
                "version": log_entry["version"],
                "output_path": str(output_dir),
                "training_time": training_time,
                "training_samples": len(train_dataset),
                "log_entry": log_entry
            }
            
        except Exception as e:
            # 실패 로그 기록
            end_time = datetime.now()
            error_log = {
                "version": self.current_version,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": str(e),
                "success": False
            }
            
            self.training_log.append(error_log)
            self._save_training_log()
            
            print(f"❌ 파인튜닝 실패: {e}")
            raise e
    
    def _save_training_log(self):
        """훈련 로그 저장"""
        log_path = self.base_output_dir / "training_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, ensure_ascii=False, indent=2)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """훈련 히스토리 반환"""
        return self.training_log.copy()
    
    def get_model_versions(self) -> List[Dict[str, Any]]:
        """모델 버전 목록 반환"""
        versions = []
        
        if not self.base_output_dir.exists():
            return versions
        
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.name.startswith(self.version_prefix):
                try:
                    version_num = int(item.name[len(self.version_prefix):])
                    
                    # 모델 정보 수집
                    version_info = {
                        "version": item.name,
                        "version_number": version_num,
                        "path": str(item),
                        "created_time": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                        "size_mb": sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
                    }
                    
                    versions.append(version_info)
                    
                except ValueError:
                    continue
        
        # 버전 번호로 정렬
        versions.sort(key=lambda x: x["version_number"], reverse=True)
        return versions