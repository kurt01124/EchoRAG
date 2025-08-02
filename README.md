# 🧠 EchoRAG: 의미를 기억하고 되돌리는 한국어 특화 RAG 시스템

EchoRAG은 한국어에 특화된 KANANA 임베딩 모델과 GPT-4o-mini를 활용하여, 의미 기반 검색과 응답을 제공하는 지능형 RAG 시스템입니다. 단순 검색을 넘어, 저장된 벡터를 의미 단위로 해석하고, 원문에 가까운 정보를 되돌려주는 "역벡터화 기반의 기억 검색 시스템"을 구현합니다.

---

## 🚀 주요 특징

| 기능                        | 설명                                              |
| ------------------------- | ----------------------------------------------- |
| **KANANA 기반 임베딩**         | 한국어 의미 임베딩 성능이 뛰어난 Kakaocorp의 KANANA 모델 사용      |
| **bfloat16 최적화 저장**       | 벡터 저장 공간을 절약하면서도 의미 정보 손실 최소화                   |
| **역벡터 복원 시도**             | 벡터를 다시 원문으로 복원하려는 기능 내장 (완전한 재현은 아니지만 의미 해석 가능) |
| **OpenAI GPT-4o-mini 연동** | GPT API와 결합한 대화형 응답 제공                          |
| **ChromaDB 통합**           | 영속적인 벡터 저장 및 검색 처리                              |
| **직관적 Web UI 제공**         | 빠르고 직관적인 채팅 프론트엔드 포함 (HTML/JS 단독 구성)            |

---

## 📁 프로젝트 구조

```
EchoRAG/
├── backend/
│   ├── models/kanana_model.py         # KANANA 모델 로딩 및 임베딩 처리
│   ├── services/vector_service.py     # 벡터 저장/검색 서비스 (ChromaDB)
│   ├── services/gpt_service.py        # OpenAI GPT API 관리
│   ├── utils/memory_manager.py        # 대화 메모리 관리
│   └── app.py                         # FastAPI 서버 진입점
├── frontend/
│   ├── index.html                     # 사용자 인터페이스 (단일 페이지)
│   └── app.js                         # 채팅 로직 및 상태 관리
└── README.md
```

---

## 🔧 설치 및 실행 방법

### 1. Python 백엔드 실행

```bash
# 환경설정
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 서버 실행
cd backend
python app.py
```

### 2. 프론트엔드 실행

```bash
# VSCode Live Server 또는 브라우저에서 frontend/index.html 열기
```

---

## 🧪 API 엔드포인트 요약

| 메서드    | 경로        | 설명                |
| ------ | --------- | ----------------- |
| GET    | `/health` | 서버 상태 및 모델 로딩 확인  |
| POST   | `/chat`   | 채팅 요청 및 RAG 응답 생성 |
| GET    | `/memory` | 대화 메모리 상태 조회      |
| DELETE | `/memory` | 대화 메모리 초기화        |
| GET    | `/stats`  | 벡터 검색/응답 통계 조회    |

---

## 📌 시스템 아키텍처

* ✅ **사용자 입력**
  └── FastAPI 수신 → KANANA 임베딩 → ChromaDB 검색
* ✅ **GPT 응답 생성**
  └── 검색 결과 + 대화 메모리 + 입력 → GPT → 응답 생성
* ✅ **벡터 저장**
  └── 응답을 다시 벡터화 → bfloat16 최적화 저장
* ✅ **역벡터 해석** *(추가 기능)*
  └── 저장된 벡터를 의미로 재해석 (근접한 문맥 유추)

---

## 📉 잠재적 문제점

* bfloat16 벡터라도 고차원(1024+)이므로, 수백/수천 개 저장 시 공간 부담 존재
* 역벡터화는 완전한 복원이 아닌 경향성 추론이므로 정밀 용도엔 부적합
* KANANA 모델의 로딩 속도는 GPU 메모리에 따라 제한될 수 있음

---

## 📚 사용 사례

* 의미 기반 검색 시스템 (RAG 기반 챗봇)
* 개인정보를 포함한 대화 기록 벡터화 후, 개인화 AI로 활용
* 검색 가능한 회의록/문서 보관 시스템

---

## 🔮 향후 개발 계획

* [ ] 벡터 압축 기법 적용 (e.g., PCA, PQ)
* [ ] 역벡터화 기능 고도화 (텍스트 복원 정확도 향상)
* [ ] 세션 기반 사용자 분리
* [ ] 문서 업로드 및 자동 분할 임베딩 기능 추가

---

## 🧾 라이선스

본 프로젝트는 **MIT 라이선스** 하에 오픈소스입니다.
KANANA 모델의 라이선스는 Kakao Corp의 규정에 따릅니다.

---

## 🙌 기여

기여는 언제든 환영합니다!

* PR / 이슈 등록 / 기능 제안 모두 환영
* 성능 개선, UI 개선, 사용 사례 공유 모두 큰 도움이 됩니다!

---

## 👤 제작자

* 개발자: \[kurtz01124]
* 이메일: [kurtz01124@gmail.com](mailto:kurtz01124@gmail.com)
* LinkedIn / GitHub: 추가 예정

---

**EchoRAG — 의미를 기억하고, 되살리는 AI 시스템**
