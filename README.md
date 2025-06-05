# 공식 문서 마스터

## 개요
- 개발 기간 : 2025.05 - 2025.05 (3주)
- 해결하고자 하는 문제 : 공식 문서를 기반으로 신뢰도 높은 자동 응답을 제공하여 사용자 만족도를 높이는 것을 목표로 함

<br>

## 핵심 기능

1.  **프레임워크 추론 및 문서 자동 식별**

    *   사용자의 질문에서 **관련 프레임워크 및 기술 키워드**를 추론하고, 해당 기술의 **공식 문서 URL을 자동 탐색**
    *   필요 시 프레임워크 문서를 **백그라운드에서 자동 크롤링 및 임베딩**

2.  **RAG 기반 답변 생성**

    *   질문과 관련된 공식 문서 내용을 벡터 검색하여, **정확하고 신뢰도 높은 문서 기반 응답 생성**

3.  **자동 재질문(paraphrasing) 및 품질 평가**

    *   문서의 관련도가 낮을 경우 **질문을 자동으로 변형(paraphrase)** 하여 재검색 수행
    *   생성된 레퍼런스 및 답변의 **품질을 자체적으로 평가**하여 재시도 또는 종료 여부 결정

4.  **비동기 작업 큐 기반 문서 수집**

    *   임베딩되지 않은 프레임워크의 경우, **백그라운드 큐에 등록 후 별도 워커가 처리**
    *   진행 상태를 사이드바 UI에서 확인 가능

5.  **관리자 수동 개입 및 프레임워크 문서 관리 도구 제공**

    *   URL 캐싱/수정/검증을 위한 **수동 관리 CLI** 및 향후 확장 가능한 UI 제공 구조 설계

<br>

## 기술 구성
**1) Prompt Engineering**

*   역할 부여하기
*   형식 기법

**2) Azure OpenAI 활용**

*   Tool Calling
*   LangGraph

**3) RAG (Retrieval-Augmented Generation)**

*   RecursiveCharacterTextSplitter
*   Chroma 벡터스토어
*   유사도 점수 임계값 검색

**4) Streamlit**

<br>

## 서비스 아키텍처
```mermaid
graph TD
  User[사용자]

  %% 프론트엔드 (UI)
  User --> Streamlit[Streamlit App - form, 사이드바, 상태 표시]

  %% LangGraph 처리
  Streamlit --> LangGraph[LangGraph Engine - 멀티노드 Agent Flow]

  LangGraph -->|검색 및 문서 수집| Tools[ReAct Tools - DuckDuckGo, 크롤러 등]

  LangGraph -->|벡터 검색| Chroma[ChromaDB - Vector Store]

  LangGraph -->|프롬프트 요청| AzureOpenAI[Azure OpenAI API - GPT-4, GPT-4o]

  %% 큐 및 워커
  LangGraph --> SQLite[SQLite DB - jobs, requests, urls]
  Worker[백그라운드 워커] --> SQLite
  Worker -->|문서 수집 + 임베딩| Chroma

  %% URL 처리
  Tools -->|URL 저장/조회| SQLite

  %% 로깅
  Streamlit --> Logging[(Logging)]
  Worker --> Logging
```

| **컴포넌트**         | **설명**                                        |
| ---------------- | --------------------------------------------- |
| **Streamlit**    | 사용자 질문 입력, 결과 출력 UI                           |
| **LangGraph**    | Agent 흐름 제어 (ref_generator → answer_critic 등) |
| **ReAct Tools**  | 공식문서 URL 탐색, 키워드 기반 문서 수집                     |
| **Chroma**       | 문서 벡터 임베딩 저장소                                 |
| **SQLite**       | 큐 관리 (jobs/requests), URL 캐시                  |
| **Worker**       | 백그라운드 문서 크롤링 및 임베딩 작업 수행                      |
| **Azure OpenAI** | LLM 질의 응답 처리                                  |
| **Logging**      | Streamlit/Worker 공통 로그 저장                     |

<br>

## 사용자 flow 다이어그램
```mermaid
---
config:
  layout: dagre
---
flowchart TD
 subgraph LangGraph["LangGraph"]
    direction LR
        F["ref_generator"]
        E(("START"))
        G["framework_ref"]
        X["END: Invalid"]
        H["general_ref"]
        Y["END: Enqueue Work"]
        I["ref_critic"]
        J["paraphrase"]
        K["answer_generator"]
        L["answer_critic"]
        Z["END"]
  end
 subgraph Worker["Worker"]
        O["작업 조회 pending"]
        N["jobs 테이블 저장"]
        P["crawl_and_embed"]
        Q["Chroma 임베딩"]
        R["작업 상태 done 처리"]
  end
    A["사용자입력"] --> B["start_research"]
    B --> C{"question_id 존재 여부"}
    C --> D["LangGraph 실행"]
    D --> E
    E --> F
    F -- 기술 질문 --> G
    F -- 비기술 질문 --> X
    G -- 문서 있음 --> H
    G -- 문서 없음 --> Y
    H --> I
    I -- 레퍼런스 부족 --> J
    I -- 레퍼런스 충분 --> K
    J --> F
    K --> L
    L -- 적절 --> Z
    L -- 부적절 + 재시도 --> F
    Y --> M["enqueue_request"]
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    Z --> S["Streamlit 결과 출력"]
    X --> S
    S --> T["사이드바 상태 업데이트"]
```
1.  **Streamlit UI**

    *   사용자가 질문을 제출하면 start_research가 실행되고, LangGraph 상태 머신이 시작

2.  **LangGraph 상태 머신**

    *   ref_generator가 질문 유형을 판별
    *   공식 문서 확인 → framework_ref → general_ref 순으로 레퍼런스를 수집·평가하며, 부족하면 paraphrase로 돌아가 재검색을 시도
    *   답변이 생성된 뒤 answer_critic가 품질을 검증하고, 통과(Y)하면 종료, 실패(N)하면 루프를 반복

3.  **큐 & 워커**

    *   framework_ref 단계에서 공식 문서가 없으면 (ENQUEUE_WORK) 경로를 통해 **jobs** 테이블에 작업을 등록하고 즉시 답변 대신 대기 안내를 반환
    *   별도 **Worker** 프로세스가 pending 상태의 잡을 선택해 문서를 크롤링·임베딩한 뒤 상태를 done으로 업데이트

4.  **결과 반환 및 UI 업데이트**

    *   LangGraph가 END에 도달하면 Streamlit에 최종 답변 또는 오류 메시지를 출력하고, 사이드바 뱃지로 진행 상황을 표시

<br>

## DB ERD
![official_docs_agent_erd](https://github.com/user-attachments/assets/d0968239-d0b6-46cf-8637-7c42163451b9)
