# Skill Injection MCP

AI 에이전트가 여러 개의 `SKILL.md` 파일 중 현재 요청에 가장 적합한 스킬을 찾아주는 MCP 서버입니다.

단순 키워드 검색뿐 아니라 의미 기반 검색과 검증을 통해 **실제로 해당 작업을 지원하는 스킬인지 근거와 함께 판단**합니다.

Python으로 작성되었으며 Windows와 Linux에서 실행할 수 있습니다.

> Python 3.11 이상 필요

---

## What is this?

AI 에이전트에게 다음과 같은 요청이 들어왔다고 가정합니다.

> Python에서 BM25 기반 전문 검색을 구현해줘.

이 서버는 등록된 여러 `SKILL.md` 파일을 검색하고, 해당 작업을 지원하는 스킬을 찾아 반환합니다.

검색 결과는 단순히 "비슷해 보이는 스킬"을 반환하는 것이 아니라 다음 정보를 포함합니다.

* 지원 여부
* 후보 스킬
* 충족하지 못한 요구사항
* 근거 문장
* 스킬 문서 내 인용 위치
* 의존성
* 검증 상태

---

# Key Features

## Hybrid Search

두 가지 검색 방식을 함께 사용합니다.

### Lexical Search

요청에 포함된 단어가 실제 스킬 문서에 존재하는지 검색합니다.

### Semantic Search

요청과 스킬 문서를 임베딩하여 의미적으로 유사한 스킬을 찾습니다.

두 검색 결과는 **RRF(Reciprocal Rank Fusion)** 방식으로 결합됩니다.

* 각 검색 채널 상위 20개 후보 사용
* 동일 점수는 항상 동일한 순서로 정렬
* 기본 검색 결과 표시 개수: `top_k = 5`

---

## Evidence-Based Verification

후보 스킬은 다음 상태 중 하나로 판정됩니다.

* `supported`
* `partial`
* `unsupported`
* `unknown`

각 판정에는 해당 스킬 문서에서 추출한 근거가 포함됩니다.

또한 서버는 다음을 검증합니다.

* 인용된 문장이 실제 후보 문서에 존재하는지
* 인용 출처가 올바른 후보인지
* 판정과 근거가 일관되는지

검증 과정에서 오류가 발생하면 결과를 억지로 통과시키지 않고 `unknown`으로 처리합니다.

---

## Optional Semantic Verification

선택적으로 AI 모델을 사용해 후보 스킬을 검증할 수 있습니다.

기본 모델:

```text
openai/gpt-oss-120b
```

검증기는 다음 내용을 함께 비교합니다.

* 원본 요구사항
* 후보 스킬 설명
* 후보 스킬 전체 본문
* 스킬이 지원하지 않는 작업
* 요청의 제약 조건

단순 키워드 일치나 임베딩 점수만으로 판단하지 않고 **실제 작업과 스킬의 기능이 맞는지 의미적으로 검증**합니다.

한국어 요청에서 영어 스킬을 찾는 등의 교차 언어 매칭도 가능합니다.

---

## Dependency Resolution

스킬 간 의존성을 지원합니다.

예를 들어:

```text
Skill A
└── requires Skill B
    └── requires Skill C
```

선택된 스킬의 의존성을 끝까지 검증합니다.

결과에는 다음이 포함됩니다.

* 필요한 선행 스킬
* 해결된 의존성
* 해결되지 않은 요구사항

`partial`, `unsupported`, `unknown` 상태의 후보는 단독으로 `complete` 결과를 만들 수 없습니다.

---

## Snapshot-Based Registry

검색 결과에는 다음 정보가 포함됩니다.

```text
registry_snapshot
```

이 값을 사용하면 검색 당시의 스킬과 현재 읽는 스킬이 동일한지 확인할 수 있습니다.

`get_skill_body` 호출 시 스냅샷을 지정할 수 있으며 문서 해시도 함께 반환합니다.

---

## Incremental Embedding

변경되지 않은 스킬 문서는 다시 임베딩하지 않습니다.

* 변경되지 않은 문서 → 기존 벡터 재사용
* 변경된 문서 → 새로운 인덱스 생성
* 인덱싱 실패 → 이전 정상 스냅샷 유지
* 임베딩 배치 크기 → 기본 `32`

서버를 재시작해도 캐시를 유지하도록 설정할 수도 있습니다.

---

# Architecture

```text
User Request
     │
     ▼
┌─────────────────┐
│ Lexical Search  │
└─────────────────┘
     │
     ├──────────────┐
     │              │
     ▼              ▼
┌─────────────────┐
│ Semantic Search │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│       RRF       │
│ Rank Fusion     │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Candidate Skills│
└─────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Semantic Verification   │
│        Optional         │
└─────────────────────────┘
     │
     ▼
┌─────────────────┐
│ Dependency Check│
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Evidence Result │
└─────────────────┘
```

---

# Requirements

* Python 3.11+
* Windows 또는 Linux

선택 사항:

* OpenRouter API Key
* 실제 임베딩 모델
* Semantic Verification

---

# Installation

## Windows

```powershell
cd path\to\skill-injection-mcp

py -3.12 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -U pip

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest -q
```

---

## Linux

```bash
cd path/to/skill-injection-mcp

python3.12 -m venv .venv

source .venv/bin/activate

pip install -U pip

pip install -e ".[dev]"

pytest -q
```

---

# Environment Configuration

`.env.example` 파일을 `.env`로 복사합니다.

```env
OPENROUTER_API_KEY=your_api_key
```

`.env` 파일은 Git에 커밋하지 않습니다.

운영 기본값은 실제 `OpenRouterEmbedder`입니다. API Key가 없으면 오류를 반환합니다. 테스트에서만 `SKILL_INJECT_USE_FAKE_EMBEDDER=true`를 명시합니다.

공식 네이티브 라이브러리는 `python scripts/setup_sqlite_vector.py --output-dir .native`로 설치합니다. Windows에서는 `SKILL_INJECT_SQLITE_VECTOR_PATH`에 `vector.dll`의 절대 경로를 지정합니다. Python 휠이나 소스 빌드는 필요하지 않습니다. `SKILL_INJECT_OPENROUTER_API_KEY_FILE`에 승인된 `.env` 경로를 지정하면 다른 MCP와 같은 키 파일을 사용할 수 있습니다.

---

# Running the Server

먼저 스킬 폴더 위치를 지정합니다.

## Windows

```powershell
$env:SKILL_INJECT_SKILLS_DIR = "C:\path\to\skills"

skill-inject-mcp
```

또는:

```powershell
python -m skill_inject_mcp
```

---

## Linux

```bash
export SKILL_INJECT_SKILLS_DIR=/path/to/skills

skill-inject-mcp
```

또는:

```bash
python -m skill_inject_mcp
```

---

# Usage

메인 MCP 도구는 다음과 같습니다.

```text
resolve_skills
```

요청 예시:

```json
{
  "request": {
    "schema_version": "1.0",
    "requirements": [
      {
        "id": "sparse",
        "description": "BM25 FTS5 sparse full-text search in Python",
        "required": true
      }
    ],
    "constraints": {
      "top_k": 3
    }
  }
}
```

---

## Requirement Options

각 요구사항에는 추가 정보를 포함할 수 있습니다.

```text
search_query
depends_on
```

또한 선택적으로 `draft_plan`을 제공할 수 있습니다.

```text
draft_plan
├── id
├── summary
└── requirement_ids
```

지원하지 않는 필드가 포함되면 조용히 무시하지 않고 오류를 반환합니다.

---

# Response

응답에는 다음 정보가 포함됩니다.

```text
match_status
checks
evidence
gaps
validation_errors
plan_bindings
retriever_degraded
registry_snapshot
verification_mode
verification_degraded
```

각 검증 결과에는 다음 정보가 포함됩니다.

* 검증기 종류
* 판정 결과
* 후보 스킬 ID
* 충족하지 못한 요구사항
* 근거
* 인용 위치

---

# Semantic Verification

의미 기반 검증을 활성화하려면 명시적으로 설정해야 합니다.

```env
SKILL_INJECT_VERIFICATION_MODE=semantic
SKILL_INJECT_VERIFICATION_MODEL=openai/gpt-oss-120b
```

API Key가 존재한다고 자동으로 Semantic Verification이 활성화되지는 않습니다.

Semantic Verification이 실패하면 결과는 `unknown`으로 처리됩니다.

---

## Verification Configuration

```env
SKILL_INJECT_VERIFICATION_TOP_K=5
SKILL_INJECT_VERIFICATION_MAX_SOURCE_CHARS=16000
SKILL_INJECT_VERIFICATION_MAX_TOKENS=8192
SKILL_INJECT_VERIFICATION_MAX_RETRIES=1
```

### Verification Candidates

기본적으로 상위 5개 후보를 검증합니다.

### Large Sources

기본적으로 16,000자를 초과하는 후보는 잘라서 보내지 않습니다.

해당 후보는 `unknown`으로 처리됩니다.

### Retry

응답이 토큰 부족으로 잘린 경우에만 동일한 요청을 한 번 재시도합니다.

기본 출력 토큰 예산:

```text
1차: 8192
2차: 16384
```

다른 오류는 자동 재시도하지 않습니다.

---

# Embedding

기본 임베딩 모델:

```text
qwen/qwen3-embedding-8b
```

특징:

* 1024차원 벡터
* OpenRouter 사용
* 배치 임베딩 지원

개발 환경에서는 항상 동일한 결과를 생성하는 `FakeEmbedder`를 사용할 수 있습니다.

---

# Offline Mode

인터넷 없이 사용할 수 있는 Lexical Search 모드를 제공합니다.

이 경우 결과에는 다음 상태가 표시됩니다.

```text
verification_mode: lexical
```

오프라인 모드는 보수적으로 동작하며 교차 언어 의미 매칭을 확정하지 않습니다.

---

# Project Structure

```text
src/skill_inject_mcp/
├── server
├── schemas
├── config
├── registry
├── embedding
├── indexing
└── retrieval

fixtures/skills/
└── 테스트용 샘플 스킬

tests/
└── pytest 테스트

AGENTS.md
└── AI 에이전트용 사용법 및 스키마 문서
```

---

# Codex Integration

이 프로젝트는 Codex 환경과 연동할 수 있습니다.

Codex의 전역 Skills 목록을 가져와 다음 정보를 가진 매니페스트를 생성합니다.

```text
Skill Name
Source SKILL.md Path
```

환경 변수로 해당 매니페스트를 사용할 수 있습니다.

```env
SKILL_INJECT_SKILL_MANIFEST=/path/to/manifest.json
```

`constraints.skills_dir`을 직접 지정하면 해당 설정이 우선합니다.

---

# Recommended Codex Configuration

스킬이 많은 환경에서는 다음 설정을 권장합니다.

```env
SKILL_INJECT_VERIFICATION_MODE=semantic

SKILL_INJECT_VERIFICATION_TOP_K=5

SKILL_INJECT_VERIFICATION_MAX_SOURCE_CHARS=100000

SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE=true

SKILL_INJECT_EMBEDDING_BATCH_SIZE=32
```

---

# Prompt Hook

검색·검증은 필수 의존성인 `neograph-engine==0.12.1`의 네이티브 DAG에서 실행합니다.
`resolve_skills`의 `execution`에는 요구사항별 `retrieve → verify → bind` 실행 기록이 들어갑니다.
훅은 `discover → validate_output`을 실행하고, 취소는 실제 비동기 HTTP 요청까지 전달합니다.
NeoGraph 설치·로드 실패는 명시적인 오류이며 별도 실행기로 대체하지 않습니다.

선택적으로 `codex_prompt_hook`을 사용할 수 있습니다.

사용자가 프롬프트를 입력하면 관련 스킬 후보를 최대 3개까지 참고 정보로 제공합니다.

특징:

* 프롬프트 실행을 차단하지 않음
* 단순한 메시지는 건너뜀
* 기본 내부 타임아웃 120초, 상위 Codex 훅 타임아웃 권장값 240초
* 기존 인덱스 스냅샷 사용
* 백그라운드 인덱싱 지원

Hook은 참고용 후보를 제공하는 역할이며 실제 요구사항 검증과 바인딩은 `resolve_skills`가 담당합니다.

---

## Codex Hook Example

```json
{
  "type": "mcp_tool",
  "server": "skill-injection",
  "tool": "codex_prompt_hook",
  "input": {
    "prompt": "${prompt}"
  },
  "timeout": 240,
  "statusMessage": "Finding installed skill candidates"
}
```

---

# Timeout Configuration

기본 HTTP 읽기 타임아웃:

| Setting                               | Default |
| ------------------------------------- | ------: |
| `SKILL_INJECT_HOOK_TIMEOUT_S`         |    120s |
| `SKILL_INJECT_EMBEDDING_TIMEOUT_S`    |    180s |
| `SKILL_INJECT_MULTI_QUERY_TIMEOUT_S`  |     30s |
| `SKILL_INJECT_VERIFICATION_TIMEOUT_S` |    120s |

Codex 등록 권장값:

| Setting                 | Value |
| ----------------------- | ----: |
| `startup_timeout_sec`   |    60 |
| `tool_timeout_sec`      |   900 |
| `UserPromptSubmit Hook` |  240s |

내부 훅 제한 120초는 전체 비동기 대기 예산이고, HTTP 읽기 제한과 별도로 적용합니다.
응답이 도착하면 즉시 반환합니다. 상위 훅 제한은 내부 제한보다 길게 설정하세요.

큰 작업은 여러 요구사항과 API 호출이 발생할 수 있으므로 하나의 큰 요청으로 처리하기보다 작은 작업 단위로 나누는 것을 권장합니다.

환경 설정을 변경한 후에는 MCP 프로세스를 재시작해야 합니다.

---

# Notes

* `sqliteai/sqlite-vector`가 기본 백엔드이며 네이티브 `vector_full_scan`으로 코사인 유사도를 계산합니다.
* DLL/SO 로드 실패는 오류입니다. NumPy 저장은 `SKILL_INJECT_DENSE_BACKEND=numpy`를 명시한 오프라인 모드에서만 사용합니다.
* BM25 점수는 확률이 아닌 문서 간 상대적인 순위 지표입니다.
* Semantic Search만으로 `supported` 판정을 만들기 위해서는 별도의 점수 기준을 충족해야 합니다.
* RRF의 최대 점수보다 `constraints.min_score`를 높게 설정하면 모든 후보가 제거될 수 있습니다.
* 인덱스 스냅샷은 기본적으로 프로세스 내부에서 유지됩니다.
* Persistent Embedding Cache를 활성화하면 서버 재시작 이후에도 변경되지 않은 문서의 임베딩을 재사용할 수 있습니다.
* `pytest`는 실제 HTTP 요청을 차단합니다.
* 실제 모델 기반 평가는 별도의 Opt-in Live Evaluation으로 실행합니다.

---

# Security & Privacy

Semantic Search 및 Semantic Verification을 활성화하면 일부 데이터가 외부 제공자에게 전송될 수 있습니다.

전송될 수 있는 데이터:

* 스킬 문서
* 사용자 요구사항
* 후보 스킬의 설명과 본문

따라서 사용자가 외부 전송을 허용한 스킬 카탈로그와 데이터에 대해서만 활성화하는 것을 권장합니다.

---

# Summary

**Skill Injection MCP**는 AI 에이전트가 작업을 수행하기 전에 현재 요청에 적합한 스킬을 자동으로 찾고 검증할 수 있도록 설계된 MCP 서버입니다.

```text
Request
   ↓
Search Skills
   ↓
Hybrid Retrieval
   ↓
Semantic Verification
   ↓
Dependency Resolution
   ↓
Evidence-Based Result
```

단순한 스킬 추천을 넘어 **요청과 스킬의 실제 지원 범위를 검증하고, 근거와 의존성까지 함께 반환하는 것**을 목표로 합니다.
