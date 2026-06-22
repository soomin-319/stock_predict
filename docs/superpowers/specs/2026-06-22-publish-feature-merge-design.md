# 로컬 main → GitHub main 통합 (publish/Colab 기준선 서빙) 설계

- 날짜: 2026-06-22
- 브랜치: `merge/local-main-into-main` (base: `origin/main` @ 26c710a)
- 목표: 로컬 `main`에만 있던 미머지 작업(특히 **publish → Colab 기준선 서빙** 기능)을 GitHub `main`에 안전하게 통합한다.

## 배경

로컬 `main`은 `origin/main` 대비 ahead 26 / behind 20 으로 갈라져 있었다. 로컬에만 있는 작업은 크게 두 묶음:

- **그룹 B — publish → Colab 기준선 서빙** (2026-06-17): `src/ops/publish_predictions.py`, `src/ops/published_store.py`, 챗봇의 published-baseline 연동, 관련 테스트/문서. **`origin/main`에 전혀 없음.**
- **그룹 A — news-impact gemma 통합** (2026-06-16): `origin/main`이 이미 #284/#296/#305 등으로 다른/최신 형태로 반영함. 일부 파일(`news_impact_fixture.py`, `NEWS_IMPACT_LLM_PROMPT.md` 등)만 main에 빠져 있음.

`origin/main`은 그동안 20개 PR이 머지되어 앞서 있으므로 **덮어쓰기(force-push)는 금지**(머지된 PR 소실). 합치기(merge) + PR 검토 방식으로 통합한다.

## 전략

1. `origin/main`에서 `merge/local-main-into-main` 브랜치 생성. (기존 20개 PR 보존)
2. 로컬 `main`을 merge.
3. 충돌/내용 처리(아래) 후 `pytest` 전체 통과 확인.
4. 브랜치 push → `main`으로 PR 생성. 머지는 검토 후 사람이 수행.

## 충돌·내용 처리 정책

merge 시뮬레이션(merge-tree) 결과 충돌 파일은 10개.

| 대상 | 처리 |
|---|---|
| `src/ops/*`, publish 테스트/문서 (신규, 충돌 없음) | 그대로 추가 |
| `docs/01_pipeline.md` ~ `docs/10_config.md`, `docs/INDEX.md` (로컬 전용 실험 구조) | **전부 제거** — `origin/main`에 없는 로컬 실험본이며 올리지 않는다 |
| `src/chatbot/kakao_colab_bot.py` (충돌) | **수작업 병합** — main 최신 + publish 기준선 연동 + gemma llm-config 와이어링이 **모두 동작**하도록 |
| `src/pipeline.py` (충돌) | **수작업 병합** — 동일 원칙 |
| 그 외 로컬 변경 중 main에 빠진 것(gemma 일부 문서/설정 등) | 자동 merge되면 포함 |

## 검증

- `pytest` 전체 실행. 특히 `tests/test_publish_predictions.py`, `tests/test_published_store.py`, `tests/test_kakao_colab_bot.py`, news_impact 관련 테스트 통과 확인.
- 테스트 실패 시 머지 결과를 수정(또는 해당 변경 보류)하고 재실행.

## 산출물

- `main`으로 가는 PR 1개. (`gh` 미설치 → push 후 PR 생성 링크 제공)
- 머지 결정은 사람이 검토 후 수행.

## 범위 밖 (YAGNI)

- 그룹 A(gemma)를 로컬 버전으로 main에 되돌리는 작업(이미 main이 최신). 충돌 시 main 우선.
- `docs/01~10`, `INDEX` 로컬 문서 구조의 main 반영.
