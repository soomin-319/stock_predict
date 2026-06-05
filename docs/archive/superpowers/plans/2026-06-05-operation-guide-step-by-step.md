# Operation Guide Step-by-Step Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `docs/PROJECT_OPERATION_GUIDE.md` so readers understand the program in actual execution order and can follow chatbot scenarios including symbol lookup, result checking, refresh, and recommendation.

**Architecture:** Keep one operation guide, but change its information order. Put operator actions and program execution stages first, then chatbot user scenarios, troubleshooting, and technical reference.

**Tech Stack:** Markdown, Mermaid, PowerShell command examples, Python source inspection

---

### Task 1: Rewrite the guide around program execution order

**Files:**
- Modify: `docs/PROJECT_OPERATION_GUIDE.md`

- [ ] **Step 1: Replace the current structure**

Rewrite the guide with these top-level sections:

1. 문서 목적과 핵심 원칙
2. 전체 실행 흐름
3. 실행 전 준비
4. 파이프라인 실행 방법
5. 프로그램 내부 실행 단계
6. 결과 파일 확인
7. 챗봇 실행 흐름
8. 사용자 시나리오
9. 문제 상황별 대응
10. 기술 상세
11. 테스트와 운영 점검표

- [ ] **Step 2: Describe each pipeline stage**

For each actual `run_pipeline()` stage, explain:

- 운영자가 보는 진행 메시지
- 프로그램이 하는 일
- 주요 입력과 출력
- 담당 모듈
- 실패하거나 데이터가 부족할 때의 동작

- [ ] **Step 3: Add chatbot user scenarios**

Add scenario tables for:

- 종목코드 입력
- `결과`
- `최신화`
- `추천`
- `도움말`

State explicitly that `요약` is not an independent command. Explain that news/disclosure summaries are generated during symbol lookup when missing.

- [ ] **Step 4: Preserve policy guardrails**

State explicitly:

- Buy/sell/hold uses only `predicted_return`.
- News, disclosures, summaries, and impact scores are display-only context.
- Chatbot `추천` is a separate real-time recommendation flow and must not be confused with the main pipeline recommendation.

### Task 2: Verify document accuracy and readability

**Files:**
- Verify: `docs/PROJECT_OPERATION_GUIDE.md`
- Reference: `src/pipeline.py`
- Reference: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Check Markdown quality**

Run:

```powershell
git diff --check -- docs/PROJECT_OPERATION_GUIDE.md
```

Expected: no output and exit code `0`.

- [ ] **Step 2: Check required scenario terms**

Run:

```powershell
Select-String -Path docs/PROJECT_OPERATION_GUIDE.md -Pattern '종목코드 입력','결과','최신화','추천','요약은 독립 명령'
```

Expected: every required scenario appears.

- [ ] **Step 3: Check policy wording**

Run:

```powershell
Select-String -Path docs/PROJECT_OPERATION_GUIDE.md -Pattern 'predicted_return','표시용','별도'
```

Expected: the main recommendation policy, display-only context rule, and separate recommendation flow are documented.

- [ ] **Step 4: Review final diff**

Run:

```powershell
git diff -- docs/PROJECT_OPERATION_GUIDE.md
```

Expected: the guide is execution-order-first, with no unrelated file changes.

- [ ] **Step 5: Commit**

```powershell
git add docs/PROJECT_OPERATION_GUIDE.md
git commit -m "Rewrite operation guide around execution flow"
```
