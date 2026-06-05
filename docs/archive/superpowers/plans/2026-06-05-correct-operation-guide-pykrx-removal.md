# Correct Operation Guide pykrx Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PROJECT_OPERATION_GUIDE.md` accurately state that pykrx is not used and investor-flow external collection is disabled.

**Architecture:** Change only the operation guide. Preserve the documented pipeline sequence while distinguishing disabled investor-flow collection from active DART, Naver News, and yfinance integrations.

**Tech Stack:** Markdown, PowerShell, Git

---

### Task 1: Correct external integration and execution-stage descriptions

**Files:**
- Modify: `docs/PROJECT_OPERATION_GUIDE.md`

- [ ] **Step 1: Update the optional external integration section**

State that live OHLCV uses yfinance, pykrx is neither used nor required, and investor-flow external collection is disabled.

- [ ] **Step 2: Update the context collection execution stage**

Remove the claim that investor-flow data is collected according to configuration. State that the stage keeps investor-flow context empty while collecting configured DART disclosures and Naver News.

- [ ] **Step 3: Add an operational troubleshooting note**

Add a row explaining that pykrx installation is unnecessary and that its absence is not an error.

- [ ] **Step 4: Verify the guide**

Run:

```powershell
git grep -n -i pykrx -- docs/PROJECT_OPERATION_GUIDE.md
git diff --check -- docs/PROJECT_OPERATION_GUIDE.md
```

Expected: pykrx appears only in explicit “not used/not required” guidance; diff check exits successfully.

- [ ] **Step 5: Commit**

```powershell
git add docs/PROJECT_OPERATION_GUIDE.md docs/superpowers/plans/2026-06-05-correct-operation-guide-pykrx-removal.md
git commit -m "Clarify pykrx removal in operation guide"
```
