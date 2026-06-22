# 문서 안내

`docs/` 루트에는 현재 기준 종합 레퍼런스 문서를 둔다.
과거 분석·제안·리뷰, 이전 가이드, 계획/설계 기록은 `docs/archive/`와 `docs/superpowers/`에 보관한다.

## 현재 문서

- [`TIMA_BENCHMARK_UPGRADE.md`](TIMA_BENCHMARK_UPGRADE.md), [`TIMA_PREDICTION_FEATURE_CANDIDATES.md`](TIMA_PREDICTION_FEATURE_CANDIDATES.md): TIMA 벤치마크 관련 문서

> 뉴스 임팩트 LLM 시스템 프롬프트는 **코드 자산**이므로 `docs/`에 두지 않고
> `src/news_impact/prompts/news_impact_llm_prompt.md`에 패키지 리소스로 둔다.
> `src/news_impact/impact_judge.py`가 런타임에 직접 로드한다.

## 보관 문서

- `archive/analysis-reports/`: 날짜별 코드베이스 분석, 개선 제안, 리뷰 리포트
- `archive/legacy-guides/`: 이전 레퍼런스 가이드 (Architecture, Operations, Features Guide, Prediction Formulas 등)
- `archive/superpowers/`, `superpowers/`: 구현 계획(`plans/`)·설계(`specs/`) 기록
