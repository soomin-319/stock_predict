# 기능별 문서 인덱스

이 폴더는 프로젝트를 기능 영역별로 분리하여 정리한 참조 문서 모음입니다.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [01_pipeline.md](01_pipeline.md) | 전체 파이프라인 흐름 및 진입점 |
| [02_data.md](02_data.md) | 데이터 레이어 (로딩, 정제, 갱신, 유니버스) |
| [03_features.md](03_features.md) | 피처 엔지니어링 (가격, 기술지표, 외부시장, 투자자 흐름) |
| [04_model.md](04_model.md) | 모델 학습 — LightGBM 멀티헤드 |
| [05_validation.md](05_validation.md) | Walk-Forward 검증 및 백테스트 |
| [06_signal_policy.md](06_signal_policy.md) | 시그널 정책 및 매수/매도/관망 추천 |
| [07_reports.md](07_reports.md) | 리포트 및 산출물 (CSV, JSON, PM 리포트) |
| [08_news_impact.md](08_news_impact.md) | 뉴스 임팩트 모듈 (수집, LLM 스코어링, 백테스트) |
| [09_chatbot.md](09_chatbot.md) | 카카오 챗봇 및 Colab 통합 |
| [10_config.md](10_config.md) | 설정 및 환경변수 |

## 빠른 참조: 모듈 → 기능 매핑

```
src/
├── pipeline.py              → 01_pipeline  (메인 진입점)
├── pipeline_support.py      → 01_pipeline  (파이프라인 지원 함수)
├── config/
│   └── settings.py          → 10_config
├── data/
│   ├── loaders.py           → 02_data
│   ├── cleaners.py          → 02_data
│   ├── fetch_real_data.py   → 02_data
│   ├── cli_refresh.py       → 02_data
│   ├── universe.py          → 02_data
│   ├── krx_universe.py      → 02_data
│   └── investor_context.py  → 02_data
├── features/
│   ├── price_features.py    → 03_features
│   ├── technical_indicators.py → 03_features
│   ├── external_features.py → 03_features
│   ├── regime_features.py   → 03_features
│   ├── investment_signals.py → 03_features
│   └── feature_selection.py → 03_features
├── models/
│   └── lgbm_heads.py        → 04_model
├── validation/
│   ├── walk_forward.py      → 05_validation
│   ├── backtest.py          → 05_validation
│   ├── baselines.py         → 05_validation
│   ├── metrics.py           → 05_validation
│   ├── support.py           → 05_validation
│   ├── signal_tuning.py     → 05_validation
│   └── result_validity.py   → 05_validation
├── inference/
│   └── predict.py           → 06_signal_policy
├── domain/
│   └── signal_policy.py     → 06_signal_policy
├── recommendation/
│   ├── close_betting.py     → 06_signal_policy
│   └── realtime_close_betting.py → 06_signal_policy
├── reports/
│   ├── output.py            → 07_reports
│   ├── result_formatter.py  → 07_reports
│   ├── pm_report.py         → 07_reports
│   ├── run_artifacts.py     → 07_reports
│   ├── report_metadata.py   → 07_reports
│   ├── issue_summary.py     → 07_reports
│   ├── news_impact_context.py → 07_reports
│   ├── context_policy.py    → 07_reports
│   └── run_artifacts.py     → 07_reports
├── news_impact/             → 08_news_impact
├── chatbot/                 → 09_chatbot
└── utils/
    ├── atomic_files.py      → 07_reports
    ├── secrets.py           → 10_config
    └── result_cleanup.py    → 07_reports
```

## 개선 및 수정 제안 (교차 요약)

각 문서 하단에 기능별 **개선 및 수정 제안** 섹션이 있다. 초기 분석에서 도출한 **P0(정확성/버그/누수)** 항목은 대부분 코드 또는 문서 정정으로 해결되어 제거했다. 아래는 아직 남아 있는 우선 과제 요약이다.

### 남은 우선 과제

| 우선순위 | 영역 | 핵심 이슈 | 위치 | 문서 |
|----------|------|-----------|------|------|
| P0 | 챗봇 | `/kakao/webhook` 인증 부재(자원 고갈 위험) | `chatbot/kakao_colab_bot.py:1889` | [09](09_chatbot.md) |
| P1 | 파이프라인 | 튜닝값을 공유 `AppConfig`에 in-place 변형(재현성 오염) | `pipeline.py:591-594` | [01](01_pipeline.md) |
| P1 | 검증 | 시그널 가중치 튜닝의 과적합/탐색 빈약 | `signal_tuning.py:11-12` | [05](05_validation.md) |
| P1 | 챗봇 | 잡 제출 레이트리밋/동시성 상한 부재 | `chatbot/kakao_colab_bot.py` | [09](09_chatbot.md) |

> 각 항목의 상세 근거와 제안은 해당 문서의 "개선 및 수정 제안" 섹션 참조.
