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

각 문서 하단에 기능별 **개선 및 수정 제안** 섹션을 추가했다. 아래는 우선순위 **P0(정확성/버그/누수)** 항목만 모은 교차 요약이다.

| 영역 | 핵심 이슈 | 위치 | 문서 |
|------|-----------|------|------|
| 데이터 | KOSDAQ 종목이 `.KS`로 강제 변환되어 수집 실패 | `data/fetch_real_data.py:43` | [02](02_data.md) |
| 데이터 | `load_ohlcv_csv` BOM 미처리(문서는 자동감지 주장) | `data/loaders.py:12` | [02](02_data.md) |
| 데이터 | 미조정(unadjusted) 가격 → 분할/증자 시 가짜 급등락 | `fetch_real_data.py:94` | [02](02_data.md) |
| 피처 | 미국 지수 "당일" 조인으로 미래정보 누수 | `features/external_features.py:185` | [03](03_features.md) |
| 피처 | 외부 피처 `bfill()`이 미래값을 과거로 역채움 | `external_features.py:184` | [03](03_features.md) |
| 피처 | 문서 오류: `vol_ratio_20`은 변동성이 아닌 거래량 비율 | `price_features.py:192` | [03](03_features.md) |
| 모델 | 분위수 교차 미보정 → `uncertainty_width` 음수 가능 | `models/lgbm_heads.py:194` | [04](04_model.md) |
| 검증 | 문서가 약속한 `min_signal_score` 필터 미적용 | `validation/backtest.py:65` | [05](05_validation.md) |
| 검증 | 선택 기준이 문서(signal_score)와 달리 predicted_return | `backtest.py:99` | [05](05_validation.md) |
| 검증 | `result_validity` 검사가 문서 주장보다 약함 | `validation/result_validity.py` | [05](05_validation.md) |
| 시그널 | 최신 예측 경로에서 이벤트 부스트 **이중 적용** | `pipeline.py:707,719` | [06](06_signal_policy.md) |
| 뉴스 | LLM 판정 키 문서가 실제 스키마와 불일치 | `news_impact/impact_judge.py:12` | [08](08_news_impact.md) |
| 챗봇 | `/webhook` 인증 부재(자원 고갈 위험) | `chatbot/kakao_colab_bot.py:1878` | [09](09_chatbot.md) |
| 설정 | 문서의 `load_app_config` 시그니처 불일치 | `config/settings.py:161` | [10](10_config.md) |

> 각 항목의 상세 근거와 제안은 해당 문서의 "개선 및 수정 제안" 섹션 참조.
