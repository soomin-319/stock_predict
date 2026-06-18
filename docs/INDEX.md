# 기능별 문서 인덱스

이 폴더는 `stock_predict` 코드베이스를 기능 영역별로 분리해 정리한 참조 문서 모음이다.
모든 출력은 리서치·운영 보조용이며 투자 자문이나 자동매매 시스템이 아니다. 매수/매도/관망 결정은
익일 기대수익률(`predicted_return`)만 사용하고, 뉴스·공시·뉴스임팩트는 표시·검토용 컨텍스트로
기대수익률·순위·추천·신호를 바꾸지 않는다.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [01_pipeline.md](01_pipeline.md) | 전체 파이프라인 흐름 및 진입점 |
| [02_data.md](02_data.md) | 데이터 로드·정제·유니버스·실데이터 수집·투자자 컨텍스트 |
| [03_features.md](03_features.md) | 가격/기술적/외부/투자자 이벤트 피처 및 피처 선택 |
| [04_model.md](04_model.md) | 모델 학습 — LightGBM 멀티헤드 |
| [05_validation.md](05_validation.md) | Walk-Forward 검증 및 백테스트 |
| [06_signal_policy.md](06_signal_policy.md) | 시그널 정책 및 매수/매도/관망 추천 |
| [07_reports.md](07_reports.md) | 리포트 및 산출물 (CSV, JSON, PM 리포트) |
| [08_news_impact.md](08_news_impact.md) | 뉴스 임팩트 모듈 (수집, LLM 스코어링, 백테스트) |
| [09_chatbot.md](09_chatbot.md) | 카카오 챗봇 및 Colab 통합 |
| [10_config.md](10_config.md) | 설정(`AppConfig`) 및 환경변수 |

## 모듈 → 기능 매핑

```
src/
├── pipeline.py              → 01_pipeline  (메인 진입점 run_pipeline)
├── pipeline_support.py      → 01_pipeline  (예측 프레임 조립 지원)
├── config/settings.py       → 10_config
├── data/
│   ├── loaders.py           → 02_data
│   ├── cleaners.py          → 02_data
│   ├── universe.py          → 02_data
│   ├── krx_universe.py      → 02_data
│   ├── fetch_real_data.py   → 02_data
│   ├── cli_refresh.py       → 02_data
│   └── investor_context.py  → 02_data
├── features/
│   ├── price_features.py        → 03_features
│   ├── technical_indicators.py  → 03_features
│   ├── external_features.py     → 03_features
│   ├── regime_features.py       → 03_features
│   ├── investment_signals.py    → 03_features
│   └── feature_selection.py     → 03_features
├── models/lgbm_heads.py     → 04_model
├── validation/              → 05_validation
│   ├── walk_forward.py / backtest.py / baselines.py
│   ├── metrics.py / support.py / signal_tuning.py / result_validity.py
├── inference/predict.py     → 06_signal_policy
├── domain/signal_policy.py  → 06_signal_policy
├── recommendation/          → 06_signal_policy
│   ├── close_betting.py / realtime_close_betting.py
├── reports/                 → 07_reports
│   ├── output.py / result_formatter.py / pm_report.py
│   ├── run_artifacts.py / report_metadata.py
│   ├── issue_summary.py / news_impact_context.py / context_policy.py
├── news_impact/             → 08_news_impact
├── chatbot/                 → 09_chatbot
│   ├── kakao_colab_bot.py / intent.py / responses.py
└── utils/                   → 07_reports
    ├── atomic_files.py / result_cleanup.py / secrets.py
```

## 콘솔 진입점

| 명령어 | 진입점 | 설명 |
|--------|--------|------|
| `stock-predict` | `src.pipeline:main` | 메인 예측 파이프라인 |
| `stock-predict-kakao` | `src.chatbot.kakao_colab_bot:main` | 카카오 챗봇 서버 |
| `stock-news-impact` | `src.news_impact.run:main` | 뉴스 임팩트 독립 실행 |

## 남은 개선 과제 요약

대부분의 초기 분석 항목(재현성 오염, 단계 실패 격리, 시그널 튜닝 과적합, 백테스트
로그/단순수익률 혼용, walk-forward 폴드 직렬화 비용 등)은 코드에 이미 반영되어 있다.
아래는 각 문서의 "개선 및 수정 제안" 섹션에 남아 있는 실제 미해결 항목이다.

| 우선순위 | 영역 | 핵심 이슈 | 문서 |
|----------|------|-----------|------|
| P1 | 데이터 | `_fetch_flow`가 항상 실패하는 스텁 — 투자자 수급(외국인/기관 순매수)이 실시간으로 수집되지 않음 | [02](02_data.md) |
| P2 | 데이터 | `resolve_fetch_symbols` 기본 로더가 `set` 반환(수집 순서 비결정적); 정렬된 리스트 변형 사용 권장 | [02](02_data.md) |
| P2 | 리포트 | KRX 영업일/공휴일 표가 2025–2026만 하드코딩 | [07](07_reports.md) |
| P2 | 뉴스임팩트 | LLM 응답 캐시 디렉터리 만료/버전 무효화 정책 부재 | [08](08_news_impact.md) |
| P2 | 챗봇 | 공식 Kakao 요청 서명 스펙 확인 후 서명 검증 추가 | [09](09_chatbot.md) |

> 각 항목의 상세 근거는 해당 문서의 "개선 및 수정 제안" 섹션 참조.
