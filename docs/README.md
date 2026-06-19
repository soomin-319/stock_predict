# 문서 안내

`docs/` 루트에는 현재 기준 정규 문서를 둔다.
과거 분석·제안·리뷰, 이전 가이드, 계획/설계 기록은 `docs/archive/`에 보관한다.

## 정규 문서 (기능별 세트)

기능 영역별 참조 문서다. 진입점은 [`INDEX.md`](INDEX.md)이며 모듈 → 문서 매핑이 함께 정리되어 있다.

- [`INDEX.md`](INDEX.md): 기능별 문서 인덱스 및 모듈 매핑
- [`01_pipeline.md`](01_pipeline.md): 전체 파이프라인 흐름 및 진입점
- [`02_data.md`](02_data.md): 데이터 로드·정제·유니버스·실데이터·투자자 컨텍스트
- [`03_features.md`](03_features.md): 피처 생성 및 선택
- [`04_model.md`](04_model.md): 모델 학습 (LightGBM 멀티헤드)
- [`05_validation.md`](05_validation.md): Walk-Forward 검증 및 백테스트
- [`06_signal_policy.md`](06_signal_policy.md): 시그널 정책 및 매수/매도/관망 추천
- [`07_reports.md`](07_reports.md): 리포트 및 산출물
- [`08_news_impact.md`](08_news_impact.md): 뉴스 임팩트 모듈
- [`09_chatbot.md`](09_chatbot.md): 카카오 챗봇 및 Colab 통합
- [`10_config.md`](10_config.md): 설정 및 환경변수

## 운영/참고 문서

- [`ROADMAP.md`](ROADMAP.md): 우선순위와 향후 작업
- `NEWS_IMPACT_LLM_PROMPT.md`: 뉴스 임팩트 LLM 시스템 프롬프트. **코드 자산**으로 `src/news_impact/impact_judge.py`가 런타임에 직접 로드하므로 경로를 옮기지 않는다.
- `TIMA_BENCHMARK_UPGRADE.md`, `TIMA_PREDICTION_FEATURE_CANDIDATES.md`: TIMA 벤치마크 관련 문서

## 보관 문서 (`archive/`)

- `archive/analysis-reports/`: 날짜별 코드베이스 분석, 개선 제안, 리뷰 리포트
- `archive/legacy-guides/`: numbered 정규 세트로 대체된 이전 레퍼런스 가이드 (Architecture, Operations, Features Guide, Prediction Formulas 등)
- `archive/superpowers/plans/`, `archive/superpowers/specs/`: 과거 구현 계획·설계
