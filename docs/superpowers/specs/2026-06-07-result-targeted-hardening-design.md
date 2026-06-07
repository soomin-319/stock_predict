# Result Targeted Hardening Design

## Goal

기존 result hardening 구조를 유지하면서 검증에서 발견된 안전·정합성 공백만 보완한다.

## Scope

- 챗봇은 `production`/`real`이며 `status=pass|warning`, `promoted=true`인 latest manifest만 신뢰한다.
- latest가 없을 때 legacy CSV fallback은 CSV 자체가 `production`/`real` 메타데이터를 가진 경우만 허용한다.
- Colab runner는 오래된 최상위 호환 경로 대신 현재 `run_pipeline()` 실행의 manifest artifact 경로를 반환한다.
- `RunArtifactManager`는 run 디렉터리 밖으로 탈출하는 상대·절대 경로를 거부한다.
- latest 승격 상태는 `pass|warning`만 허용한다.
- cleanup은 문서대로 `result/test/` 하위 산출물도 정리하되 `latest/`, runtime 상태 JSON, 허용 루트 밖은 보호한다.
- 진행 문서의 모순과 오래된 테스트 수를 갱신한다.

## Compatibility

검증된 latest 결과가 우선이다. 기존 최상위 CSV는 운영 메타데이터가 포함된 경우에만 fallback으로 사용한다. 메타데이터 없는 오래된 결과를 운영 추천에 사용하는 것보다 안전 차단을 우선한다.

## Error Handling

- 잘못된 manifest 또는 안전하지 않은 legacy 결과는 없는 결과처럼 처리한다.
- Colab artifact가 manifest에 없으면 빈 경로를 반환한다.
- 안전하지 않은 artifact 경로는 `ValueError`로 즉시 거부한다.
- cleanup은 지정된 허용 하위 루트만 삭제한다.

## Testing

- 챗봇: fail/unpromoted latest와 sample/metadata 없는 legacy 차단
- Colab: 현재 실행 manifest artifact 경로 반환
- artifacts: 경로 탈출과 미허용 상태 승격 차단
- cleanup: `result/test/` 정리 및 보호 경로 유지
- 전체 pytest와 sample smoke pipeline 검증
