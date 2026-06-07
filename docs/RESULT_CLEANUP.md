# result/ 산출물 정리 절차

`result/`는 실행 산출물과 runtime 상태를 저장하는 로컬 영역이다.
Git 추적 대상이 아니며, 필요한 결과만 외부에 따로 백업한다.

## 보존 정책

- 성공 run: 최신 10개 및 최대 30일
- 실패 run: 최대 30일
- `result/runtime/logs/`: 최대 14일
- `result/latest/`와 `result/runtime/*.json`: 자동 삭제 금지

정리는 `src.utils.result_cleanup.cleanup_result_artifacts()`를 사용한다. 삭제 대상은
`result/runs/`, `result/test/`, `result/runtime/logs/` 내부로 제한한다.

## 안전 원칙

- 삭제 전 현재 경로가 저장소 루트인지 확인한다.
- 보존해야 하는 리포트는 먼저 다른 위치로 복사한다.
- `result/` 밖의 파일은 이 절차로 삭제하지 않는다.

## pytest 임시 파일만 정리

```powershell
Resolve-Path result/.pytest_tmp
Remove-Item -Recurse -Force -LiteralPath result/.pytest_tmp
```

성공한 pytest 세션의 전용 임시 디렉터리는 자동 삭제된다. 조사 목적으로 보존하려면:

```powershell
$env:KEEP_TEST_ARTIFACTS='1'
pytest
```
## 전체 산출물 정리

필요한 결과를 백업한 뒤에만 실행한다.

```powershell
Resolve-Path result
Remove-Item -Recurse -Force -LiteralPath result
New-Item -ItemType Directory -Path result | Out-Null
```
