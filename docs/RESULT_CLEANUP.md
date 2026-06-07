# result/ 산출물 정리 절차

`result/`는 CSV, JSON, pytest 임시 파일을 저장하는 로컬 산출물 영역이다.
Git 추적 대상이 아니며, 필요한 결과만 외부에 따로 백업한다.

## 안전 원칙

- 삭제 전 현재 경로가 저장소 루트인지 확인한다.
- 보존해야 하는 리포트는 먼저 다른 위치로 복사한다.
- `result/` 밖의 파일은 이 절차로 삭제하지 않는다.

## pytest 임시 파일만 정리

```powershell
Resolve-Path result/.pytest_tmp
Remove-Item -Recurse -Force -LiteralPath result/.pytest_tmp
```

## 전체 산출물 정리

필요한 결과를 백업한 뒤에만 실행한다.

```powershell
Resolve-Path result
Remove-Item -Recurse -Force -LiteralPath result
New-Item -ItemType Directory -Path result | Out-Null
```
