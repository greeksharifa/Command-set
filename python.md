# Python

Python 인터프리터, 가상환경, 패키지 관리, 경로와 환경 점검에 자주 쓰는 명령을 모아둔 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 현재 Python 버전 | `python --version` | `python3 --version`도 자주 사용 |
| 실행 파일 위치 | `which python` | Windows는 `where python` |
| 모듈 실행 | `python -m <module>` | `python -m pip --version` |
| 인터프리터 진입 | `python` | REPL 진입 |
| 한 줄 실행 | `python -c 'print("hi")'` | 짧은 검사에 유용 |

## 인터프리터 / one-liner

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| REPL 진입 | `python` | 대화형 확인 |
| 한 줄 출력 | `python -c 'print("hello")'` | 쉘에서 즉시 확인 |
| 모듈 실행 | `python -m pip --version` | `-m`으로 현재 인터프리터 기준 실행 |
| 스크립트 인자 확인 | `python -c 'import sys; print(sys.argv)' foo bar` | 인자 전달 테스트 |
| 경로 확인 | `python -c 'import sys; print(sys.executable)'` | 실제 사용 중인 인터프리터 확인 |
| 작업 디렉터리 확인 | `python -c 'import os; print(os.getcwd())'` | 상대경로 문제 점검 |

```bash
python -c 'import sys, os; print(sys.version); print(sys.executable); print(os.getcwd())'
```

## 가상환경

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| venv 생성 | `python -m venv .venv` | 프로젝트별 격리 |
| 활성화 | `source .venv/bin/activate` | Windows PowerShell은 `.\.venv\Scripts\Activate.ps1` |
| 비활성화 | `deactivate` | 현재 셸 종료 없이 해제 |
| 가상환경의 Python 확인 | `which python` | 활성화 후 `.venv/bin/python` 형태여야 함 |
| 가상환경 삭제 | `rm -rf .venv` | 재생성 필요 시 사용 |

```bash
python -m venv .venv
source .venv/bin/activate
python -V
```

## pip / requirements

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| pip 업그레이드 | `python -m pip install -U pip` | `pip`보다 `python -m pip` 권장 |
| 패키지 설치 | `python -m pip install <pkg>` | 현재 인터프리터 기준 설치 |
| 특정 버전 설치 | `python -m pip install 'numpy==1.26.4'` | 따옴표 사용 권장 |
| requirements 설치 | `python -m pip install -r requirements.txt` | 표준 설치 |
| requirements 저장 | `python -m pip freeze > requirements.txt` | 현재 환경 고정 |
| 설치 패키지 목록 | `python -m pip list` | 버전 포함 목록 |
| 상세 정보 | `python -m pip show <pkg>` | 설치 위치 확인에 유용 |
| 패키지 제거 | `python -m pip uninstall <pkg>` | 복수 패키지도 가능 |
| 의존성 확인 | `python -m pip check` | 깨진 의존성 탐지 |

```bash
python -m pip install -r requirements.txt
python -m pip freeze | sed -n '1,20p'
```

## 패키지 / 경로 / 환경 확인

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 사이트 패키지 경로 | `python -c 'import site; print(site.getsitepackages())'` | 시스템/가상환경 경로 확인 |
| sys.path 확인 | `python -c 'import sys; print("\n".join(sys.path))'` | import 문제 추적 |
| 모듈 위치 확인 | `python -c 'import <pkg>; print(<pkg>.__file__)'` | 예: `import torch` |
| 환경 변수 확인 | `python -c 'import os; print(os.environ.get("PYTHONPATH"))'` | 경로 주입 여부 확인 |
| 패키지 메타데이터 | `python -m pip show <pkg>` | 설치 위치, 요구사항 확인 |
| 현재 버전 정보 | `python -V && python -c 'import sys; print(sys.version_info)'` | 런타임 차이 확인 |

```bash
python -c 'import sys; print(sys.path)'
python -c 'import site; print(site.getsitepackages())'
```

## 프로젝트 작업 팁

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 상대 import 경로 추가 | `PYTHONPATH=. python ...` | 임시 확인에만 사용 |
| 스크립트를 모듈로 실행 | `python -m package.module` | 패키지 루트 기준 실행 |
| 로컬 패키지 editable 설치 | `python -m pip install -e .` | 개발 중 권장 |
| 깨끗한 재설치 | `python -m pip install --force-reinstall <pkg>` | 버전 꼬임 정리 |
| 캐시 무시 재설치 | `python -m pip install --no-cache-dir <pkg>` | 다운로드 문제 회피 |

## 자주 쓰는 짧은 확인

```bash
python -c 'import sys; print(sys.executable)'
python -c 'import os; print(os.getcwd())'
python -m pip --version
python -m pip check
```

## 참고 메모

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 절대 경로 import 임시 추가 | `import sys; sys.path.append("/home/test")` | 코드 내 임시 확인용 |
| ASCII 변환 | `python -m pip install Unidecode` | `from unidecode import unidecode` 사용 |

