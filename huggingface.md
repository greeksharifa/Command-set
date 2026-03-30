# Hugging Face

Hugging Face 로그인, 모델과 데이터셋 다운로드, 캐시, git-lfs, repo 작업을 CLI 중심으로 정리한 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 로그인 | `hf auth login` | 현재 공식 CLI |
| 로그인 상태 확인 | `hf auth whoami` | 계정 확인 |
| 토큰 목록 | `hf auth list` | 저장된 토큰 확인 |
| 모델 다운로드 | `hf download <repo_id> --local-dir <dir>` | 예: `bert-base-uncased` |
| 캐시 위치 확인 | `hf env` | `HF_HOME`, `HF_HUB_CACHE` 등 출력 |
| Git LFS 확인 | `git lfs version` | 대형 파일 다운로드 준비 |

## 인증 / 계정

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| CLI 로그인 | `hf auth login` | 토큰 붙여넣기 |
| 로그인 상태 확인 | `hf auth whoami` | 현재 계정 표시 |
| 토큰 목록 | `hf auth list` | 저장된 토큰 확인 |
| 토큰 재설정 | `hf auth logout` | 세션 정리 |
| 환경변수 토큰 | `export HF_TOKEN=...` | 비대화형 실행 시 사용 |
| 토큰 확인 | `python -c 'import os; print(os.environ.get("HF_TOKEN"))'` | 자동화 스크립트 점검 |

```bash
hf auth login
hf auth whoami
```

## 모델 다운로드

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| CLI로 다운로드 | `hf download <repo_id> --local-dir <dir>` | 단일 파일/스냅샷 모두 가능 |
| 특정 파일만 다운로드 | `hf download <repo_id> config.json --local-dir <dir>` | 파일명 추가 |
| Python 스냅샷 다운로드 | `python -c 'from huggingface_hub import snapshot_download; snapshot_download("bert-base-uncased", local_dir="model")'` | 코드 내 자동화 |
| 캐시 경로 지정 | `HF_HOME=~/.cache/huggingface hf download <repo_id>` | 환경에 따라 유용 |
| Git clone | `git clone https://huggingface.co/<repo_id>` | git-lfs와 함께 사용 |

```bash
hf download bert-base-uncased --local-dir ./bert-base-uncased
git clone https://huggingface.co/bert-base-uncased
```

## 데이터셋 다운로드

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| datasets 로드 | `python -c 'from datasets import load_dataset; ds = load_dataset("<dataset_id>"); print(ds)'` | 식별자 확인 |
| 로컬 저장 | `python -c 'from datasets import load_dataset; ds = load_dataset("<dataset_id>"); ds.save_to_disk("./data")'` | 로컬 캐시/저장 |
| 특정 split 확인 | `python -c 'from datasets import load_dataset; ds = load_dataset("<dataset_id>", split="train"); print(ds[0])'` | 샘플 점검 |

```bash
python -c 'from datasets import load_dataset; ds = load_dataset("squad"); ds.save_to_disk("./squad")'
```

## 캐시 / 저장 위치

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| HF 홈 확인 | `hf env` | 기본 경로를 먼저 확인 |
| hub 캐시 확인 | `echo "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"` | 모델 캐시 |
| datasets 캐시 확인 | `echo "${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"` | 데이터셋 캐시 |
| 캐시 내용 점검 | `du -sh ~/.cache/huggingface/* 2>/dev/null` | 용량 확인 |
| 캐시 목록 | `hf cache ls` | 현재 저장소 단위 목록 |
| 캐시 상세 목록 | `hf cache ls --revisions` | revision 단위 목록 |
| 캐시 정리 | `hf cache rm <repo_id>` | 특정 캐시 제거 |
| 캐시 prune | `hf cache prune` | detached revision 정리 |

```bash
env | grep -E '^HF_|^HUGGINGFACE_'
hf cache ls
```

## git-lfs / repo 작업

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| git-lfs 설치 확인 | `git lfs version` | 없으면 설치 필요 |
| git-lfs 초기화 | `git lfs install` | 한 번 실행 |
| 대형 파일 clone | `git clone https://huggingface.co/<repo_id>` | clone 후 `git lfs pull` 권장 |
| 큰 파일 추적 | `git lfs track "*.bin"` | 커밋 전에 `.gitattributes` 생성 |
| LFS 파일 pull | `git lfs pull` | 클론 후 실제 파일 받기 |
| repo 생성 | `hf repo create <repo_id> --repo-type model` | `hf repos create`는 alias |
| repo 브랜치 생성 | `hf repo branch create <repo_id> <branch>` | `hf repos branch create`는 alias |

```bash
git lfs install
git lfs track "*.bin"
git lfs pull
```

## 업로드 / 배포

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 폴더 업로드 | `hf upload <repo_id> <path> <path_in_repo>` | 현재 디렉터리도 가능 |
| 모델 파일 업로드 | `hf upload <repo_id> model.safetensors model.safetensors` | 파일 단위 전송 |
| repo 상태 확인 | `git remote -v` | 원격 URL 확인 |
| repo 갱신 | `git pull && git push` | 일반 git 흐름 유지 |

## 자주 쓰는 확인

```bash
hf auth whoami
hf download bert-base-uncased --local-dir ./model
git lfs version
hf cache ls
```

## 참고 메모

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| Python API 설치 | `python -m pip install -U huggingface_hub datasets` | CLI와 함께 사용 |
| 토큰 없이 공개 모델 접근 | `hf download <repo_id>` | 공개 리포지토리는 가능 |
