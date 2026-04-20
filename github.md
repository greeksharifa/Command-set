# GitHub / Git

Git 저장소 생성, 브랜치, 동기화, 히스토리 확인, 복구를 CLI 기준으로 바로 실행할 수 있게 정리한 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 새 저장소 시작 | `git init` / `git add` / `git commit` / `git branch -M main` / `git remote add origin ...` / `git push -u origin main` | 아래 `새 저장소 초기화` 참고 |
| 기존 저장소 연결 | `git remote add origin ...` / `git push -u origin main` | 원격만 붙일 때 사용 |
| 현재 상태 확인 | `git status -sb` | 짧은 상태 출력 |
| 변경 확인 | `git diff` / `git diff --staged` | 작업 트리와 staged 구분 |
| 브랜치 이동 | `git switch <branch>` | 새 브랜치 생성은 `-c` |
| 원격 반영 | `git pull --rebase` / `git push` | 히스토리 정리에는 `--rebase` |
| 작업 임시 저장 | `git stash push -m "msg"` | 브랜치 전환 전 유용 |
| 태그 확인 | `git tag -l` | 릴리스 포인트 확인 |
| HTTPS 인증 저장 | `git config --global credential.helper store` | 토큰을 평문 파일에 저장하므로 개인 장비에서만 사용 |
| GH CLI | `gh auth login` / `gh pr create` | GitHub 원격 작업 자동화 |

## 저장소 시작

### 저장소 URL 예시

```bash
https://github.com/greeksharifa/asdfasdfdsfdsa.git
```

### 새 저장소 초기화

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| README 생성 | `echo "# repo-name" >> README.md` | 첫 커밋용 |
| 저장소 초기화 | `git init` | 새 폴더에서 실행 |
| 추적 등록 | `git add README.md` / `git commit -m "first commit"` | 초기 커밋 |
| 기본 브랜치명 변경 | `git branch -M main` | `master` 대신 `main` |
| 원격 연결 | `git remote add origin https://github.com/USER/REPO.git` | URL만 교체 |
| 첫 푸시 | `git push -u origin main` | 업스트림 설정 포함 |

### 기존 저장소에 remote 연결

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 원격 추가 | `git remote add origin https://github.com/USER/REPO.git` | 이미 origin 있으면 `set-url` 사용 |
| 기본 브랜치 맞추기 | `git branch -M main` | 원격 브랜치와 이름 정렬 |
| 첫 동기화 | `git push -u origin main` | 이후 `git push`만 사용 가능 |

## 기본 흐름

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 작업 상태 확인 | `git status -sb` | 현재 브랜치와 변경 파일 확인 |
| 파일 스테이징 | `git add <file>` / `git add .` | 선택 스테이징 권장 |
| 커밋 | `git commit -m "message"` | 의미 있는 단위로 커밋 |
| 마지막 커밋 수정 | `git commit --amend` | 메시지/내용 보정 |
| 로그 간단 확인 | `git log --oneline --decorate --graph --all` | 브랜치 흐름 파악 |
| 파일 단위 이력 | `git log -- <path>` | 특정 파일 추적 |

## 브랜치

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 브랜치 목록 | `git branch` / `git branch -a` | 로컬/원격 브랜치 확인 |
| 새 브랜치 생성 후 이동 | `git switch -c feature/x` | 가장 자주 쓰는 패턴 |
| 브랜치 이동 | `git switch main` | 작업 브랜치 전환 |
| 브랜치 이름 변경 | `git branch -m old new` | 현재 브랜치면 `git branch -m new` |
| 브랜치 삭제 | `git branch -d feature/x` | 병합된 브랜치 삭제 |
| 강제 삭제 | `git branch -D feature/x` | 미병합 브랜치 정리 |

## diff / log / blame

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 작업 트리 변경 | `git diff` | 아직 add 안 한 변경 |
| staged 변경 | `git diff --staged` | 커밋 직전 확인 |
| 특정 파일 diff | `git diff -- path/to/file` | 파일 하나만 확인 |
| 두 커밋 비교 | `git diff <base>..<head>` | 브랜치 비교 |
| 최근 로그 | `git log --oneline -n 20` | 짧게 보기 |
| 그래프 로그 | `git log --graph --oneline --decorate --all` | 브랜치 구조 확인 |
| 라인 단위 추적 | `git blame <file>` | 누가 언제 바꿨는지 확인 |

## 원격 동기화

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 원격 보기 | `git remote -v` | fetch/push URL 확인 |
| 원격 이름 변경 | `git remote rename origin upstream` | fork 구조에서 유용 |
| 원격 URL 변경 | `git remote set-url origin <url>` | origin 교체 |
| 원격 가져오기 | `git fetch --all --prune` | 로컬은 유지, 원격만 갱신 |
| pull 기본형 | `git pull` | fetch + merge |
| rebase pull | `git pull --rebase` | 커밋 히스토리 직선화 |
| push | `git push` | 현재 업스트림 기준 |
| upstream 설정 | `git push -u origin <branch>` | 첫 푸시 때 사용 |

## 인증 / credential 저장

GitHub HTTPS 원격은 비밀번호 대신 personal access token(PAT)이나 `gh auth login`으로 발급된 인증 정보를 사용한다. 반복 입력을 줄이고 싶을 때 Git credential helper를 설정한다.

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 현재 credential helper 확인 | `git config --global --get credential.helper` | 전역 설정 확인 |
| 토큰을 파일에 저장 | `git config --global credential.helper store` | `~/.git-credentials`에 평문 저장 |
| 저장 위치 지정 | `git config --global credential.helper "store --file ~/.git-credentials"` | 기본값을 명시할 때 사용 |
| 저장된 인증 정보 삭제 | `git credential reject` | 아래 입력 예시처럼 protocol/host를 넣고 빈 줄로 종료 |
| helper 설정 해제 | `git config --global --unset credential.helper` | 이후 인증 정보를 다시 물어봄 |
| macOS Keychain 사용 | `git config --global credential.helper osxkeychain` | macOS에서는 평문 저장보다 권장 |

`store`는 편하지만 보안 저장소가 아니라 단순 파일 저장이다. 공유 장비, 서버, 노출 가능성이 있는 개발 컨테이너에서는 `store` 대신 OS 키체인, `cache`, 또는 `gh auth login`을 우선 사용한다.

```bash
git config --global credential.helper store
git push
# Username: USERNAME
# Password: GitHub personal access token
```

저장된 GitHub 인증 정보를 지울 때는 다음처럼 입력한다.

```bash
printf "protocol=https\nhost=github.com\n\n" | git credential reject
```

## stash

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 임시 저장 | `git stash push -m "wip"` | 브랜치 전환 전 사용 |
| 파일 일부만 저장 | `git stash push -p` | hunk 단위 선택 |
| stash 목록 | `git stash list` | 저장된 작업 확인 |
| stash 적용 | `git stash apply` | 목록은 유지 |
| stash 적용 후 삭제 | `git stash pop` | 가장 최근 항목 |
| stash 삭제 | `git stash drop stash@{0}` | 특정 항목 제거 |
| stash 전체 삭제 | `git stash clear` | 주의해서 사용 |

## tag / release

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 태그 목록 | `git tag -l` | 릴리스 버전 확인 |
| 주석 태그 생성 | `git tag -a v1.0.0 -m "release v1.0.0"` | 릴리스용 권장 |
| 특정 커밋 태그 | `git tag -a v1.0.0 <commit>` | 과거 커밋에 태깅 |
| 태그 푸시 | `git push origin v1.0.0` | 개별 태그 |
| 전체 태그 푸시 | `git push --tags` | 다수 태그 한 번에 |
| 태그 삭제(로컬) | `git tag -d v1.0.0` | 로컬만 삭제 |
| 태그 삭제(원격) | `git push origin :refs/tags/v1.0.0` | 원격 태그 제거 |

## 초기화 / 복구

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 파일 수정 되돌리기 | `git restore <file>` | 작업 트리 복구 |
| staged 해제 | `git restore --staged <file>` | add만 취소 |
| 커밋 되돌리기 | `git revert <commit>` | 히스토리 보존형 취소 |
| 브랜치 이동 없이 커밋 이동 | `git reset --soft <commit>` | staged 상태 유지 |
| 작업 내용까지 되돌림 | `git reset --mixed <commit>` | 기본값에 가까움 |
| 강한 되돌림 | `git reset --hard <commit>` | 위험, 신중히 사용 |
| 원격 브랜치 기준 복구 | `git fetch origin && git reset --hard origin/main` | 작업 내용 사라짐 |

## gh CLI

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 로그인 | `gh auth login` | GitHub 계정 연결 |
| 인증 상태 | `gh auth status` | 토큰/호스트 확인 |
| 저장소 정보 | `gh repo view` | 현재 repo 메타데이터 |
| 원격 저장소 생성 | `gh repo create` | 로컬 repo에서 바로 생성 가능 |
| PR 생성 | `gh pr create` | 제목/본문/베이스 지정 |
| PR 체크아웃 | `gh pr checkout <number>` | 로컬로 PR 브랜치 받기 |
| PR 확인 | `gh pr view <number>` | 상태/리뷰 확인 |
| 이슈 목록 | `gh issue list` | 필터와 함께 사용 |
| 릴리스 생성 | `gh release create v1.0.0` | 태그 기반 배포 |
| 웹 열기 | `gh repo view --web` / `gh pr view --web` | 브라우저 연결 |

## 의존성 재설치

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| Python 의존성 강제 재설치 | `pip install --upgrade --no-deps --force-reinstall -r requirements.txt` | 기존 내용 보존용 |
| 패키지 충돌 점검 | `pip check` | 설치 무결성 확인 |
