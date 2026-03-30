# Remote / Transfer

원격 서버 접속, 파일 전송, 포트포워딩, 키 관리, 원격 다운로드를 CLI 기준으로 정리한 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| SSH 접속 | `ssh user@host` | 기본 원격 접속 |
| 키로 SSH 접속 | `ssh -i ~/.ssh/id_ed25519 user@host` | 특정 키 사용 |
| 파일 복사 | `scp local.txt user@host:/tmp/` | 단일 파일 전송 |
| 디렉토리 복사 | `scp -r folder user@host:/tmp/` | 재귀 복사 |
| 동기화 | `rsync -avP ./ local:/path/` | 대용량 전송에 유리 |
| SFTP 접속 | `sftp user@host` | 대화형 파일 전송 |
| 포트포워딩 | `ssh -L 8080:127.0.0.1:80 user@host` | 로컬 포트 터널링 |
| 원격 다운로드 | `wget URL` | 서버에서 직접 받기 |

## SSH

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 기본 접속 | `ssh user@host` | `user`, `host`만 맞추면 됨 |
| 포트 지정 | `ssh -p 2222 user@host` | SSH 포트가 기본값이 아닐 때 |
| 키 지정 | `ssh -i ~/.ssh/id_ed25519 user@host` | 개인키 파일 직접 지정 |
| verbose 모드 | `ssh -v user@host` | 인증/연결 문제 디버깅 |
| 명령 한 번 실행 | `ssh user@host "uname -a"` | 로그인 없이 명령만 실행 |
| 원격 파일 경로 확인 | `ssh user@host "pwd; ls -lah"` | 경로 확인 후 전송할 때 유용 |

## SCP

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 로컬 -> 원격 | `scp file.txt user@host:/tmp/` | 업로드 |
| 원격 -> 로컬 | `scp user@host:/tmp/file.txt ./` | 다운로드 |
| 디렉토리 복사 | `scp -r folder user@host:/tmp/` | `-r` 필요 |
| 포트 지정 | `scp -P 2222 file.txt user@host:/tmp/` | 대문자 `-P` 주의 |
| 키 지정 | `scp -i ~/.ssh/id_ed25519 file.txt user@host:/tmp/` | |
| 진행률 보기 | `scp -v file.txt user@host:/tmp/` | 문제 추적 시 |

## RSYNC

> 주의: `--delete`는 대상에만 있는 파일을 삭제한다. 동기화 전 경로와 옵션을 반드시 확인한다.

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 기본 동기화 | `rsync -av ./src/ user@host:/data/src/` | 재전송 최소화 |
| 진행률 표시 | `rsync -avP ./src/ user@host:/data/src/` | `-P`는 부분 파일/진행률 |
| 삭제 포함 동기화 | `rsync -av --delete ./src/ user@host:/data/src/` | 양쪽 정합성 유지용 |
| SSH 포트 지정 | `rsync -av -e "ssh -p 2222" ./src/ user@host:/data/src/` | SSH 옵션 삽입 |
| 드라이런 | `rsync -av --dry-run ./src/ user@host:/data/src/` | 실제 변경 전 점검 |

## SFTP

`put`을 쓰기 전에는 로컬 시스템에서 해당 파일이 있는 경로로 먼저 이동해야 한다.

| 작업 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 접속 | `sftp user@host` | 대화형 세션 시작 |
| 다운로드 | `get "file-path"` | 원격 -> 로컬 |
| 업로드 | `put "file-path"` | 로컬 -> 원격 |
| 디렉토리 업로드 | `put -r folder` | 재귀 업로드 |
| 포트 지정 | `sftp -P 2222 user@host` | |
| 종료 | `bye` | 또는 `exit` |

## PSCP

다운로드: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html

| 작업 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 다운로드 | `pscp -r user@host:/home/user/dir ./` | 원격 디렉토리 수신 |
| 업로드 | `pscp file user@host:/home/user/dir[/file]` | 단일 파일 전송 |
| 업로드 디렉토리 | `pscp -r dir user@host:/home/user/dir/` | 디렉토리 전송 |
| 포트 지정 | `pscp -P 22 file user@host:/home/user/dir[/file]` | `ssh_init` 오류 대응 예시 유지 |
| 키 지정 | `pscp -i key.ppk file user@host:/home/user/dir/` | PuTTY 키 파일 사용 |

## 포트 포워딩

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 로컬 포워딩 | `ssh -L 8080:127.0.0.1:80 user@host` | 로컬 `8080` -> 원격 `80` |
| 원격 포워딩 | `ssh -R 9000:127.0.0.1:3000 user@host` | 원격에서 로컬 서비스 노출 |
| 동적 포워딩 | `ssh -D 1080 user@host` | SOCKS 프록시 |
| 백그라운드 유지 | `ssh -fN -L 8080:127.0.0.1:80 user@host` | 접속만 유지 |

## 키 관리

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 키 생성 | `ssh-keygen -t ed25519 -C "comment"` | 기본 권장 |
| 공개키 배포 | `ssh-copy-id user@host` | 서버에 키 등록 |
| 공개키 확인 | `cat ~/.ssh/id_ed25519.pub` | 등록할 키 확인 |
| 권한 수정 | `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519` | 권한이 너무 넓으면 SSH가 거부할 수 있음 |
| known_hosts 삭제 | `ssh-keygen -R 147.46.242.124` | 기존 예시 유지 |
| 연결 테스트 | `ssh -T user@host` | 셸 없이 인증만 확인 |

## 원격 다운로드

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 일반 URL 다운로드 | `wget URL -O file` | 파일명 지정 |
| 재개 다운로드 | `wget -c URL -O file` | 중단 후 이어받기 |
| 헤더 확인 | `curl -I URL` | 응답 상태만 보기 |
| 본문 저장 | `curl -L URL -o file` | 리다이렉트 포함 |
| 압축 파일 해제 없이 조회 | `tar -tzf file.tar.gz` | 내려받은 뒤 구조 확인 |
| Google Drive 다운로드 | `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=파일ID' -O 파일명` | 기존 예시 유지 |
| Google Drive 대용량 다운로드 | `wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=파일ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=파일ID" -O 파일명 && rm -rf /tmp/cookies.txt` | 기존 예시 유지 |

