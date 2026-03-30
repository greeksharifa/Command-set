# Ubuntu / Linux

리눅스에서 자주 쓰는 CLI를 작업별로 모은 치트시트다. 탐색, 검색, 권한, 프로세스, 디스크, 압축, 네트워크, 패키지 관리 위주로 바로 쓸 수 있게 정리했다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 현재 위치 확인 | `pwd` | 작업 디렉토리 확인 |
| 파일/디렉토리 목록 | `ls -lah` | 숨김 파일, 용량, 권한까지 보기 |
| 디렉토리 트리 | `tree` | `sudo apt install tree` 필요 |
| 지정 디렉토리 파일 개수 | `find /path -type f | wc -l` | 폴더 아래 파일 수만 세기 |
| 큰 파일 빠르게 찾기 | `du -ah /path \| sort -hr \| head` | 용량 큰 항목 상위 확인 |
| 문자열 검색 | `grep -RIn "pattern" /path` | 파일명 제외, 내용만 재귀 검색 |
| 파일 찾기 | `find /path -name "*.log"` | 이름 패턴으로 검색 |
| 명령 기록 | `history \| tail` | 최근 명령 재확인 |

## 파일 / 디렉토리 탐색

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 현재 위치 출력 | `pwd` | 현재 작업 경로 확인 |
| 상세 목록 | `ls -lah` | 권한, 소유자, 숨김 파일 포함 |
| 디렉토리 이동 | `cd /path/to/dir` | `cd ..`, `cd -` 자주 사용 |
| 홈으로 이동 | `cd ~` | 사용자 홈 디렉토리 |
| 디렉토리 구조 보기 | `tree -L 2 /path` | 깊이 제한 가능 |
| 파일 메타정보 | `stat file.txt` | 생성/수정/권한 정보 확인 |
| 디렉토리 크기 | `du -sh /path` | 요약 크기 확인 |
| 하위 항목 크기 | `du -sh /path/*` | 무엇이 큰지 빠르게 파악 |

## 파일 개수 / 용량

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 현재 디렉토리 파일 수 | `find . -maxdepth 1 -type f | wc -l` | 현재 폴더만 계산 |
| 재귀 파일 수 | `find . -type f | wc -l` | 하위 폴더까지 포함 |
| 특정 확장자 수 | `find . -name "*.csv" | wc -l` | 패턴 기반 집계 |
| 디렉토리만 세기 | `find . -type d | wc -l` | 폴더 수 확인 |
| 상위 용량 항목 | `du -ah . \| sort -hr \| head -n 20` | 큰 파일/폴더 찾기 |
| 파일 크기 정렬 | `ls -lhS` | 큰 파일부터 정렬 |

## 검색

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 문자열 재귀 검색 | `grep -RIn "text" .` | `-n` 줄 번호, `-I` 바이너리 제외 |
| 대소문자 무시 | `grep -RIni "text" .` | 대소문자 구분 없이 검색 |
| 여러 패턴 검색 | `grep -RInE "foo|bar" .` | 확장 정규식 사용 |
| 파일명 검색 | `find . -name "*report*"` | 이름에 `report` 포함 |
| 내용+파일명 검색 | `rg -n "text" .` | `ripgrep`이 있으면 가장 편함 |
| 특정 확장자만 검색 | `find . -name "*.md" -exec grep -nH "text" {} \;` | 파일 타입 제한 |
| 최근 수정 파일 찾기 | `find . -type f -mtime -3` | 3일 이내 수정 파일 |

## 텍스트 처리

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 첫 줄 보기 | `head file.txt` | 기본 10줄 |
| 마지막 줄 보기 | `tail file.txt` | 로그 확인에 유용 |
| 실시간 로그 보기 | `tail -f app.log` | `Ctrl+C`로 종료 |
| 특정 줄 범위 | `sed -n '10,20p' file.txt` | 10~20행 출력 |
| 문자열 치환 | `sed 's/old/new/g' file.txt` | 미리보기용 |
| 줄바꿈 문자 제거 | `sed -i 's/\r$//' file.txt` | `"$'\r': command not found"` 대응 |
| 컬럼 추출 | `awk '{print $1, $3}' file.txt` | 공백 기준 기본 필드 |
| 중복 제거 | `sort file.txt | uniq` | 정렬 후 중복 제거 |
| 빈 줄 제거 | `grep -v '^$' file.txt` | 공백 라인 제외 |
| 줄 수 세기 | `wc -l file.txt` | 파일 행 수 |

## 권한

> 주의: `chmod`, `chown`, `rm -rf`는 범위를 잘못 잡으면 데이터를 망가뜨릴 수 있다.

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 권한 확인 | `ls -l` | `user/group/other` 확인 |
| 읽기/쓰기/실행 추가 | `chmod u+rwx file` | 소유자 권한 부여 |
| 재귀 권한 변경 | `chmod -R ugo+rwx folder` | 전체 하위 항목 적용 |
| 실행 권한 제거 | `chmod -R go-x folder` | 그룹/기타 실행 제거 |
| 디렉토리만 실행 부여 | `chmod -R a+X folder` | 디렉토리에만 실행 비트 부여 |
| 소유자 변경 | `sudo chown -R user:group folder` | 소유권 이전 |
| 기본 권한 확인 | `umask` | 새 파일 기본 권한 확인 |
| ACL 확인 | `getfacl file` | ACL 사용 시 확인 |

## 프로세스 / 서비스

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 프로세스 확인 | `ps aux | less` | 전체 프로세스 확인 |
| 특정 프로세스 검색 | `ps aux | grep nginx` | `grep` 프로세스도 함께 보임 |
| 실시간 모니터링 | `top` | 기본 시스템 상태 |
| 더 편한 모니터링 | `htop` | 설치 필요할 수 있음 |
| PID로 종료 | `kill 1234` | 정상 종료 시도 |
| 강제 종료 | `kill -9 1234` | 주의: 데이터 손상 가능 |
| 이름으로 종료 | `pkill -f "python app.py"` | 주의: 매칭 범위 확인 |
| 서비스 상태 | `systemctl status nginx` | systemd 기반 |
| 서비스 시작/중지 | `sudo systemctl start nginx` | `stop`, `restart`, `reload` 사용 |
| 부팅 자동 시작 | `sudo systemctl enable nginx` | `disable`로 해제 |
| 로그 확인 | `journalctl -u nginx -f` | 서비스 로그 실시간 확인 |

## 디스크

> 주의: 정리 명령은 되돌리기 어렵다. 삭제 전에 반드시 경로를 재확인한다.

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 디스크 사용량 확인 | `df -h` | 파일시스템 기준 |
| 디렉토리 크기 확인 | `du -sh /path` | 특정 폴더 요약 |
| 큰 항목 찾기 | `du -ah /path | sort -hr | head` | 빠른 용량 진단 |
| 마운트 확인 | `mount` | 현재 마운트 상태 |
| 블록 장치 확인 | `lsblk` | 디스크/파티션 구조 |
| inode 사용량 확인 | `df -i` | 파일 수 과다 이슈 점검 |
| 캐시 정리 | `sudo apt autoremove --purge` | 불필요 패키지 제거 |
| snap 정리 | `sudo apt autoremove --purge snapd` | `/dev/loop`가 많고 공간 부족할 때 사용 |

## 압축

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| gzip 압축 | `gzip file` | 원본 파일 대체 |
| gzip 해제 | `gunzip file.gz` | |
| tar.gz 생성 | `tar -czf archive.tar.gz folder/` | 폴더 압축 |
| tar.gz 해제 | `tar -xzf archive.tar.gz` | |
| tar 내용 보기 | `tar -tzf archive.tar.gz` | 압축 해제 없이 목록 확인 |
| zip 압축 | `zip -r archive.zip folder/` | |
| zip 해제 | `unzip archive.zip` | |

## 네트워크

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| IP 확인 | `ip a` | 기본 네트워크 인터페이스 정보 |
| 라우팅 확인 | `ip route` | 기본 게이트웨이 확인 |
| 외부 접속 확인 | `curl -I https://example.com` | 헤더만 확인 |
| DNS 확인 | `nslookup example.com` | |
| 포트 확인 | `ss -lntp` | 열려 있는 TCP 포트 |
| 연결 테스트 | `ping -c 4 8.8.8.8` | ICMP 응답 확인 |
| 다운로드 | `wget URL` | `curl -O URL`도 자주 사용 |
| 원격 전송 확인 | `nc -vz host 22` | 포트 열림 여부 확인 |

## 패키지 관리

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 패키지 목록 갱신 | `sudo apt update` | 설치 전 먼저 실행 |
| 패키지 업그레이드 | `sudo apt upgrade` | 전체 갱신 |
| 패키지 설치 | `sudo apt install tree` | |
| 패키지 제거 | `sudo apt remove tree` | 설정 파일은 남을 수 있음 |
| 완전 제거 | `sudo apt purge tree` | 설정 파일까지 삭제 |
| 의존성 정리 | `sudo apt autoremove` | 더 이상 필요 없는 의존성 제거 |
| 검색 | `apt search ripgrep` | 패키지 이름 검색 |
| 정보 확인 | `apt show ripgrep` | 패키지 설명/버전 |

## 버전 / 기본 설정

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| Ubuntu 버전 확인 | `cat /etc/issue` | 기존 예시 유지 |
| 배포판 정보 | `cat /etc/os-release` | 더 일반적인 확인 방법 |
| 기본 편집기 변경 | `sudo update-alternatives --config editor` | 기존 예시 유지 |
| 한글 입력 설정 | `ibus-setup` | 기존 예시 유지 |

## 자주 쓰는 설치 예시

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| Python 패키지 설치 | `pip install tensorflow` | 기존 예시 유지 |
| 딥러닝 패키지 설치 | `pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html` | CUDA 버전 맞춤 필요 |
| 분석/시각화 패키지 설치 | `pip install numpy pandas matplotlib seaborn beautifulsoup4 nltk scipy scikit-learn tqdm` | 프로젝트 공통 도구 모음 |

