# Troubleshooting

에러 유형별로 바로 진단하고 복구할 수 있도록 PATH, 라이브러리, 패키지 버전, 프로세스, 포트, 권한, 줄바꿈, JSON 파싱 명령을 모아둔 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 실행 파일 위치 확인 | `which <cmd>` / `type -a <cmd>` | PATH 우선순위 확인 |
| 환경 변수 확인 | `echo $PATH` / `echo $LD_LIBRARY_PATH` | 현재 셸 기준 |
| 패키지 버전 확인 | `python -V` / `pip show <pkg>` / `conda list <pkg>` | 설치/버전 충돌 점검 |
| 프로세스 확인 | `ps aux | grep <name>` / `pgrep -af <name>` | 실행 중인 서비스 찾기 |
| 포트 확인 | `ss -ltnp` / `lsof -i :<port>` | 점유 프로세스 확인 |
| 권한 확인 | `ls -l` / `id` / `namei -l <path>` | 디렉토리 권한 추적 |
| 줄바꿈 확인 | `file <file>` | CRLF 여부 확인 |
| JSON 확인 | `jq . file.json` / `python -m json.tool file.json` | 파싱 전용 검증 |

## PATH / 실행 파일

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 현재 실행 파일 찾기 | `which python` | 쉘이 보는 경로 확인 |
| 후보 경로 전체 출력 | `type -a python` | alias 포함 확인 |
| 셸 해시 초기화 | `hash -r` | 새로 설치한 명령이 안 잡힐 때 |
| PATH 임시 추가 | `export PATH=/usr/local/bin:$PATH` | 현재 셸에서만 적용 |
| PATH 영구 추가 | `echo 'export PATH=...' >> ~/.zshrc` | 셸 설정 파일 수정 |
| conda 환경 확인 | `conda info --envs` | 활성 환경과 경로 점검 |

## 라이브러리 / 동적 링크

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 바이너리 의존성 확인 | `ldd <binary_or_so>` | 빠진 `.so` 찾기 |
| 공유 라이브러리 캐시 확인 | `ldconfig -p | grep <name>` | 시스템 등록 여부 |
| 런타임 경로 확인 | `echo $LD_LIBRARY_PATH` | CUDA/커스텀 라이브러리 점검 |
| 패키지 설치 확인 | `python -c "import <pkg>; print(<pkg>.__version__)"` | import 자체가 되는지 확인 |
| wheel 재설치 | `pip uninstall <pkg> -y && pip install --no-cache-dir --force-reinstall <pkg>` | 깨진 설치 복구 |
| 빌드 격리 없이 설치 | `pip install <pkg> --no-build-isolation --no-cache-dir` | `flash-attn` 류에서 자주 사용 |
| CUDA 경로 추가 | `export PATH=/usr/local/cuda/bin:$PATH` | `nvcc`/CUDA 툴체인 노출 |
| CUDA 라이브러리 추가 | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` | `undefined symbol` 점검용 |

### Undefined Symbol

다음과 같은 오류:

```bash
ImportError: /home/ywjang/miniconda3/envs/IGVLM/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops15sum_IntList_out4callERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS5_10ScalarTypeEERS2_
```

복구 순서:

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 패키지 제거 | `pip uninstall flash-attn -y` | 충돌 버전 제거 |
| CUDA 경로 노출 | `export PATH=/usr/local/cuda/bin:$PATH` | 툴체인 확인 |
| CUDA 라이브러리 노출 | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` | 런타임 링크 해결 |
| 재설치 | `pip install flash-attn --no-build-isolation --no-cache-dir` | 기존 내용 유지 |

## 패키지 버전 / Python

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| Python 버전 | `python -V` | 인터프리터 버전 확인 |
| pip 버전 | `pip -V` | 어떤 Python에 묶였는지 확인 |
| 설치된 패키지 목록 | `pip freeze` / `pip list` | 재현용 스냅샷 |
| 패키지 상세 | `pip show <pkg>` | 위치/의존성 확인 |
| 충돌 검사 | `pip check` | 누락/충돌 패키지 확인 |
| requirements 재설치 | `pip install --upgrade --no-deps --force-reinstall -r requirements.txt` | 기존 의존성 정리용 |
| 특정 버전 고정 | `pip install pkg==1.2.3` | 버전 롤백/재현 |

## 프로세스 / 포트

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 프로세스 검색 | `pgrep -af <name>` | PID와 커맨드라인 함께 보기 |
| 프로세스 전체 보기 | `ps aux | grep <name>` | 전통적인 점검법 |
| 포트 점유 확인 | `ss -ltnp` | 리슨 포트와 PID 확인 |
| 특정 포트 확인 | `lsof -i :8000` | 8000번 사용 주체 찾기 |
| 프로세스 종료 | `kill <pid>` | 정상 종료 우선 |
| 강제 종료 | `kill -9 <pid>` | 최후 수단 |
| 포트 비우기 | `fuser -k 8000/tcp` | 해당 포트 프로세스 종료 |

## 권한 / 소유자

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 파일 권한 확인 | `ls -l` | 읽기/쓰기/실행 확인 |
| 디렉토리 경로별 권한 | `namei -l <path>` | 상위 디렉토리까지 추적 |
| 소유자 변경 | `sudo chown -R user:group <path>` | 소유권 문제 복구 |
| 권한 변경 | `chmod -R ugo+rwx <path>` | 광범위 변경, 신중히 |
| 실행 권한 부여 | `chmod +x <file>` | 스크립트 실행용 |
| 현재 사용자 확인 | `id` | UID/GID 확인 |
| sudo 사용 가능 여부 | `sudo -l` | 권한 범위 점검 |

## 줄바꿈 / 인코딩

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 파일 형식 확인 | `file <file>` | CRLF, UTF-8, 바이너리 확인 |
| CRLF 제거 | `sed -i 's/\r$//' <file>` | `bash: $'\r'` 계열 오류 복구 |
| dos2unix 변환 | `dos2unix <file>` | 설치돼 있으면 가장 간단 |
| 끝문자 확인 | `cat -A <file>` | `^M` 등 표시 |
| 인코딩 확인 | `locale` / `iconv -f utf-8 -t utf-8 <file> >/dev/null` | 인코딩 불일치 점검 |

### `$'\\r': command not found`

```bash
sed -i 's/\r$//' 파일명
```

## JSON / 파싱

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| JSON 포맷 검사 | `jq . file.json` | 가장 빠른 확인 |
| Python 표준 파서 | `python -m json.tool file.json` | 들여쓰기 포함 출력 |
| 문자열 파싱 테스트 | `python - <<'PY' ... PY` | 복잡한 경우에 사용 |
| key 유무 확인 | `jq -e '.key' file.json` | 실패 코드로 검증 가능 |
| JSONL 확인 | `jq -c . file.jsonl | head` | 한 줄 한 객체 점검 |

### JSON single quote / apostrophe

```python
def safe_literal_eval(s):
    """
    Safely evaluates a single-quoted string containing apostrophes into a Python object.
    """
    s_fixed = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', s)

    print(f"Fixed string: {s_fixed}")

    try:
        result = literal_eval(s_fixed)
        return result
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

    try:
        parsed = json.loads(s_fixed)
        return parsed
    except json.JSONDecodeError as e:
        print(f"Error: Unable to parse string. Details: {e}")
        return None
```

## 네트워크 / 다운로드

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| DNS 확인 | `getent hosts <host>` | 호스트명 해석 확인 |
| 연결성 확인 | `ping -c 1 <host>` | ICMP 허용 시 |
| HTTP 응답 확인 | `curl -I <url>` | 헤더만 확인 |
| 다운로드 재시도 | `wget --retry-connrefused --waitretry=1 <url>` | 불안정한 네트워크 대응 |

## 시스템 로그 / 디버그

| 목적 | 명령 | 예시/비고 |
| -------- | -------- | -------- |
| 최근 커널 로그 | `dmesg | tail -n 50` | 디바이스/드라이버 문제 |
| 서비스 로그 | `journalctl -xe` | systemd 환경 |
| 디스크 확인 | `df -h` | 공간 부족 진단 |
| 메모리 확인 | `free -h` | OOM 가능성 점검 |

