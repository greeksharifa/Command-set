# tmux 치트시트

tmux 세션, 창, 패널, 레이아웃, 복사모드, 플러그인, 저장/복원을 빠르게 다루는 실전 명령 모음이다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 접속 | `tmux` | 새 세션 생성 또는 기존 세션 재사용 |
| 세션 목록 | `tmux ls` | 현재 세션 확인 |
| 세션 재접속 | `tmux attach -t <세션명>` | 단축: `tmux a -t <세션명>` |
| 세션 종료 | `tmux kill-session -t <세션명>` | `주의`: 세션 내 작업 종료 |
| prefix | `Ctrl-b` | tmux 기본 prefix |

## 세션

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 새 세션 생성 | `tmux new -s <이름>` | 이름을 명시하면 관리가 쉬움 |
| 분리(detach) | `prefix` + `d` | 백그라운드로 보내기 |
| 목록 갱신 | `tmux ls` | 세션 상태 확인 |
| 특정 세션 접속 | `tmux attach -t <이름>` | 원격 SSH에서도 동일 |

```bash
tmux new -s work
tmux ls
tmux attach -t work
```

## 윈도우

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 새 윈도우 | `prefix` + `c` | 작업 분리 |
| 다음 / 이전 윈도우 | `prefix` + `n` / `prefix` + `p` | 순차 이동 |
| 윈도우 이름 변경 | `prefix` + `,` | 이름 지정 권장 |
| 윈도우 목록 | `prefix` + `w` | 선택 메뉴 |

## 패널

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 수직 분할 | `prefix` + `%` | 좌우 분할 |
| 수평 분할 | `prefix` + `"` | 상하 분할 |
| 패널 이동 | `prefix` + 방향키 | 현재 레이아웃에서 이동 |
| 패널 닫기 | `prefix` + `x` | 현재 패널 종료 |
| 패널 번호 표시 | `prefix` + q | 패널 번호 점검 |

```text
prefix + %   # 세로 분할
prefix + "   # 가로 분할
prefix + x   # 패널 닫기
```

## 레이아웃

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 레이아웃 순환 | `prefix` + `space` | 기본/균등 레이아웃 전환 |
| 패널 균등화 | `prefix` + `=` 또는 `space` | 버전에 따라 동작 차이 가능 |
| 레이아웃 저장/복원 | `tmux-resurrect` | 플러그인 사용 |
| 장기 복원 | `tmux-continuum` | 자동 저장/자동 복원에 유용 |

## 복사모드

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 복사모드 진입 | `prefix` + `[` | 스크롤/선택 시작 |
| 검색 | `/` | 복사모드 안에서 텍스트 검색 |
| 선택 시작 | `Space` | 마크 설정 |
| 복사 | `Enter` | 선택 영역 복사 |
| 붙여넣기 | `prefix` + `]` | tmux 버퍼 붙여넣기 |

```text
prefix + [
/검색어
Space
Enter
prefix + ]
```

## 설정 재적용

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 설정 다시 읽기 | `tmux source-file ~/.tmux.conf` | 설정 수정 후 즉시 반영 |
| 확인용 | `tmux show -g` | 현재 전역 설정 확인 |

```bash
tmux source-file ~/.tmux.conf
```

## TPM

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| TPM 설치 | `git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm` | 최초 1회 |
| TPM 로드 | `run '~/.tmux/plugins/tpm/tpm'` | `~/.tmux.conf`에 추가 |
| 플러그인 설치 | `prefix` + `I` | tmux 안에서 실행 |

```bash
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
```

`~/.tmux.conf`

```tmux
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'
set -g @continuum-restore 'on'

run '~/.tmux/plugins/tpm/tpm'
```

## 저장 / 복원 흐름

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 수동 저장 | `prefix` + `Ctrl-s` | 재부팅 전 저장 |
| 수동 복원 | `prefix` + `Ctrl-r` | 재부팅 후 복원 |
| 자동 복원 | `tmux-continuum` | 백그라운드 자동 저장/복원 |

> `주의`: `tmux-resurrect`는 세션, 윈도우, 패널 레이아웃, 작업 디렉터리를 복원하지만, 실행 중 프로세스의 메모리 상태까지 그대로 보존하지는 않는다.

### 권장 흐름

1. `tmux`에서 작업을 시작한다.
2. 재부팅 전에 `prefix` + `Ctrl-s`로 저장한다.
3. 재부팅 후 SSH 재접속 뒤 `tmux`를 실행한다.
4. 자동 복원이 안 되면 `prefix` + `Ctrl-r`로 복원한다.

## 참고

- prefix 기본값은 `Ctrl-b`다.
- 여러 tmux 세션을 함께 저장/복원할 수 있다.
- 장기 실행 작업을 확실히 유지해야 하면 `systemd` 또는 `nohup` 같은 별도 방법이 더 적합하다.
