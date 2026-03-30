# Windows 치트시트

Windows에서 자주 쓰는 CLI, 시스템 도구, 부팅, 서비스, 디스크 작업을 빠르게 찾는 실전 명령 모음이다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 시스템 정보 | `msinfo32` | BIOS/UEFI, 하드웨어, 부팅 정보 확인 |
| 버전 확인 | `winver` | Windows 빌드/에디션 빠르게 확인 |
| 고급 시작 옵션 | `msconfig` | 멀티부팅, 부팅 옵션, 진단 시작 |
| 사용자 계정 설정 | `netplwiz` | 자동 로그인/비밀번호 입력 해제 설정 |
| 컴퓨터 관리 | `compmgmt.msc` | 디스크 관리, 이벤트 뷰어, 서비스로 진입 |
| 디스크 정리 | `cleanmgr` | 임시 파일 정리 |
| 공인 IP 확인 | `nslookup myip.opendns.com. resolver1.opendns.com` | DNS 조회 기반 공인 IP 확인 |

## CMD / PowerShell 기본

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 관리자 CMD 실행 | `cmd` | `Win + R` 후 입력 가능 |
| 관리자 PowerShell | `powershell` | 관리자 권한으로 실행 권장 |
| 현재 폴더 파일 수 | `powershell -Command "(Get-ChildItem -File -Recurse | Measure-Object).Count"` | 파일 수만 셀 때 유용 |
| 큰 파일 생성 | `fsutil file createnew 파일명 사이즈` | 예: `fsutil file createnew test.bin 1048576` |

## 프로세스 / 작업 종료

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 프로세스 목록 | `tasklist` | 실행 중 프로세스 확인 |
| 이름으로 강제 종료 | `taskkill /f /im cmd.exe` | `주의`: 강제 종료 |
| PID로 강제 종료 | `taskkill /f /PID 36036` | `주의`: 대상 PID 확인 후 사용 |

```bat
tasklist
taskkill /f /im cmd.exe
taskkill /f /PID 36036
```

## 파일 / 디스크

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 파일 강제 삭제 | `del /s /q 파일명` | `주의`: 복구 어려움, 경로 확인 필요 |
| 폴더 강제 삭제 | `rd /s /q 폴더명` | `주의`: 하위 파일까지 삭제 |
| 디스크 정리 | `cleanmgr` | 시스템 임시 파일 정리 |
| 고성능 전원 계획 | `powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61` | PowerShell/관리자 권한 권장 |

```bat
del /s /q filename
rd /s /q foldername
```

## 네트워크

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 공인 IP 조회 | `nslookup myip.opendns.com. resolver1.opendns.com` | 외부 DNS를 이용한 조회 |
| 기본 네트워크 정보 | `ipconfig /all` | 필요 시 추가 확인용 |
| 라우팅 확인 | `route print` | 네트워크 경로 점검 |
| 연결 확인 | `ping 8.8.8.8` | DNS 문제와 분리해서 점검 |

```bat
ipconfig /all
route print
ping 8.8.8.8
```

## 시스템 정보 / 진단

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| OS/부팅 상세 | `msinfo32` | UEFI/Legacy 포함 확인 |
| Windows 버전 | `winver` | 빌드 번호 확인 |
| 장치/관리 도구 | `compmgmt.msc` | 이벤트 뷰어, 서비스, 디스크 관리 |
| 디스크 정리 | `cleanmgr` | 불필요한 파일 제거 |

## 서비스 / 앱 관리

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 서비스 관리 | `services.msc` | 서비스 시작/중지/자동 시작 설정 |
| 앱 제거(Appx) | `Get-AppxPackage *xbox* | Remove-AppxPackage` | `주의`: 사용자 앱 제거, 복구 번거로움 |
| 기본 앱 제거 | `Get-AppxPackage *windowscamera* | Remove-AppxPackage` | 패턴에 맞춰 개별 제거 |
| Defender 실시간 감시 끄기 | `Set-MpPreference -DisableRealtimeMonitoring $true` | `주의`: 보안 약화, 관리자 권한 필요 |

```powershell
Get-AppxPackage *3dbuilder* | Remove-AppxPackage
Get-AppxPackage *windowscamera* | Remove-AppxPackage
Get-AppxPackage *xbox* | Remove-AppxPackage
Set-MpPreference -DisableRealtimeMonitoring $true
```

## Windows Terminal

| 목적 | 명령 / 경로 | 예시/비고 |
| --- | --- | --- |
| 설정 파일 열기 | `C:\Users\<username>\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json` | 사용자별 설정 경로 |
| 여러 줄 붙여넣기 경고 끄기 | `"multiLinePasteWarning": false` | 설정 JSON에 추가 |

```json
{
  "multiLinePasteWarning": false
}
```

## 부팅 / 멀티부팅

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 부팅 항목 확인 | `bcdedit /v` | 식별자 확인 |
| 부팅 이름 변경 | `bcdedit /set {identifier} description "name"` | 멀티부팅 이름 수정 |
| 부팅 옵션 확인 | `msconfig` | 진단 시작, 기본 OS 설정 |

```bat
bcdedit /v
bcdedit /set {identifier} description "name"
```

## 레지스트리 / 키 매핑

| 목적 | 명령 / 경로 | 예시/비고 |
| --- | --- | --- |
| 키 매핑 위치 | `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout` | 레지스트리 편집기에서 수정 |
| 캡스락/한영키 매핑 값 | `00 00 00 00 00 00 00 00 ...` | `주의`: 레지스트리 백업 후 적용 |

```text
00 00 00 00 00 00 00 00
02 00 00 00 38 00 3A 00
3A 00 38 00
```

## 라이선스 / 활성화 상태 확인

> `주의`: 관리자 권한 `cmd`에서 실행하는 것이 안전하다.

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 라이선스 요약 확인 | `slmgr /dli` | 현재 라이선스 상태 요약 |
| 상세 라이선스 정보 | `slmgr /dlv` | 활성화/채널/부분 키 확인 |
| 영구 활성화 여부 확인 | `slmgr /xpr` | 만료 여부 확인 |

```bat
slmgr /dli
slmgr /dlv
slmgr /xpr
```

## 참고

### 드라이버 / 장치

- Magic Trackpad 2: https://github.com/imbushuo/mac-precision-touchpad/releases/tag/2105-3979
