# macOS

macOS에서 자주 필요한 설치와 설정 명령을 정리하는 문서다.

## vivid 설정

macOS에서 `vivid`를 최소 설정으로 붙일 때 필요한 명령만 정리했다.

### 1. vivid와 GNU ls 설치

```bash
brew install vivid coreutils
```

### 2. ls 색상 설정

```bash
echo 'export LS_COLORS="$(vivid generate one-dark)"' >> ~/.zshrc
echo 'alias ls="gls --color=auto"' >> ~/.zshrc
source ~/.zshrc
```
