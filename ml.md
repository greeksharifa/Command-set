# ML / CUDA / Visualization

CUDA, GPU, PyTorch/TensorFlow 환경, Jupyter 변환, ffmpeg, graphviz 점검에 자주 쓰는 명령을 모아둔 치트시트다.

## 빠른 찾기

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| GPU 확인 | `nvidia-smi` | 가장 먼저 확인 |
| CUDA 버전 확인 | `nvcc --version` | CUDA Toolkit 설치 상태 점검 |
| PyTorch CUDA 확인 | `python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'` | 바로 실행 가능한 최소 확인 |
| TensorFlow GPU 확인 | `python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'` | GPU 인식 여부 확인 |

## CUDA / GPU 점검

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| NVIDIA 드라이버 / GPU 상태 | `nvidia-smi` | 모델명, 드라이버, VRAM 확인 |
| CUDA Toolkit 버전 | `nvcc --version` | 컴파일러 기준 버전 확인 |
| CUDA 설치 경로 | `ls -ld /usr/local/cuda*` | 심볼릭 링크 포함 확인 |
| cuDNN 헤더 버전 | `grep -A 2 CUDNN_MAJOR /usr/local/cuda/include/cudnn_version.h` | 설치된 cuDNN 버전 확인 |
| GPU 사용 프로세스 | `nvidia-smi pmon -c 1` | 점유 프로세스 확인 |
| 실시간 모니터링 | `watch -n 1 nvidia-smi` | 반복 확인 |

```bash
nvidia-smi
nvcc --version
grep -A 2 CUDNN_MAJOR /usr/local/cuda/include/cudnn_version.h
```

## PyTorch 환경 점검

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 설치 버전 | `python -c 'import torch; print(torch.__version__)'` | 현재 PyTorch 버전 |
| CUDA 사용 가능 여부 | `python -c 'import torch; print(torch.cuda.is_available())'` | `True/False` 확인 |
| CUDA 런타임 버전 | `python -c 'import torch; print(torch.version.cuda)'` | PyTorch 빌드 기준 버전 |
| cuDNN 버전 | `python -c 'import torch; print(torch.backends.cudnn.version())'` | backend 점검 |
| GPU 개수 | `python -c 'import torch; print(torch.cuda.device_count())'` | 다중 GPU 확인 |
| 현재 GPU 이름 | `python -c 'import torch; print(torch.cuda.get_device_name(0))'` | 인덱스 0 기준 |

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
python -c 'import torch; print(torch.cuda.device_count())'
```

## TensorFlow 환경 점검

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| TensorFlow 버전 | `python -c 'import tensorflow as tf; print(tf.__version__)'` | 설치 버전 확인 |
| GPU 장치 목록 | `python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'` | GPU 인식 여부 |
| 빌드 정보 | `python -c 'import tensorflow as tf; print(tf.sysconfig.get_build_info())'` | CUDA/cuDNN 빌드 정보 |
| 메모리 증가 설정 확인 | `python -c 'import tensorflow as tf; print(tf.config.experimental.get_memory_growth)'` | 설정 함수 존재 여부 |

```bash
python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
```

## GPU 지정 / 디버그

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 특정 GPU만 보이게 실행 | `CUDA_VISIBLE_DEVICES=1 python train.py` | 1번 GPU만 노출 |
| 복수 GPU 제한 | `CUDA_VISIBLE_DEVICES=0,1 python train.py` | 선택적 노출 |
| 디바이스 설정 확인 | `python -c 'import os; print(os.environ.get("CUDA_VISIBLE_DEVICES"))'` | 환경 변수 확인 |
| 메모리 사용 확인 | `nvidia-smi` | 프로세스별 메모리 확인 |

```bash
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.is_available())'
```

## Jupyter / Notebook 변환

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| notebook을 script로 변환 | `jupyter nbconvert --to script notebook.ipynb` | `.py` 생성 |
| notebook을 markdown으로 변환 | `jupyter nbconvert --to markdown notebook.ipynb` | 문서 추출 |
| notebook 실행 후 저장 | `jupyter nbconvert --to notebook --execute notebook.ipynb` | 실행 검증 |
| nbconvert 설치 | `python -m pip install -U nbconvert` | 버전 문제 시 최신화 |

```bash
jupyter nbconvert --to script notebook.ipynb
```

## ffmpeg

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 설치 여부 확인 | `ffmpeg -version` | 버전 / 빌드 정보 |
| 인코더 확인 | `ffmpeg -encoders | grep 264` | `libx264` 존재 여부 점검 |
| 영상 길이 확인 | `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 input.mp4` | 길이만 출력 |
| 구간 자르기 | `ffmpeg -ss 10 -i input.mp4 -t 5 -c:v libx264 -c:a aac -y out.mp4` | 자주 쓰는 형태 |
| 오디오 없이 변환 | `ffmpeg -i input.mp4 -an out.mp4` | 테스트용 |

```bash
ffmpeg -ss 10 -i input.mp4 -t 5 -c:v libx264 -c:a aac -y out.mp4
ffmpeg -encoders | grep 264
```

## Graphviz

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| 설치 여부 확인 | `dot -V` | Graphviz 바이너리 확인 |
| 렌더링 | `dot -Tpng input.dot -o output.png` | 가장 기본적인 변환 |
| SVG 렌더링 | `dot -Tsvg input.dot -o output.svg` | 문서 삽입에 유용 |
| 파이썬 패키지 설치 | `python -m pip install graphviz` | Python 바인딩 |
| 시스템 PATH 확인 | `echo $PATH | tr ':' '\n' | grep -i graphviz` | `dot` 경로 포함 여부 |

```bash
dot -Tsvg model.dot -o model.svg
dot -Tpng model.dot -o model.png
```

## 모델 시각화

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| torchviz 설치 | `python -m pip install torchviz` | PyTorch 그래프 시각화 |
| 간단한 dot 생성 | `python -c 'from graphviz import Source; Source("digraph{a->b}").render("out", format="svg", cleanup=True)'` | 복붙용 최소 예제 |
| 의존성 확인 | `python -c 'import graphviz; print(graphviz.__file__)'` | 패키지 위치 확인 |

## 자주 쓰는 확인

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
ffmpeg -version
dot -V
```

## 참고 메모

| 목적 | 명령 | 예시/비고 |
| --- | --- | --- |
| CUDA / cuDNN 자료 | `nvidia-smi` / `nvcc --version` | 설치가 맞는지 먼저 확인 |
| 오래된 nbconvert 이슈 회피 | `python -m pip install "nbconvert==5.6.1"` | 필요한 경우에만 고정 |
| Windows용 Graphviz 경로 | `set PATH=%PATH%;C:\Program Files\Graphviz\bin` | 임시 세션용 |

