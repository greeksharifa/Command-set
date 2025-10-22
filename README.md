# Command-set

### Github

![image](https://github.com/greeksharifa/Command-set/assets/26247624/a6ce7c34-72be-45e6-8fae-7e6c1188b04e)

```bash
https://github.com/greeksharifa/asdfasdfdsfdsa.git

# ------------------------------------------------------------------------------------

echo "# asdfasdfdsfdsa" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/greeksharifa/asdfasdfdsfdsa.git
git push -u origin main

# ------------------------------------------------------------------------------------

git remote add origin https://github.com/greeksharifa/asdfasdfdsfdsa.git
git branch -M main
git push -u origin main
```

#### dependency 관련 오류
강제 설치
```bash
pip install --upgrade --no-deps --force-reinstall -r requirements.txt
```

### huggingface

#### model local download
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="bert-base-uncased", cache_dir="/data2/")
```

### Windows

####

Magic Trackpad 2: https://github.com/imbushuo/mac-precision-touchpad/releases/tag/2105-3979

#### 실행(Win + R)
| Run | Command |
| -------- | -------- |
| OS 정보 확인(Legacy/UEFI) | msinfo32 |
| 부팅 정보 확인(멀티부팅) | msconfig | 
| 부팅 시 암호 입력 해제 | netplwiz |
| 빌드 버전 확인 | winver |
| 컴퓨터 관리 | compmgmt.msc |
| 디스크 정리 | cleanmgr | 
| 공인ip | nslookup myip.opendns.com. resolver1.opendns.com |

#### 강제종료

tasklist \
taskkill /f /im cmd.exe  \
taskkill /f /PID 36036

#### 파일 강제삭제

del /s /q (filename) \ 
rd /s /q (name)

#### Windows Terminal 여러줄 붙이기 경고 끄기

`C:\Users\<username>\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState` 경로에서 `settings.json` 수정

```json
{
    "multiLinePasteWarning": false,
    ...
}
```

#### 멀티부팅 이름 변경
bcdedit /v \
bcdedit /set {identifier} description "name"


#### 키 매핑 바꾸기
- https://lightinglife.tistory.com/entry/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%97%90%EC%84%9C-%EB%A7%A5%EC%B2%98%EB%9F%BC-Capslock%EC%BA%A1%EC%8A%A4%EB%9D%BD%ED%82%A4%EB%A5%BC-%ED%95%9C%EC%98%81%ED%82%A4%EB%A1%9C-%EB%B3%80%EA%B2%BD%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95by-%EB%A0%88%EC%A7%80%EC%8A%A4%ED%8A%B8%EB%A6%AC-%ED%8E%B8%EC%A7%91
- 컴퓨터\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout
- 00 00 00 00 00 00 00 00
- 02 00 00 00 38 00 3A 00
- 3A 00 38 00

#### windows 11
cmd를 관리자 권한으로 실행한다

```bash
slmgr /ipk W269N-WFGWX-YVC9B-4J6C9-T83GX
slmgr /skms kms8.msguides.com
slmgr /ato
```


#### Powershell

최고성능 해제 \
powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61

파일 수 세기 recursively \ 
powershell -Command "(Get-ChildItem -File -Recurse | Measure-Object).Count"

기본 앱 제거

```
Get-AppxPackage *3dbuilder* | Remove-AppxPackage
Get-AppxPackage *windowscamera* | Remove-AppxPackage
Get-AppxPackage *officehub* | Remove-AppxPackage
Get-AppxPackage *skypeapp* | Remove-AppxPackage
Get-AppxPackage *getstarted* | Remove-AppxPackage
Get-AppxPackage *windowsmaps* | Remove-AppxPackage
get-appxpackage *solitaire* | remove-appxpackag
Get-AppxPackage *onenote* | Remove-AppxPackage
Get-AppxPackage *people* | Remove-AppxPackage
Get-AppxPackage Microsoft.Microsoft3DViewer | Remove-AppxPackage
get-appxpackage *messaging* | remove-appxpackage
get-appxpackage *sway* | remove-appxpackage
get-appxpackage *communicationsapps* | remove-appxpackage
get-appxpackage *zunevideo* | remove-appxpackage
get-appxpackage *bingfinance* | remove-appxpackage
get-appxpackage *bingsports* | remove-appxpackage
get-appxpackage *bingnews* | remove-appxpackage
get-appxpackage *camera* | remove-appxpackage
get-appxpackage *maps* | remove-appxpackage
get-appxpackage *soundrecorder* | remove-appxpackage
get-appxpackage *wallet* | remove-appxpackage
get-appxpackage *connectivitystore* | remove-appxpackage
get-appxpackage *oneconnect* | remove-appxpackage
get-appxpackage *sticky* | remove-appxpackage
get-appxpackage *holographic* | remove-appxpackage
Get-AppxPackage *zunemusic* | Remove-AppxPackage
Get-AppxPackage *soundrecorder* | Remove-AppxPackage
Get-AppxPackage -allusers Microsoft.549981C3F5F10 | Remove-AppxPackage


get-appxpackage *commsphone* | remove-appxpackage
get-appxpackage *phone* | remove-appxpackage
get-appxpackage *xbox* | remove-appxpackage
Get-AppxPackage *xboxapp* | Remove-AppxPackage
get-appxpackage -allusers Microsoft.549981C3F5F10 | Remove-AppxPackage
Set-MpPreference -DisableRealtimeMonitoring $true
Get-AppxPackage *bingweather* | Remove-AppxPackage
Get-AppxPackage Microsoft.YourPhone -AllUsers | Remove-AppxPackage
Get-appxpackage *windowsphone* | remove-appxpackage
Get-AppxPackage Microsoft.YourPhone -AllUsers | Remove-AppxPackage
```

### python 절대 경로 import

```python
import sys
sys.path.append('/home/test')
```



### CUDA 설치

- https://www.python.org/downloads/release/python-386/
- https://www.tensorflow.org/install/gpu
- https://pytorch.org/get-started/locally/

- https://www.nvidia.com/en-us/geforce/geforce-experience/
- https://developer.nvidia.com/cuda-toolkit-archive
- https://developer.nvidia.com/rdp/cudnn-archive

Tensorflow 버전 확인:  \
https://coding-groot.tistory.com/87

cuda / cudnn 버전 확인
```python
import torch
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("cuda version: {}".format(torch.version.cuda))
```
```bash
# cudnn
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### GPU 번호 지정

```python
# GPU 할당 변경하기
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(GPU_NUM)/1024**3,1), 'GB')
```

### ipynb to py

pip uninstall nbconvert
pip install nbconvert==5.6.1

jupyter nbconvert --to script [filename].ipynb 


### Ubuntu(Linux)

#### ubuntu 버전 확인
```bash
cat /etc/issue
```

| Run | Command |
| -------- | -------- |
| 기본 편집기 변경 | sudo update-alternatives --config editor |
| 한글 설치 | ibus-setup |

#### $'￦r': command not found 
```bash
sed -i 's/\r$//' 파일명
```


#### 권한
```bash
# user(owner)/group/other_user
# +: 권한부여, -: 권한삭제
chmod -R ugo+rwx 'folder'
chmod -R go-x 'folder'
chmod -R -x+X -- 'folder with restored backup'    # How to recursively remove execute permissions from files without touching folders
```

#### python 필수 설치 목록

pip install tensorflow

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy pandas matplotlib seaborn beautifulsoup4 nltk scipy scikit-learn tqdm


### terminal에서 파일 트리 출력

```bash
sudo apt-get install tree
# To see the directory tree, use
tree /path/to/folder
# Or navigate to a directory and just use
tree
```

### no left space on device

/dev/loop/ 들이 많고 가득 찼을 경우

```bash
sudo apt autoremove --purge snapd
```


### 구글 드라이브 파일 받기

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=파일ID' -O 파일명

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=파일ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=파일ID" -O 파일명 && rm -rf /tmp/cookies.txt

#### Python font list 출력하기

```python
import matplotlib.font_manager
from IPython.core.display import HTML

print(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))

### HTML로 보고 싶다면? (잘 되는진 모름)

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))
```

```python
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
```


#### Turn a Unicode string to plain ASCII
thanks to https://stackoverflow.com/a/518232/2809427

```python
import unidecode

accented_string = 'Plav (České Budějovice)'
unaccented_string = unidecode.unidecode(accented_string)

print(unaccented_string)
```

#### conda install < requirements.txt in windows

| Type | Command |
| -------- | -------- |
| manually | for /f %i in (requirements.txt) do conda install --yes %i |
| BAT script | for /f %%i in (requirements.txt) do conda install --yes %%i |

#### sftp usage

put의 경우 local system의 해당 파일이 있는 곳으로 먼저 이동해야 함.

| Load | Command | 
| -------- | -------- | 
| Connect | sftp 계정명@ip주소
| Download | get "file-path"
| Upload | put "file-path"

#### pscp usage

다운로드: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html

| Load | Command | 
| -------- | -------- | 
| Download | pscp -r 계정명@ip주소:/home/계정명/디렉토리 ./ | 
| Upload | pscp 파일 계정명@ip주소:/home/계정명/디렉토리\[/파일명] | 
| | pscp -r 디렉토리 계정명@ip주소:/home/계정명/디렉토리/ |
| ssh_init 에러 시 | pscp -P 22 파일 계정명@ip주소:/home/계정명/디렉토리\[/파일명] | 

#### ssh

ssh-keygen -R 147.46.242.124

### Windows cmd

#### 원하는 크기의 파일명 

fsutil file createnew 파일명 사이즈지정


### Graphviz

#### Install

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

For Windows:
1. Install windows package from: http://www.graphviz.org/download/
2. Install python graphviz package
3. Add C:\Program Files (x86)\Graphviz2.38\bin to User path
4. Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path

#### Visualization

```python
import torch
from torch import nn
from torchviz import *
import graphviz
from subprocess import check_call

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

dot_graph = make_dot(y.mean(), params=dict(model.named_parameters()))

def save_graph_as_svg(dot_string, output_file_name):
    if type(dot_string) is str:
        g = graphviz.Source(dot_string)
    elif isinstance(dot_string, (graphviz.dot.Digraph, graphviz.dot.Graph)):
        g = dot_string
    g.format='svg'
    g.filename = output_file_name
    g.directory = '../../assets/images/markdown_img/'
    g.render(view=False)
    return g

save_graph_as_svg(dot_graph, 'simple_model')

check_call(['dot','-Tpng','simple_model','-o','simple_model.png'])
```

### clip video by ffmpeg
```python
"""
conda uninstall ffmpeg  
conda install -c conda-forge ffmpeg 

# if Unknown encoder 'libx264': check
ffmpeg -encoders | grep 264

"""

import csv
import subprocess
import os
from tqdm import tqdm


#  ffmpeg -y -ss start_time -to end_time -i input_path -codec copy output_path
def clip_video(input_video, output_video, start_time, end_time):
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', input_video,
        '-t', str(end_time - start_time),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',
        output_video
    ]
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        print(f"Successfully clipped: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error clipping {input_video}: {e}")
        print(f"FFmpeg error output: {e.stderr.decode()}")

def main():
    csv_file = 'Video_Segments.csv'  # Replace with your CSV file name
    video_directory = '/data/charades/Charades_v1_480/'   # Replace with the directory containing your videos
    output_directory = 'videos/'  # Replace with the directory where you want to save the clipped videos

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(tqdm(csv_reader)):
            video_id = row['video_id']
            start_time = float(row['start'])
            end_time = float(row['end'])

            input_video = os.path.join(video_directory, f"{video_id}.mp4")
            output_video = os.path.join(output_directory, f"{video_id}_{start_time}_{end_time}.mp4")

            print(f"Clipping {input_video} from {start_time} to {end_time}")
            clip_video(input_video, output_video, start_time, end_time)
            # if i > 5: break

if __name__ == "__main__":
    main()
```

---

# Huggingface

## 모델 로컬 다운로드
```bash
git lfs clone https://huggingface.co/<model_id>
```

## 데이터셋 로컬 다운로드
```python
from datasets import load_dataset
dataset = load_dataset(hugging face dataset)
DATA_PATH = './' #현재 폴더 위치
dataset.save_to_disk(DATA_PATH)
```

---


# Resolving Error 

## Undefined Symbol

```bash
ImportError: /home/ywjang/miniconda3/envs/IGVLM/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops15sum_IntList_out4callERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS5_10ScalarTypeEERS2_
```
```bash
pip uninstall flash-attn -y
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation --no-cache-dir
```

## JSON single quote and aposthrophe
```python
def safe_literal_eval(s):
    """
    Safely evaluates a single-quoted string containing apostrophes into a Python object.
    """
    # Step 1: Escape single quotes inside the string values
    # Matches single quotes that are not part of the enclosing quotes
    s_fixed = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', s)
    
    print(f"Fixed string: {s_fixed}")
    
    try:
        # Step 2: Evaluate the fixed string
        result = literal_eval(s_fixed)
        return result
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    # or
    # Step 2: Fix JSON style (ensure keys are quoted)
    try:
        # Use json.loads to validate the result
        parsed = json.loads(s_fixed)
        return parsed
    except json.JSONDecodeError as e:
        print(f"Error: Unable to parse string. Details: {e}")
        return None
```

