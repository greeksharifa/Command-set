# Command-set

### Windows

#### 실행(Win + R)
| Run | Command |
| -------- | -------- |
| OS 정보 확인(Legacy/UEFI) | msinfo32 |
| 부팅 정보 확인(멀티부팅) | msconfig | 
| 부팅 시 암호 입력 해제 | netplwiz |
| 빌드 버전 확인 | winver |
| 컴퓨터 관리 | compmgmt.msc |
| 디스크 정리 | cleanmgr | 

#### 강제종료

tasklist \
taskkill /f /im cmd.exe  \
taskkill /f /PID 36036

#### 파일 강제삭제

del /s /q (filename) \ 
rd /s /q (name)


#### 멀티부팅 이름 변경
bcdedit /v \
bcdedit /set {identifier} description "name"



#### Powershell

최고성능 해제 \
powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61


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
Get-AppxPackage *xboxapp* | Remove-AppxPackage
Get-AppxPackage Microsoft.Microsoft3DViewer | Remove-AppxPackage
get-appxpackage *messaging* | remove-appxpackage
get-appxpackage *sway* | remove-appxpackage
get-appxpackage *commsphone* | remove-appxpackage
get-appxpackage *phone* | remove-appxpackage
get-appxpackage *communicationsapps* | remove-appxpackage
get-appxpackage *zunevideo* | remove-appxpackage
get-appxpackage *bingfinance* | remove-appxpackage
get-appxpackage *bingsports* | remove-appxpackage
get-appxpackage *bingnews* | remove-appxpackage
get-appxpackage *camera* | remove-appxpackage
get-appxpackage *maps* | remove-appxpackage
get-appxpackage *soundrecorder* | remove-appxpackage
get-appxpackage *xbox* | remove-appxpackage
get-appxpackage *wallet* | remove-appxpackage
get-appxpackage *connectivitystore* | remove-appxpackage
get-appxpackage *oneconnect* | remove-appxpackage
get-appxpackage *sticky* | remove-appxpackage
get-appxpackage *holographic* | remove-appxpackage
get-appxpackage -allusers Microsoft.549981C3F5F10 | Remove-AppxPackage


Set-MpPreference -DisableRealtimeMonitoring $true

Get-AppxPackage Microsoft.YourPhone -AllUsers | Remove-AppxPackage

Get-AppxPackage *zunemusic* | Remove-AppxPackage
Get-AppxPackage *soundrecorder* | Remove-AppxPackage
Get-AppxPackage *bingweather* | Remove-AppxPackage
Get-appxpackage *windowsphone* | remove-appxpackage
Get-AppxPackage Microsoft.YourPhone -AllUsers | Remove-AppxPackage
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


### Ubuntu 18.04
| Run | Command |
| -------- | -------- |
| 기본 편집기 변경 | sudo update-alternatives --config editor |
| 한글 설치 | ibus-setup |

#### python 필수 설치 목록

pip install tensorflow

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy pandas matplotlib seaborn beautifulsoup4 nltk scipy scikit-learn tqdm


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
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
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
