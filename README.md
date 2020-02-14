# Command-set

### Windows

#### 실행(Win + R)
| Run | Command |
| -------- | -------- |
| 부팅 정보 확인(멀티부팅) | msconfig | 
| 부팅 시 암호 입력 해제 | netplwiz |
| 빌드 버전 확인 | winver |
| 컴퓨터 관리 | compmgmt.msc |
| 디스크 정리 | cleanmgr | 

#### Powershell

최고성능 해제
powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61

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

| Load | Command | 
| -------- | -------- | 
| Download | pscp -r 계정명@ip주소:/home/계정명/디렉토리 ./ | 
| Upload | pscp 파일 계정명@ip주소:/home/계정명/디렉토리\[/파일명] | 
| | pscp -r 디렉토리 계정명@ip주소:/home/계정명/디렉토리/ |


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
