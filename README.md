# Command-set

#### conda install < requirements.txt in windows

| Type | Command |
| -------- | -------- |
| manually | for /f %i in (requirements.txt) do conda install --yes %i |
| BAT script | for /f %%i in (requirements.txt) do conda install --yes %%i |

#### pscp usage

| Load | Command | 
| -------- | -------- | 
| Download | pscp -r 계정명@ip주소:/home/계정명/디렉토리 ./ | 
| Upload | pscp 파일 계정명@ip주소:/home/계정명/디렉토리\[/파일명] | 
| | pscp -r 디렉토리 계정명@ip주소:/home/계정명/디렉토리/ |
