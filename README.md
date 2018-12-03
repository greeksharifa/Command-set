# Command-set

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

#### ssh / sftp for windows 10 (build >= 1709)

| Type | Command | 
| -------- | -------- | 
| SSH | ssh username@address | 
| SFTP | sftp username@address | 
