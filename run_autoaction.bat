@echo off
call venv\Scripts\activate.bat

call :filedialog file
if "%file%"=="" (
    echo No file selected. Exiting...
    exit /b
)
if /i not "%file:~-4%"==".txt" (
    echo Selected file is not a .txt file. Exiting...
    exit /b
)
echo Selected file is: "%file%"
timeout /t 5 /nobreak

python autoaction.py "%file%"
pause
exit /b

:filedialog :: &file
setlocal 
set dialog="about:<input type=file id=FILE><script>FILE.click();new ActiveXObject
set dialog=%dialog%('Scripting.FileSystemObject').GetStandardStream(1).WriteLine(FILE.value);
set dialog=%dialog%close();resizeTo(0,0);</script>"
for /f "tokens=* delims=" %%p in ('mshta.exe %dialog%') do set "file=%%p"
endlocal & set %1=%file%
