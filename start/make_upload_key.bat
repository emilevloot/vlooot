@echo off
rem The upload key for the Play Store: made once, by you (it asks for a password
rem and your name). Google checks every new version against it. Keep the file
rem AND the password safe (a backup outside this pc): without them you need
rem Google support to set a new key.
set KEYS=%USERPROFILE%\android-keys
if exist "%KEYS%\looot-upload.jks" (
  echo There is already a key: %KEYS%\looot-upload.jks
  pause
  exit /b
)
mkdir "%KEYS%" 2>nul
"%USERPROFILE%\android-tools\jdk\bin\keytool.exe" -genkeypair -v -keystore "%KEYS%\looot-upload.jks" -alias upload -keyalg RSA -keysize 2048 -validity 10000
echo.
echo The key is in %KEYS%\looot-upload.jks - make a backup of it now.
pause
