@echo off
rem The version for the Play Store: the version number one up, the app built
rem as an Android App Bundle (.aab) and signed with your upload key (it asks
rem for the key's password). Upload app\release\looot-*.aab in the Play Console.
cd /d "%~dp0.."
set JAVA_HOME=%USERPROFILE%\android-tools\jdk
set KEY=%USERPROFILE%\android-keys\looot-upload.jks
if not exist "%KEY%" (
  echo No upload key yet: run start\make_upload_key.bat first.
  pause
  exit /b
)
python tools\make_app.py --bump || goto :fail
cd app
call npx cap sync android || goto :fail
cd android
call gradlew.bat bundleRelease || goto :fail
cd ..
for /f %%v in ('python ..\tools\make_app.py --version') do set VER=%%v
mkdir release 2>nul
copy /y android\app\build\outputs\bundle\release\app-release.aab release\looot-%VER%.aab >nul
"%JAVA_HOME%\bin\jarsigner.exe" -keystore "%KEY%" release\looot-%VER%.aab upload || goto :fail
echo.
echo Done: app\release\looot-%VER%.aab - upload it in the Play Console.
echo Commit the new version number (app\android\app\build.gradle).
pause
exit /b
:fail
echo Something went wrong (see above).
pause
