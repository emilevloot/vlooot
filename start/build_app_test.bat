@echo off
rem A test version of the Android app (APK) for your own phone: copy
rem app\android\app\build\outputs\apk\debug\app-debug.apk to the phone and open it.
cd /d "%~dp0.."
set JAVA_HOME=%USERPROFILE%\android-tools\jdk
python tools\make_app.py || goto :fail
cd app
call npx cap sync android || goto :fail
cd android
call gradlew.bat assembleDebug || goto :fail
echo.
echo Done: app\android\app\build\outputs\apk\debug\app-debug.apk
pause
exit /b
:fail
echo Something went wrong (see above).
pause
