LOOOT AS AN ANDROID APP
=======================

The app is the same game as the web site (../index.html), packed with
Capacitor: a small Android program that shows the page in a WebView, with
every file inside the app, so it plays without internet from the first
start. Python (Pyodide), numpy, the two networks, the fonts, PeerJS and the
QR code: about 31 MB (app/www), 17 MB as a Play Store bundle.

  App ID       io.github.emilevloot.looot   (fixed for ever once on the Play Store)
  Target       Android 16 (API 36), the Play Store's rule for new apps since
               31 August 2026; runs on Android 7 (API 24) and newer
  Version      app/android/app/build.gradle (versionCode, versionName);
               start\build_app_release.bat puts it one up

What is different in the app (index.html, window.LOOOT_APP, set by
tools/make_app.py): no service worker, no "Install app" button, no links to
the dashboards and the editor (not in the app), invite links that point to
the web site (friends can join there without the app), and the Android back
button (in a game: the menu; on the menu: close the app).


THE TOOLS (installed in C:\Users\emile\android-tools, no admin rights needed)
  jdk\                 Java 21 (Temurin)
  sdk\                 the Android SDK: platform 36, build-tools 36,
                       platform-tools (adb), the emulator and an Android 16 image
Delete that folder to remove them.


BUILDING
  start\build_app_test.bat     a test version (APK) for your own phone:
                               app\android\app\build\outputs\apk\debug\app-debug.apk
                               Put it on the phone (cable, Google Drive, mail to
                               yourself), open it and allow "install unknown apps".
  start\make_upload_key.bat    ONCE: your upload key (asks for a password and your
                               name). Keep C:\Users\emile\android-keys\looot-upload.jks
                               and the password safe, with a backup elsewhere.
  start\build_app_release.bat  the version for the Play Store: version number one
                               up, signed with your key (asks for the password):
                               app\release\looot-<version>.aab. Commit build.gradle.

By hand, step by step:
  python tools\make_app.py           the game's files into app\www
  cd app && npx cap sync android     ... into the Android project
  cd android && gradlew assembleDebug   (or bundleRelease)
  (JAVA_HOME=C:\Users\emile\android-tools\jdk)
Testing without a phone: the Android emulator (a Pixel 7 with Android 16,
"looot_test"):
  set ANDROID_AVD_HOME=C:\Users\emile\android-tools\avd
  C:\Users\emile\android-tools\sdk\emulator\emulator.exe -avd looot_test
  C:\Users\emile\android-tools\sdk\platform-tools\adb.exe install -r app-debug.apk
In a test version Chrome can look inside the app: chrome://inspect.
After changing the icon: python tools\make_icons.py (also the Android icons
and app\store\icon-512.png).


THE PLAY STORE, STEP BY STEP
 0. The name and the game. Looot is a published game by Gigamic; the name is
    theirs. Before the app goes PUBLIC, ask Gigamic for permission, or give
    the app its own name and theme. A closed test with friends (step 6) is
    something else - but Google can also take an app down after a complaint.
 1. A developer account: play.google.com/console - $25 once, and an identity
    check. (You do this yourself.)
    A new personal account must first run a CLOSED TEST with at least 12
    testers, for 14 days in a row, before the app may go public.
 2. start\make_upload_key.bat (once) - and make a backup of the key.
 3. Test on your own phone: start\build_app_test.bat.
 4. In the Play Console: Create app - name Looot, language English (United
    States) (add Dutch later as a translation), Game, Free.
 5. "App content" (the forms; see the answers below), "Store listing" (the
    texts and pictures below).
 6. Testing > Closed testing: a track, the testers' e-mail addresses (a Google
    group or a list), upload app\release\looot-<version>.aab (made by
    start\build_app_release.bat). Accept "Play App Signing" (Google keeps the
    real signing key; you sign uploads with your upload key).
 7. After 14 days with 12+ testers: "Apply for production". Then the app can
    go public (review: a few hours to a few days).
 Every new version: start\build_app_release.bat, upload the .aab, commit.


THE STORE LISTING (app\store)
  App icon           icon-512.png (512 x 512)
  Feature graphic    feature-graphic.png (1024 x 500; feature.html, a picture
                     of it: chrome --headless --screenshot --window-size=1024,500)
  Phone screenshots  phone\01_menu.png ... 05_lobby.png (1080 x 1920)
  Category           Game > Board
  Contact            your e-mail (shown in the store; required)
  Privacy policy     https://emilevloot.github.io/vlooot/privacy.html

  App name (30)      Looot
  Short (80)         Viking board game with a neural network opponent and a coach for every move
  Full description:

    Sail out, claim the land and fill your longships! Looot is a Viking board
    game for 1 to 4 players: place your Vikings on the landscape, gather wood,
    sheep and gold, capture houses, watchtowers and castles, and surround
    your longships to score.

    - Play against the computer or against a neural network trained on
      millions of games.
    - A coach like a chess trainer: every move judged from Brilliant to
      Blunder, how many points it cost, and WHY the better move is better.
    - Game review: your accuracy, the win chances through the game, and every
      mistake on the board.
    - Pass and play: 2 to 4 people on one screen.
    - Play with friends online: make a lobby, send the link or the QR code.
      Friends can join from the app or from the web site - no account needed.
    - Plays offline. No ads, no accounts, no tracking.

  Nederlands (vertaling, kort 80):
    Vikingbordspel met een neuraal netwerk als tegenstander en een coach per zet
  (Full description: translate the English one when adding Dutch.)


THE FORMS IN "APP CONTENT" (my reading of Google's rules - check them yourself)
  Privacy policy     the URL above
  Ads                No ads
  App access         All functionality available without special access
  Content rating     (IARC questionnaire) Game; no violence, no blood, no
                     gambling, no bad language, no drugs; users can
                     interact: yes (online games share the name you choose
                     and your moves; there is no chat); no location; no
                     purchases. Expect "PEGI 3" / "Everyone".
  Target audience    13 and older (younger ages bring the Families rules,
                     which ask more of apps that connect to other services)
  Data safety        Collects data: No. Shares data: No.
                     Why: the maker receives nothing. The player's name goes
                     to the other players only when the player starts or
                     joins an online game (a user-initiated transfer, which
                     Google does not count as sharing); PeerJS sees the game
                     code and the internet address only to connect the
                     players. Data is encrypted in transit (WebRTC, HTTPS).
                     There are no accounts, so nothing to delete.
  News / health / financial / government features: none.
