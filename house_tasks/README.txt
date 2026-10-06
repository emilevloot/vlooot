ONZE HUISTAKEN ON THE NEXTION SCREEN
====================================

The house tasks app (Onze huistaken) on the 3.5" Nextion touchscreen
(NX4832K035_011, 480 x 320), in the app's colours and letters:

  - The two people at the top, with their points this week (big) and this
    month. A new week starts on Monday, a new month on the 1st.
  - 12 tasks. Tap one that's done, then who did it: Nadia, Vriend or Samen
    (both get the points). The task turns grey and shows when it comes
    back ("morgen", "over 4 d").
  - When a task is due it says "vandaag"; later "2 d te laat". As in the
    app, late tasks give bonus points: x1.5 when half their number of days
    late (at least 2 days, yellow bar), x2 when a whole round late (at
    least 3 days, red).
  - Tap a grey (done) task to undo the last tick (the points come off) or
    to tick it again.
  - Menu (top right): the savings goal (spaardoel) with how far you are,
    "Nieuw spaardoel beginnen" (tap twice) after the reward, the date and
    time, and "Klok gelijkzetten".

The screen keeps its own list, separate from the app on the phones. It
keeps the points and ticks in its memory, also without power. It needs
only 5 V power: no Arduino, no Wi-Fi.

PUT IT ON THE SCREEN (about 15 minutes)
  Huistaken.HMI is the finished Nextion Editor project: pictures, code and
  settings are all in it. Nothing to build, type or paste.
  1. Install Nextion Editor (free, Windows):
     https://nextion.tech/nextion-editor/
  2. Get Huistaken.HMI from this folder (on GitHub: click the file, then
     "Download raw file").
  3. Nextion Editor: File > Open, choose Huistaken.HMI. If it asks to
     convert the project to its own version, say yes.
  4. Optional: click Debug (top) to try it on the computer first, with
     the mouse. (If it opens the clock page: set a date, tap Opslaan.)
  5. File > TFT file output, then Output. It compiles first: the bottom
     must say 0 errors. Save Huistaken.tft.
  6. Copy Huistaken.tft to an empty microSD card (FAT32, 32 GB or
     smaller; the only file on it). Screen without power: put the card in
     its slot. Power on: the screen shows the update going to 100%, then
     "Update successed!" (about a minute). Power off, take the card out,
     power on.
  7. The first time it opens the clock page: set the date and time with
     - and +, then Opslaan. The task list starts with 0 points and every
     task "nooit gedaan".

  If something isn't right, send me the message (a photo is fine):
  - Nextion Editor can't open the file, or shows errors at step 5.
  - The screen says the model doesn't match at step 6. (The project is
    for NX4832K035_011; R and C screens of that size use the same file.)
  - Taps land in the wrong place: keep a finger on the screen while you
    power it on, and tap the crosses it shows (resistive screens only).

WHAT YOU NEED
  - a Windows computer for Nextion Editor (only for step 3-5)
  - a microSD card, 32 GB or smaller
  - 5 V power, 0.5 A is plenty: a USB phone charger. Use the cable that
    came with the screen: the wires at +5V and GND (printed next to the
    white plug; usually red and black) go to the charger's + and -. If a
    small USB power board came with the screen, plug it into that. The
    other two wires (TX, RX) stay free.
  - a CR1220 coin battery for the round holder: it keeps the clock going
    when the power is off. Without it the screen asks for the date and
    time again after every power cut (the points and ticks stay anyway).
    If the date in the menu is ever wrong: Menu > Klok gelijkzetten.

CHANGE THE TASKS, NAMES OR GOAL
  Edit tasks.json:
    people       the two names
    goal         the reward and how many points it needs
    tasks        up to 12: name (has to fit its tile: the script says when
                 one is too long), points, every (comes back after this
                 many days: 1 = every day, 7 = every week)
    fresh_start  raise it by 1 to start the screen at 0 points and no
                 ticks at its next start
  Then in a terminal in the vlooot folder:
      pip install pillow          (only once)
      python house_tasks\make_screens.py
      python house_tasks\check.py
  and do steps 3, 5 and 6 again with the new Huistaken.HMI. (Or send me
  the list and I'll do it.)
  The screen remembers ticks by place (1st task, 2nd task ...): moving
  tasks around moves their ticks, so raise fresh_start when you do.

FILES
  tasks.json        the two names, the goal and the tasks
  make_screens.py   makes Huistaken.HMI: draws the pictures and writes
                    the code
  Huistaken.HMI     the Nextion Editor project
  nextion_hmi.py    writes (and reads back) Nextion Editor's .HMI files
  check.py          tests Huistaken.HMI on the computer, makes preview.png
  nextion_sim.py    the small Nextion that check.py runs the code in
  nextion_code.txt  all the code in Huistaken.HMI, to read
  screens\          the pictures in Huistaken.HMI, to look at
  preview.png       how it looks (drawn by check.py's simulated screen)
  fonts\            the app's letters (free fonts, licences included)

THE CHECK ON THE COMPUTER
  check.py reads Huistaken.HMI the way Nextion Editor does (every
  checksum), runs its code in a small Nextion (nextion_sim.py) with
  Nextion's rules (no spaces or brackets in a line, sums from left to
  right, 4-byte numbers, the 1 KB memory, the clock, touch), sets the
  clock with the buttons, then plays 400 days of random ticks, undos, new
  goals and power cuts. After every step it compares every score, every
  task and every tile on the screen with the app's rules. It also checks
  the day counting for every day from 2025 to 2099, and that every text
  fits its space.
  What it can't check: Nextion Editor itself (opening and compiling the
  file), and the real screen's speed. So far no Nextion Editor was
  available here: step 3 and 5 are the first real test.

HOW IT WORKS
  One page and one timer. Every 50 ms the timer looks whether something
  changed and draws only that: whole screens with pic (8 pictures), a
  done or late tile with picq, and every number and word with xpic from a
  strip of letters (picture 8), so the project needs no fonts. Once a
  second it reads the clock (rtc0-rtc6) and counts the day number; a new
  day redraws the list. A tap only changes numbers (which screen, which
  task); the next tick draws it.
  Every task has 7 invisible Variables on the page (vl0, vw0, vp0, vv0,
  vd0, ve0, vq0 for task 1 ...): the day it was done, who, the points,
  the day before (for undo), the goal's number, and its two settings
  (every how many days, points). The code reaches them by number with
  b[...], so one loop handles all tasks.
  The screen's memory (EEPROM, 1 KB) holds:
     0         4270 + fresh_start once it is set up
     4, 8      points this week (Nadia, Vriend)
    12, 16     points this month
    20, 24     the Monday of that week, the 1st of that month
    28         points for the goal (Samen counts for both, as in the app)
    32         the goal's number (+1 for every new goal)
    64 + 4*n   the day task n was last done
  Undo works until the power goes off (who did what is kept in the
  Variables, not in the memory).

  The .HMI file format and its checksums come from Newmatik's research
  (github.com/newmatik/nextion-hmi-writer, MIT licence). The fixed bytes
  for this screen model were taken from a real NX4832K035_011 project
  (WPSD_Nextion by the M17 Project). Before writing this project,
  nextion_hmi.py was tried on 59 real pages from three projects made with
  Nextion Editor 1.65 and 1.68: it wrote every one byte for byte the same.
