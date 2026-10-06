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
    "Nieuw spaardoel beginnen" (tap twice) after the reward, and the clock.

The screen keeps its own list, separate from the app on the phones. It
keeps the points and ticks in its memory, also without power. It needs
only 5 V power: no Arduino, no Wi-Fi.

FILES
  tasks.json        the two names, the goal and the tasks (up to 12)
  make_screens.py   draws the pictures and writes nextion_code.txt
  screens\          the 15 pictures for the screen
  nextion_code.txt  all the code to paste into Nextion Editor
  check.py          tests that code on the computer and makes preview.png
  nextion_sim.py    the small Nextion that check.py uses
  preview.png       how it looks (made by check.py, not for the screen)
  fonts\            the app's letters (free fonts, licences included)

WHAT YOU NEED
  - Nextion Editor (free, Windows): https://nextion.tech/nextion-editor/
  - a microSD card, 32 GB or smaller, formatted FAT32
  - 5 V power, 0.5 A is plenty: a USB phone charger. Use the cable that
    came with the screen: the wires at +5V and GND (printed next to the
    white plug; usually red and black) go to the charger's + and -. If a
    small USB power board came with the screen, plug it into that. The
    other two wires (TX, RX) stay free.
  - a CR1220 coin battery for the round holder: it keeps the clock going
    when the power is off. Without it the screen asks for the date and
    time again after every power cut (the points and ticks stay anyway).

BUILD IT IN NEXTION EDITOR (once, about an hour)
  1. The letters: double-click fonts\AtkinsonHyperlegible-Regular.ttf
     and click Install; the same for fonts\BricolageGrotesque-ExtraBold.ttf.
     (Start Nextion Editor after this.)
  2. File > New, save as Huistaken.HMI.
     Device: the Enhanced series, NX4832K035_011. There is an R
     (resistive touch) and a C (capacitive) version: take the one on your
     box (R if you don't know; if the screen later says the model doesn't
     match, take the other one and make the .tft again).
     Display: horizontal (480 x 320), character encoding iso-8859-1. OK.
  3. Two fonts, in this order (Tools > Font Generator; after Generate
     font, save it and answer Yes to adding it to the project):
       font 0: Atkinson Hyperlegible, height 16, Bold not ticked
       font 1: Bricolage Grotesque ExtraBold, height 32
     Encoding iso-8859-1 for both. (Not in the list? Use Arial and Arial
     Black.)
  4. The pictures: in the Picture panel (bottom left) click + and select
     all 15 files in screens\ at once. Check that 00_taken.png got number
     0 and 14_klok_in.png number 14.
  5. Program.s (the tab at the top of the editing area, next to the
     page): replace everything in it with the PROGRAM.S block from
     nextion_code.txt.
  6. The pages: the Page panel (top right) has page0. Rename it to taken.
     Add four pages and name them wie, terug, menu, klok (in that order).
  7. Page taken: from the Toolbox (left) add three Variables and a Timer
     (they're invisible, put them anywhere):
       va0, va1, va2   sta = String, txt_maxl = 60, vscope = global
       tm0             tim = 60000, en = 1
  8. Every page, one by one, as in nextion_code.txt:
     - click an empty spot of the page: sta = image, pic = the number
       given there.
     - the Event panel (bottom) has tabs: Preinitialize Event,
       Postinitialize Event, Touch Press Event, Touch Release Event.
       Paste each block from nextion_code.txt into its tab: everything
       between the ---- lines. The Timer Event block goes in tm0's tab
       (click tm0 first).
  9. Compile (top): it must say 0 errors. If a line gives an error, send
     me the message with the line.
  10. Debug (top) opens a simulated screen: click the tasks with the mouse.
  11. File > TFT file output: save Huistaken.tft.

PUT IT ON THE SCREEN
  1. Copy Huistaken.tft to the empty microSD card (the only file on it).
  2. Screen without power: put the card in its slot.
  3. Power on: the screen shows the update going to 100%, then
     "Update successed!" (about a minute).
  4. Power off, take the card out, power on.
  5. The first time it opens the clock page: set the date and time with
     - and +, then Opslaan. The task list starts with 0 points and every
     task "nooit gedaan".
  If taps land in the wrong place: keep a finger on the screen while you
  power it on, and tap the crosses it shows (resistive screens only).

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
  In Nextion Editor: delete all pictures and add the 15 new ones (step 4),
  paste all the code blocks again (step 5 and 8), then make the .tft
  again and put it on the screen.
  The screen remembers ticks by place (1st task, 2nd task ...): moving
  tasks around moves their ticks, so raise fresh_start when you do.

THE CHECK ON THE COMPUTER
  check.py runs the code from nextion_code.txt in a small Nextion
  (nextion_sim.py) with Nextion's rules: no spaces or brackets in a line,
  sums worked out from left to right, 4-byte numbers, the 1 KB memory and
  the clock. It sets the clock with the buttons, then plays 400 days of
  random ticks, undos, new goals and power cuts, and after every step
  compares every score and task with the app's rules. It also checks the
  day counting for every day from 2025 to 2099, and that every text fits
  its space. What it can't check: whether Nextion Editor accepts every
  line exactly as written, the real fonts, and how fast the screen is.

HOW IT WORKS
  The pictures hold all the drawing. picq copies a tile from picture 1, 2
  or 3 over the open tile in picture 0; xstr writes the points and "over
  4 d" on top. Taps are found from where the finger is (tch0, tch1), so
  the pages need no buttons: only the code.
  The day number (gdag) is counted from the clock (rtc0-rtc6). A task is
  due when its last day + its number of days is today; earlier it's done.
  The screen's memory (EEPROM, 1 KB) holds:
     0  4271 + fresh_start once it is set up
     4, 8      points this week (Nadia, Vriend)
    12, 16     points this month
    20, 24     the Monday of that week, the 1st of that month
    28         points for the goal (Samen counts for both, as in the app)
    32         the goal's number (+1 for every new goal)
    64 + 20*n  task n: last day, who, points, the day before, goal number
