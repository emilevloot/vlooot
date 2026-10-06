HOUSE TASKS ON THE NEXTION SCREEN
=================================

A list of house tasks on the 3.5" Nextion touchscreen (NX4832K035_011,
480 x 320). Tap a task when it's done: it turns green and its points go
to the total at the top. Tap it again to take it back. The menu (top
right) clears all ticks for a new day, or sets the points back to 0 after
they're spent on a reward. The screen keeps the ticks and the points in
its own memory, also when the power goes off. It only needs 5 V power:
no Arduino, no Wi-Fi.

The screen can't show web pages: it runs a small program made in Nextion
Editor. This folder has everything that goes into it.

FILES
  tasks.json          the tasks and their points (up to 12)
  make_screens.py     draws the pictures and writes nextion_code.txt
  screens\            the 6 pictures for the screen
  nextion_code.txt    which parts go where in Nextion Editor, with all
                      the code to copy
  preview.png         how it looks (not for the screen)

WHAT YOU NEED
  - Nextion Editor (free, Windows): https://nextion.tech/nextion-editor/
  - a microSD card, 32 GB or smaller, formatted FAT32
  - 5 V power, 0.5 A is plenty: a USB phone charger. Use the cable that
    came with the screen: the wires at +5V and GND (printed next to the
    white plug; usually red and black) go to the charger's + and -. If a
    small USB power board came with the screen, plug it into that. The
    other two wires (TX, RX) stay free.

BUILD IT IN NEXTION EDITOR (once, about 45 minutes)
  1. File > New, save as HouseTasks.HMI.
     Device: the Enhanced series, NX4832K035_011. There is an R
     (resistive touch) and a C (capacitive) version: take the one on your
     box (R if you don't know; if the screen later says the model doesn't
     match, take the other one and make the .tft again).
     Display: horizontal (480 x 320), character encoding iso-8859-1. OK.
  2. The font for the points: Tools > Font Generator. Height 32, Bold
     ticked, font Segoe UI (or Arial), encoding iso-8859-1. Type a name
     (points), click Generate font, save it, and answer Yes to adding it
     to the project. It is font 0.
  3. The pictures: in the Picture panel (bottom left) click + and add the
     6 files from screens\, in the order of their numbers (0 to 5). Check
     that 0_main.png got number 0, and so on.
  4. The pages: the Page panel (top right) has page0. Rename it to main.
     Add two pages and name them menu and confirm (in that order).
  5. Now open nextion_code.txt and work through it page by page:
     - the page itself: click an empty spot, set sta to image and pic to
       the picture number. Paste the page's code (main only) in the
       Preinitialize Event tab at the bottom.
     - every part: take it from the Toolbox (left): Number, Button or
       Dual-state button. In the Attribute panel (right) type its x, y, w,
       h, set sta to crop image and the picture numbers (picc, picc2,
       picc0, picc1) and delete the text in txt. Paste its code in the
       Touch Release Event tab.
     The names (n0, b0, bt0 ...) must be exactly as in nextion_code.txt.
     Quickest for the 12 tasks: finish bt0 completely, then Ctrl+C and
     Ctrl+V; the copy is called bt1. Change its x and y and the numbers in
     its code (the points and the bt number + memory place in the last
     two lines). Do the same for bt2, bt3 ...
  6. Try it on the computer: click Debug (top). A simulated screen opens;
     click the tasks with the mouse, try the menu.
     Compile (top) must say 0 errors.
  7. File > TFT file output: save HouseTasks.tft.

PUT IT ON THE SCREEN
  1. Copy HouseTasks.tft to the empty microSD card (the only file on it).
  2. Screen without power: put the card in its slot.
  3. Power on: the screen shows the update going to 100%, then
     "Update successed!" (about a minute).
  4. Power off, take the card out, power on: the tasks are there.
  The first start sets the memory to 0 points and no ticks.

CHANGE THE TASKS
  Edit tasks.json (up to 12 tasks; a name has to fit its tile: the script
  says when one is too long). Then in a terminal in the vlooot folder:
      pip install pillow          (only once)
      python house_tasks\make_screens.py
  In Nextion Editor, replace the 6 pictures (select one, Replace; keep the
  numbers). Changed points: change the two numbers in that task's code.
  More or fewer tasks: add or delete the Dual-state buttons (and the repo
  lines in the main page's Preinitialize Event) as in the new
  nextion_code.txt. Make the .tft again and put it on the screen.

HOW IT WORKS
  The pictures hold all the drawing. A Dual-state button with "crop
  image" shows its own rectangle of picture 0 (open) or picture 1 (done),
  so a tap swaps a white tile for a green one. The points are the only
  text the screen writes itself (n0, with font 0).
  The screen's memory (EEPROM, 1 KB) holds the points at place 0, the
  tick of task 1-12 at places 4, 8, ... 48, and 4271 at place 100 once it
  has been set up (wepo writes, repo reads).
  The screen's clock (the battery holder) isn't used: "New day" is a
  button in the menu.
