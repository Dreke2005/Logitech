# PaperBridge

Start the workbench from the project directory.

    uv run paper_workbench.py

Choose an app icon on page 1 and press the key for what you're about to write, such as **Code**, **Plot**, or **Equation**. If the camera is off, that key starts it; **Camera**, the top-left key on every app page, turns it on and off by hand. The key captures a reference frame and starts tracking. The green box shows the detected paper or the writing area you selected by dragging in the preview. The yellow line shows hand motion. A magenta crosshair marks the pen tip, estimated from the thumb–index grip along the pen axis and snapped to the visible nib when the shaft can be seen (filled center). Blue lines are the registered pen-down strokes: a tip sample counts only when new ink appears under it, so hover moves are left out. Keep the paper still while you write.

To finish, draw a small closed circle after the text, apart from it and a bit larger than your letters, and hold still briefly. The orange ring confirms it, and generation starts without a key press. The circle itself is not transcribed. For **Diagram** and **Copy** a circle is part of the content, so finish those by moving your hand away or pressing the key again. That also works for every other key.

Reading uses the pen trace first. The model gets only the recorded pen-down strokes, drawn black on white, turned upright for the writer, and scaled to about 80 px per line of text. The key you pressed tells it what to expect, such as Python code, a MATLAB plot expression, a LaTeX table, a branch name, or git diagram commands. Each answer is checked for that key: an equation must parse, code must be balanced, and table rows must have the same number of cells. A failed check is retried once with the reason. Before the trace is read, it is registered on the new ink in the final frame: the whole trace is shifted onto the ink, and any move that left no ink under it (a hover hop to the next letter) is cut out, so glyphs stay separate. An accepted trace reading is then compared with a photo crop of the same writing. The photo confirms it, corrects a few symbols, or, when it reads something quite different, leaves the trace reading in place with a note to compare with your page. Only when the trace is too sparse or the retry also fails does the model read the photo on its own. That crop covers only the new writing, plus half a line height. New ink counts only as thin, dark marks on the paper near your strokes. Hands, the forearm, shadows, exposure changes, and objects you put down do not count. The status line shows whether the result came from the trace, the trace checked against the photo, or the photo alone.

Frames and traces stay in memory. **Capture page** freezes a frame for manual selection and a one-press conversion. **Resume live** returns to the camera. Pressing **Camera** again stops the stream.

| Output | What the workbench produces |
| --- | --- |
| LaTeX | **Equation** opens a TeX draft locally. **Table** turns a drawn table into a booktabs table (see [Tables](#tables)). **Report** creates and opens a report skeleton. **Browse** selects and opens an existing project for **Paste**, **Table**, and **Overleaf**. |
| MATLAB | **Code** generates and opens a `.m` file. **Project** creates a folder of MATLAB files and opens its entry file in the running MATLAB, or starts MATLAB in that folder. **Plot** creates figures from supported expressions. |
| VS Code | **Code** generates and opens a source file. **Project** opens the scaffold folder in VS Code. **Language** opens the language icons. |
| ChatGPT | **Brief** drafts a prompt from notes and sketches. **Paste** stages copied text or code in a review prompt. **Open ChatGPT** copies the draft and opens the browser. Paste and submit it when ready. |
| KiCad | **Schematic** drafts an editable legacy `.sch` for resistors, capacitors, inductors, and diodes. Mark component references, values, and net labels on the sketch. |
| LTspice | **Schematic** reads a drawn circuit into an `.asc` schematic and opens it in LTspice. **Simulate** runs it. See [LTspice](#ltspice). |
| ChatGPT flowchart | **Flow** turns drawn boxes and arrows into a Mermaid flowchart in a Markdown file, opens it in VS Code, and copies it for pasting. |
| Plan | **Gantt** turns a written pipeline of tasks, durations, and milestones into a Gantt chart. See [Plan](#plan). |
| Version control | Commit, push, pull, branch, and GitHub or GitLab actions for the project you're working in. See [Version control](#version-control). |

Compare **Read from paper** with your page and check signs, subscripts, variable names, and assumptions. You can edit the transcription before clicking **Open VS Code Chat** in the workbench. That action passes a local Markdown file with the reviewed transcription to VS Code Chat in Ask mode. It uses the chat provider configured in VS Code. Open the generated draft in the editor to continue with inline suggestions or your installed Codex extension. [VS Code chat CLI](https://code.visualstudio.com/docs/configure/command-line#_start-chat-from-the-command-line)

## Work without looking at the workbench

You don't need the workbench window. Started from the keypad, it stays minimized.

- **One key press.** An action key starts the camera when needed, records, reads, generates, and opens the result in the app: VS Code, MATLAB, LTspice, or the browser.
- **Status on the key.** The key you pressed shows a bar along its bottom edge: red while recording, amber while reading and generating, green when done, and red at top and bottom when it failed.
- **Pop-up results.** While the window is minimized or in the background, each status message appears for a few seconds at the bottom right of the screen.
- **Folders are asked once.** The Git repository, the LaTeX project, and the VS Code language are remembered in `%LOCALAPPDATA%\LogiInsightPaper\settings.json`. Use **Browse** to change them.
- **Context from your project.** The reader is told what the pressed key expects, and also gets names from the selected project: functions, classes, and variables for code, macros and labels for LaTeX, and branch names for Git. That narrows what an unclear word can be.
- **Inline AI gets the handwriting.** A generated code file starts with your transcription as a comment, so Copilot or another inline assistant in the editor continues from what you wrote.
- **Math is checked.** For equations, each step of a derivation is checked with sympy, and the first step that doesn't follow is named in the notes and the pop-up. A line ending in `= ?` gets its result. A definition such as `y = x^2 + 1` is left alone.

## Copy and paste between apps

To copy something on paper, draw a **dashed oval** around it, with short gaps and space between the oval and the writing. Lift your hand and press **Copy**. If the oval is missed, freeze the page and drag around it. Copy also takes the current draft, or the text selected in it. PaperBridge records *what* it copied (equation, code, table, plot, diagram, todo, or text) in `%LOCALAPPDATA%\LogiInsightPaper\clipboard.json`, with the source app and any rendered figure. The plain text also goes to the Windows clipboard. A copied plot is rendered right away.

Open another app's page and press **Paste**. The result depends on what was copied and where you paste it. Generated files open in the target app, and the status line names the copied kind and what Paste made of it. Paste uses this typed copy only while the Windows clipboard still holds the copied text. Anything you copy elsewhere afterward is pasted as before.

| Copied | LaTeX | VS Code | MATLAB | ChatGPT | Git |
| --- | --- | --- | --- | --- | --- |
| Equation | `equation` environment file plus `\input` | Function in the selected language | Anonymous function plus `fplot` script | Explain-and-check prompt | Commit message |
| Code | `lstlisting` file | New file in the open project, or a draft | `.m` file | Review prompt | Commit message |
| Table | Booktabs table in `tables/` | CSV file | `table()` or `array2table` script | Explain prompt | — |
| Plot | The equation or code it was made from, then the figure, inserted in the text; the file goes to `figures/` | Matplotlib script | Plot script | Explain prompt | — |
| Diagram | Figure if rendered, otherwise `verbatim` | Markdown note | Comments | Explain prompt | Runs it as a [diagram](#draw-a-branch-diagram) |
| Todo | `itemize` list | TODO comments | TODO comments | Planning prompt | Adds the open items to `TODO.md` and commits only that file |
| Text | Paragraph | Markdown note | Comments | Review prompt | Commit message for current changes |

In LaTeX, fragments go into the project chosen with **Browse**. PaperBridge adds the `\input` line before `\end{document}` of `main.tex` or the only complete document, and loads `booktabs` or `listings` there if needed. Without a clear root file, the `\input` line goes to the clipboard. Without a project, Paste asks for one and saves a standalone draft if you cancel. ChatGPT pastes are staged in the workbench; press **ChatGPT** to copy the prompt and open the chat.

### Tables

On the LaTeX page, press **Table** and draw a grid. Separate the columns with vertical lines and write one row per line. A first row of labels becomes the header. PaperBridge escapes LaTeX special characters and right-aligns numeric columns. It saves a booktabs `tabular` in a `table` float as `tables/table_<n>.tex` in the selected project, with `\input` and `\usepackage{booktabs}` added as for equations. It then opens the root file. Without a selected project, it opens a standalone draft instead.

## Transfer a graph or code

The **Overleaf** icon packages the selected local project as a ZIP and opens the Overleaf dashboard. Choose **New project → Existing project (.zip)** to upload it. Generated working files are saved under `%LOCALAPPDATA%\LogiInsightPaper\drafts`. [Overleaf project upload](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)

1. In MATLAB or VS Code, start the camera, press **Plot**, then write one equation. If the equation is already on paper, use **Capture page** and select it before pressing **Plot**. You can also use the **Paste** key to stage an equation copied from an editor. The plotter accepts scalar functions of `x`, basic math functions, and simple `plot(x, y)` snippets. It uses the written range when available and defaults to −5 through 5. Use **Open plot preview** in the workbench to view the PNG.
2. When the plot is ready, use the keypad **Copy** icon. It copies the plot together with the equation it was made from. Open the LaTeX menu and press **Paste**. The figure file is copied into `figures/`, and one fragment with the equation followed by the figure is added to the document through `\input`, with `graphicx` loaded if needed. A plot made from code gets a code listing instead of the equation, and a figure without either is inserted on its own. The LaTeX project is asked for once and remembered.
3. For a more complex plot, generate it in MATLAB, VS Code, or PyCharm and export a PDF or PNG. Copy the file in Explorer and use **Paste** in the LaTeX menu. The same action accepts an image from the Windows clipboard. For code review, copy text in the editor, then use **Paste** in the ChatGPT menu to stage it.

Without the LogiInsight camera, the workbench uses the first ordinary webcam instead, such as a laptop's built-in camera, and says so in the status line. It runs the same capture worker, hand tracking, and writing session, at whatever resolution the camera delivers, so a demo on a laptop camera exercises the same code. Tilt the lid so the camera sees the paper, or hold the sheet up to it; writing that appears upside down is turned upright from the direction of your hand. To check a camera before a demo, run `uv run paper_camera_check.py` and write a few letters: it reports resolution, frame rate, whether your hand and pen grip are seen, how many pen-tip samples counted as pen-down, and the writing region it found.

The camera path uses 1080p MJPG via the LogiInsight SDK's OpenCV capture helper. The PC vision model reads the page from RGB frames. The camera's SFTP AI JSON provides face detections and landmarks. [Logitech camera SDK](https://raw.githubusercontent.com/Logitech/hackathons/master/docs/logiinsight/API.md)

## LTspice

Open **LTspice** on page 1. Draw the circuit with a reference and value on every part (`V1 5`, `R1 1k`, `C1 100n`, `L1 1m`, `D1`, `I1 1m`), label the nets you care about (`in`, `out`), and draw a ground symbol or write `GND`. A written command such as `.tran 10m` or `.ac dec 20 1 1Meg` is used as the simulation command. Without one, PaperBridge picks `.tran 10m` for SINE or PULSE sources, an AC sweep for `AC` sources, and `.op` otherwise.

- **Schematic** captures the page, reads the parts and their nets, and opens the generated `.asc` in LTspice. Each part is placed on a grid with a net flag on both pins, so parts that share a label are connected. The schematic is valid for any topology, but it isn't laid out like your drawing; move parts and draw wires in LTspice if you want that. The netlist that was read is shown under the draft for checking.
- **Simulate** runs the schematic in batch mode, reports problems from the log in the status line, and on success opens it in LTspice with the simulation run, so you can probe waveforms.
- **Paste** stages copied SPICE text as a `.cir` netlist. **Copy** takes the draft, for example to paste it into ChatGPT for review.

The local 2B vision model reads references, values, and commands well, and simple wiring usually. Check the netlist for larger circuits: a note warns when every part was read between the same two nets. A net that reaches only one part is read again once and flagged if it stays that way.

## Version control

Version control always acts on **the project you're working in**, so you never pick a folder first:

- From VS Code it's the folder of the VS Code window you used last. With several projects open, the one in front wins.
- From LaTeX it's the selected LaTeX project. From MATLAB, LTspice, and the other apps it's the folder of the file PaperBridge last opened there, and otherwise the front VS Code window.
- **Browse** picks another repository by hand. That choice holds until you switch to a different app page.

Each status message names the repository it used. There are two ways in: the **Git** icon on page 1, or the keypad's page buttons from any app. Page 2 is version control and page 3 holds the GitHub and GitLab actions, and both leave the app page as it is.

| Page | Keys |
| --- | --- |
| 2: version control | Camera, Commit, Push / Pull, Branch, Diagram / Host, Browse, Home |
| 3: host | Camera, Issue, Request / Publish, Open, Repo / Paste, Back, Home |

- **Commit** opens a terminal in the repository. It lists the staged and the unstaged files and asks whether to stage everything: the default is yes when nothing is staged yet, and no when you staged files yourself. Then it asks for the message, commits, and offers to push. An empty message cancels.
- **Push** and **Pull** work on the current branch. Push never forces, and Pull only fast-forwards.
- **Branch**: write a short name such as *add login page*. It becomes `add-login-page` and is checked out.
- **Diagram** runs a drawn branch graph; see below.
- **Issue** and **Request**: write a title, and optionally a description under it. PaperBridge reads whether the remote is GitHub or GitLab (self-hosted too) and opens the issue, pull request, or merge request in your browser with everything filled in. You submit it there, so no access token is needed. Request pushes the branch first and targets the default branch.
- **Publish**, for a repository without a remote: the first press opens the new-repository page. Create it, copy its URL, and press **Publish** again to add it as `origin` and push.
- **Open** shows the repository page. **Repo** turns a written project idea into a new repository with a scaffold and an initial commit. **Paste** commits a copied note: text becomes the commit message, and a todo list is added to `TODO.md`.

### Draw a branch diagram

Press **Diagram** and draw the history you want. Draw each branch as a horizontal line with its name at the left. A fork starts a new branch and a join is a merge. Circle a node and write the message next to it for a commit. Circle a branch name to push it. Press **Diagram** again or lift your hand when done. The reader turns the drawing into one command per line, which you can check in **Read from paper**:

    branch feature/login from main
    commit feature/login "Add login form"
    merge feature/login into main
    push main
    tag v1.0 on main

PaperBridge shows the numbered plan in the draft panel and runs it right away; pressing the key is the confirmation. Steps run in order on the selected repository:

- A new branch is created from its parent with `git switch -c`. An unknown parent is first created from the branch you're on. Existing branches are reused.
- A commit takes your current changes, the staged ones if any. When nothing is left to commit, it makes an empty commit so the drawn history still appears, and the status line says so.
- A merge uses `git merge --no-ff`. On a conflict it's aborted, nothing changes, and the run stops there.
- A push uses the normal safe push. It's skipped with a note when the repository has no remote.

PaperBridge never forces, deletes branches, or rewrites history. It reports lines it couldn't read and returns to your original branch at the end.

## Plan

Open **Plan** on page 1 and press **Gantt**. Write your pipeline top to bottom while you brainstorm: one task per line with a duration (`2w`, `10d`, `1m`), an arrow when a task follows something other than the line above, and a star or the word *milestone* for milestones. Press **Gantt** again or lift your hand when done.

PaperBridge schedules the tasks from today, each after the one above unless an arrow says otherwise, and opens two results: the rendered chart (PNG and PDF) and a Markdown file with the same plan as a Mermaid `gantt` block, which VS Code and GitHub render. **Copy** takes the chart; **Paste** in LaTeX then inserts it as a figure. Lines that couldn't be read as a task or milestone are listed in the notes.

## Run the local model

Install Ollama if it is unavailable and pull the vision model once.

    ollama pull qwen3-vl:2b-instruct

The workbench sends the captured still image to the Ollama service on this computer. It also detects the portable model installation used for testing under **%LOCALAPPDATA%\LogiInsightTools**.

## Configure the MX Keypad

For a new installation, use the .NET 10 SDK with Logi Options+ and Logi Plugin Service 6.4 or newer.

    uv sync --locked
    .\keypad-plugin\setup.ps1

`setup.ps1` builds the plugin and registers it. If the actions are missing, use the Options+ setting to restart Logi Plugin Service. In **All Applications**, the **PaperBridge** profile has three pages. Assign **LogiInsight Paper → Paper key 1–9** to page 1, **Paper key 10–18** to page 2, and **Paper key 19–27** to page 3, in order. Page 1 redraws itself: it shows the app icons (VS Code, MATLAB, LaTeX, ChatGPT, KiCad, LTspice, Git, Plan), and pressing one shows that app's keys. **Home** returns to the apps. Pages 2 and 3 always show version control and its host actions. When the workbench is closed, page 1 still shows the apps, and pressing one starts the workbench on that app. New apps and keys come from `paper_menus.json`, so the profile never needs editing again.

If the keys stay blank after the plugin service was stopped by hand, run `setup.ps1` again: it clears the crash marker that makes the service skip the plugin.

Keys show only the symbol. The **Paper Apps** folder actions still exist, but Options+ decorates folder keys, so the nine paper keys are the cleaner setup. If a key shows its name or a cropped symbol, select the key in Options+, reset its icon, and turn off the title.

| App | Top row | Middle row | Bottom row |
| --- | --- | --- | --- |
| VS Code | Camera, Code, Project | Language, Plot, Copy | Paste, Open, Home |
| MATLAB | Camera, Code, Project | Plot, Copy, Paste | Open, Save, Home |
| LaTeX | Camera, Equation, Table | Report, Copy, Paste | Browse, Overleaf, Home |
| ChatGPT | Camera, Brief, Copy | Paste, ChatGPT, Flow | Photo, Save photo, Home |
| KiCad | Camera, Schematic, Copy | Paste, Open, Save | Photo, Home |
| LTspice | Camera, Schematic, Simulate | Copy, Paste, Open | Save, Photo, Home |
| Git (also page 2) | Camera, Commit, Push | Pull, Branch, Diagram | Host, Browse, Home |
| Host (page 3) | Camera, Issue, Request | Publish, Open, Repo | Paste, Back, Home |
| Plan | Camera, Gantt, Copy | Paste, Open, Save | Home |

Use the All Applications profile to access PaperBridge when an app-specific profile takes focus. [Logitech profiles guide](https://support.logi.com/hc/en-nz/articles/25575505956119-Profiles-MX-Creative-Console-MX-Keypad) · [Logitech native folders](https://logitech.github.io/actions-sdk-docs/csharp/plugin-features/implementing-dynamic-folders/)

## Use both devices on a laptop with one USB-C port

Connect the LogiInsight camera directly to the laptop's USB-C port. Connect the MX Keypad to USB-A with a USB-A-to-USB-C **data cable**. If you need to share the USB-C port, use a powered SuperSpeed USB hub with downstream **data** ports. Reconnect both devices and check that Options+ sees the keypad and the RGB preview lists **LVNext Camera**. [Logitech camera SDK](https://raw.githubusercontent.com/Logitech/hackathons/master/docs/logiinsight/API.md) · [Microsoft USB guide](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-faq--introductory-level)
