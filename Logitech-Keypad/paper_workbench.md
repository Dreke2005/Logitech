# PaperBridge

Start the workbench from the main Logitech project directory. The main-folder
launcher reuses this workbench and the shared RGB camera code.

    uv run .\paper_workbench.py

The on-screen controls work without the MX Keypad. If you later install the
optional keypad plugin, run `Logitech-Keypad\keypad-plugin\setup.ps1`; it will
launch the same workbench with the keypad connection enabled.

Choose an app icon on page 1. Start **Camera**, the top-left key on every app page, then press **Code**, **Plot**, or **Project** before you write. The key captures a reference frame and starts tracking. The green box shows the detected paper or the writing area you selected by dragging in the preview. The yellow line shows hand motion. A magenta crosshair marks the pen tip, estimated from the thumb–index grip along the pen axis and snapped to the visible nib when the shaft can be seen (filled center). Blue lines are the registered pen-down strokes: a tip sample counts only when new ink appears under it, so hover moves are left out. Keep the paper still while you write.

To finish, draw a small closed circle after the text, apart from it and a bit larger than your letters, and hold still briefly. The orange ring confirms it, and generation starts without a key press. The circle itself is not transcribed. For **Diagram** and **Copy** a circle is part of the content, so finish those by moving your hand away or pressing the key again. That also works for every other key.

Reading uses the pen trace first. The model gets only the recorded pen-down strokes, drawn black on white, turned upright for the writer, and scaled to about 80 px per line of text. The key you pressed tells it what to expect, such as Python code, a MATLAB plot expression, a LaTeX table, a branch name, or git diagram commands. Each answer is checked for that key: an equation must parse, code must be balanced, and table rows must have the same number of cells. A failed check is retried once with the reason. Only when the trace is too sparse or the retry also fails does the model read the photo. That crop covers only the new writing, plus half a line height. New ink counts only as thin, dark marks on the paper near your strokes. Hands, the forearm, shadows, exposure changes, and objects you put down do not count. The status line shows whether the trace or the photo was read.

Frames and traces stay in memory. **Capture page** freezes a frame for manual selection and a one-press conversion. **Resume live** returns to the camera. Pressing **Camera** again stops the stream.

| Output | What the workbench produces |
| --- | --- |
| LaTeX | **Equation** opens a TeX draft locally. **Table** turns a drawn table into a booktabs table (see [Tables](#tables)). **Report** creates and opens a report skeleton. **Browse** selects and opens an existing project for **Paste**, **Table**, and **Overleaf**. |
| MATLAB | **Code** generates and opens a `.m` file. **Project** creates a folder of MATLAB files and opens its entry file in the running MATLAB, or starts MATLAB in that folder. **Plot** creates figures from supported expressions. |
| VS Code | **Code** generates and opens a source file. **Project** opens the scaffold folder in VS Code. **Language** opens the language icons. |
| ChatGPT | **Brief** drafts a prompt from notes and sketches. **Paste** stages copied text or code in a review prompt. **Open ChatGPT** copies the draft and opens the browser. Paste and submit it when ready. |
| KiCad | **Schematic** drafts an editable legacy `.sch` for resistors, capacitors, inductors, and diodes. Mark component references, values, and net labels on the sketch. |
| Git | **Repo** turns a written project idea into a scaffold with an initial commit. **Branch** and **Commit** read the branch name or commit message from your writing. **Diagram** runs a drawn branch graph. See [Git](#git). |

Compare **Read from paper** with your page and check signs, subscripts, variable names, and assumptions. You can edit the transcription before clicking **Open VS Code Chat** in the workbench. That action passes a local Markdown file with the reviewed transcription to VS Code Chat in Ask mode. It uses the chat provider configured in VS Code. Open the generated draft in the editor to continue with inline suggestions or your installed Codex extension. [VS Code chat CLI](https://code.visualstudio.com/docs/configure/command-line#_start-chat-from-the-command-line)

## Copy and paste between apps

To copy something on paper, draw a **dashed oval** around it, with short gaps and space between the oval and the writing. Lift your hand and press **Copy**. If the oval is missed, freeze the page and drag around it. Copy also takes the current draft, or the text selected in it. PaperBridge records *what* it copied (equation, code, table, plot, diagram, todo, or text) in `%LOCALAPPDATA%\LogiInsightPaper\clipboard.json`, with the source app and any rendered figure. The plain text also goes to the Windows clipboard. A copied plot is rendered right away.

Open another app's page and press **Paste**. The result depends on what was copied and where you paste it. Generated files open in the target app, and the status line names the copied kind and what Paste made of it. Paste uses this typed copy only while the Windows clipboard still holds the copied text. Anything you copy elsewhere afterward is pasted as before.

| Copied | LaTeX | VS Code | MATLAB | ChatGPT | Git |
| --- | --- | --- | --- | --- | --- |
| Equation | `equation` environment file plus `\input` | Function in the selected language | Anonymous function plus `fplot` script | Explain-and-check prompt | Commit message |
| Code | `lstlisting` file | New file in the open project, or a draft | `.m` file | Review prompt | Commit message |
| Table | Booktabs table in `tables/` | CSV file | `table()` or `array2table` script | Explain prompt | — |
| Plot | Figure in `figures/` | Matplotlib script | Plot script | Explain prompt | — |
| Diagram | Figure if rendered, otherwise `verbatim` | Markdown note | Comments | Explain prompt | Runs it as a [diagram](#draw-a-branch-diagram) |
| Todo | `itemize` list | TODO comments | TODO comments | Planning prompt | Adds the open items to `TODO.md` and commits only that file |
| Text | Paragraph | Markdown note | Comments | Review prompt | Commit message for current changes |

In LaTeX, fragments go into the project chosen with **Browse**. PaperBridge adds the `\input` line before `\end{document}` of `main.tex` or the only complete document, and loads `booktabs` or `listings` there if needed. Without a clear root file, the `\input` line goes to the clipboard. Without a project, Paste asks for one and saves a standalone draft if you cancel. ChatGPT pastes are staged in the workbench; press **ChatGPT** to copy the prompt and open the chat.

### Tables

On the LaTeX page, press **Table** and draw a grid. Separate the columns with vertical lines and write one row per line. A first row of labels becomes the header. PaperBridge escapes LaTeX special characters and right-aligns numeric columns. It saves a booktabs `tabular` in a `table` float as `tables/table_<n>.tex` in the selected project, with `\input` and `\usepackage{booktabs}` added as for equations. It then opens the root file. Without a selected project, it opens a standalone draft instead.

## Transfer a graph or code

The **Overleaf** icon packages the selected local project as a ZIP and opens the Overleaf dashboard. Choose **New project → Existing project (.zip)** to upload it. Generated working files are saved under `%LOCALAPPDATA%\LogiInsightPaper\drafts`. [Overleaf project upload](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)

1. In MATLAB or VS Code, start the camera, press **Plot**, then write one equation. If the equation is already on paper, use **Capture page** and select it before pressing **Plot**. You can also use the **Paste** key to stage an equation copied from an editor. The plotter accepts scalar functions of `x`, basic math functions, and simple `plot(x, y)` snippets. It uses the written range when available and defaults to −5 through 5. Use **Open plot preview** in the workbench to view the PNG.
2. When the plot is ready, use the keypad **Copy** icon. It copies the path to the rendered PDF as a typed plot. Open the LaTeX menu and press **Paste**. Choose a LaTeX project folder if you haven't saved a report in the workbench. The import copies the figure into `figures/` and places the matching `\includegraphics` snippet on your clipboard.
3. For a more complex plot, generate it in MATLAB, VS Code, or PyCharm and export a PDF or PNG. Copy the file in Explorer and use **Paste** in the LaTeX menu. The same action accepts an image from the Windows clipboard. For code review, copy text in the editor, then use **Paste** in the ChatGPT menu to stage it.

The camera path uses 1080p MJPG via the LogiInsight SDK's OpenCV capture helper. The PC vision model reads the page from RGB frames. The camera's SFTP AI JSON provides face detections and landmarks. [Logitech camera SDK](https://raw.githubusercontent.com/Logitech/hackathons/master/docs/logiinsight/API.md)

## Git

Open **Git**, its own icon on page 1, and press **Camera** to start tracking. Generated code, projects, and reports open automatically in VS Code or MATLAB.

- **Repo**: write a project idea, then lift your hand or press **Repo** again. PaperBridge generates a scaffold in the selected VS Code language, runs `git init` and an initial commit under `%LOCALAPPDATA%\LogiInsightPaper\repos`, and opens the folder in VS Code.
- **Browse** selects an existing repository and opens it in VS Code. **Branch**, **Commit**, **Diagram**, and **Paste** ask for one when none is selected.
- **Branch**: write a short name such as *add login page*. It becomes a valid branch name, `add-login-page`, and is checked out.
- **Commit**: write the message and draw a small circle after its last word to end the recording. PaperBridge commits staged changes, or all changes when nothing is staged.
- **Push** pushes the current branch and sets its upstream on the first push. It never force-pushes. If the repository has no remote, the status line says so; add one with `git remote add origin URL`.
- **Paste** commits a copied note: text becomes the commit message for the current changes, and a todo list is appended to `TODO.md` and committed on its own.

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

## Run the local model

Install Ollama if it is unavailable and pull the vision model once.

    ollama pull qwen3-vl:2b-instruct

The workbench sends the captured still image to the Ollama service on this computer. It also detects the portable model installation used for testing under **%LOCALAPPDATA%\LogiInsightTools**.

## Configure the MX Keypad

From the main project folder, install the Python environment first. The keypad
plugin is optional; the Paper Workbench's on-screen controls work without it.
For a new keypad installation, use the .NET 10 SDK with Logi Options+ and Logi
Plugin Service 6.4 or newer.

    uv sync
    .\Logitech-Keypad\keypad-plugin\setup.ps1

If the actions are missing, use the Options+ setting to restart Logi Plugin Service. In **All Applications**, create or choose the **PaperBridge** profile. In **LogiInsight Paper → Paper Apps**, assign **VS Code**, **MATLAB**, **LaTeX**, **ChatGPT**, **KiCad**, and **Git** to page 1. Each app opens a native folder with icon-only actions. Keys show only the symbol; the setup script installs full-image icon templates without a title. Use **Home** to return to the app icons. If a key still shows its name or a cropped symbol, select the key in Options+, reset its icon, and turn off the title, so it uses the plugin's full-image template.

| App | Top row | Middle row | Bottom row |
| --- | --- | --- | --- |
| VS Code | Camera, Code, Project | Language, Plot, Copy | Paste, Open, Home |
| MATLAB | Camera, Code, Project | Plot, Copy, Paste | Open, Save, Home |
| LaTeX | Camera, Equation, Table | Report, Copy, Paste | Browse, Overleaf, Home |
| ChatGPT | Camera, Brief, Copy | Paste, ChatGPT, Photo | Save photo, Home |
| KiCad | Camera, Schematic, Copy | Paste, Open, Save | Photo, Home |
| Git | Camera, Repo, Branch | Commit, Diagram, Push | Browse, Paste, Home |

Use the All Applications profile to access PaperBridge when an app-specific profile takes focus. [Logitech profiles guide](https://support.logi.com/hc/en-nz/articles/25575505956119-Profiles-MX-Creative-Console-MX-Keypad) · [Logitech native folders](https://logitech.github.io/actions-sdk-docs/csharp/plugin-features/implementing-dynamic-folders/)

## Use both devices on a laptop with one USB-C port

Connect the LogiInsight camera directly to the laptop's USB-C port. Connect the MX Keypad to USB-A with a USB-A-to-USB-C **data cable**. If you need to share the USB-C port, use a powered SuperSpeed USB hub with downstream **data** ports. Reconnect both devices and check that Options+ sees the keypad and the RGB preview lists **LVNext Camera**. [Logitech camera SDK](https://raw.githubusercontent.com/Logitech/hackathons/master/docs/logiinsight/API.md) · [Microsoft USB guide](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-faq--introductory-level)
