# Silent transcription

Use the RGB preview to mouth a short English phrase and see Auto-AVSR's draft in the same window. Visually similar sounds can produce different words. Published accuracy on spoken video measures a different task from recognizing deliberately silent speech.

## Set up the model on Windows

Run these commands from the Logitech project directory. The model environment lives beside the project and has its own dependencies.

```powershell
git clone https://github.com/mpc001/auto_avsr.git ..\auto_avsr
git -C ..\auto_avsr checkout --detach 182b62837773ab01052d4ac21ef1d2203ea7d267
uv venv --python 3.12 ..\.venv-auto-avsr
uv pip install --python ..\.venv-auto-avsr\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
uv pip install --python ..\.venv-auto-avsr\Scripts\python.exe "mediapipe==0.10.21" "numpy<2" "opencv-contrib-python==4.11.0.86" "pytorch-lightning<3" sentencepiece scikit-image six
```

Download the visual-only `vsr_trlrs2lrs3vox2avsp_base.pth` checkpoint (about 1 GB) to the parent directory. Review the pretrained model terms before use. [Auto-AVSR checkpoint](https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view?usp=sharing) · [Auto-AVSR model and license information](https://github.com/mpc001/auto_avsr#license)

## Read lips in the live preview

```powershell
uv run rgb_preview.py --mode lip
```

Wait for the model to load. Keep your full face visible in even light, press **Space** to start, then mouth one phrase without voicing it. Press **Space** again to see the result below the preview. Processing starts automatically at 10 seconds. The app keeps the frames in memory while processing and uses RGB video only.

If you installed the model elsewhere, use `--vsr-root PATH` with a folder containing the `upstream` checkout, `.venv/Scripts/python.exe`, and `checkpoint.pth`.

### Suggest corrections using context

With the local Ollama model installed, add `--lip-correct qwen3-vl:2b-instruct`.
You can also give it a subject, such as `--lip-topic "machine learning"`.
The raw draft appears first. The correction step proposes small word changes
that are close to the visual draft. Press **R** to compare a suggestion with the
original. A badly misread phrase can still be wrong after correction.

With the machine-learning topic, the correction changed “gradient dissent” to
“gradient descent.” The corrected drafts had 70 word errors over 97 reference
words on the ten deliberately mouthed test phrases below, the same total as
the raw visual model.

### Accuracy on deliberately mouthed phrases

The Windows CPU worker matched four of five short test clips word for word after ignoring case and punctuation. The fifth draft raised the total to 10 word edits over 30 reference words. On the first five phrases from the SoniSpeech silent evaluation session `009`, it matched none exactly and made 60 word edits over 67 reference words. These small samples show how much results can vary across recordings. [StormHacks recording protocol](https://github.com/LMSAIH/stormhacks2026/blob/master/.context/b2-recording-guide.md#each-clip) · [Cornell SoniSpeech dataset](https://doi.org/10.7298/xjjr-9m85)

## Transcribe a video file

```powershell
uv run rgb_preview.py --no-hands --record-video mouthed-phrase.mp4
```

Face the camera with your mouth well lit. Press Space to start recording, mouth a short phrase, then press Space to stop. Keep one face in frame and keep the clip under 15 seconds. The preview records RGB video without microphone audio.

```powershell
uv run silent_transcription.py mouthed-phrase.mp4 --repo ..\auto_avsr --python ..\.venv-auto-avsr\Scripts\python.exe --checkpoint ..\vsr_trlrs2lrs3vox2avsp_base.pth --output draft.txt
```

The program prints the estimated text and, with `--output`, saves it to a new UTF-8 file. Use a new output name for each draft. You can also pass an existing video file with a single clearly visible face. Inference runs locally on CPU by default. [Auto-AVSR inference notebook](https://github.com/mpc001/auto_avsr/blob/main/tutorials/inference.ipynb)
