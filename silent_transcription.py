"""Transcribe a short English clip from mouth movement using local Auto-AVSR."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

UPSTREAM = "https://github.com/mpc001/auto_avsr"


def _require_file(path, label):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    return path


def backend_paths(repo, python, checkpoint):
    """Validate the external checkout and model installation."""
    python = _require_file(python, "Auto-AVSR environment's Python executable")
    checkpoint = _require_file(checkpoint, "Visual-only checkpoint")
    repo = repo.expanduser().resolve()
    required = ("lightning.py", "datamodule/transforms.py",
                "preparation/detectors/mediapipe/video_process.py", "spm/unigram/unigram5000.model")
    if not all((repo / path).is_file() for path in required):
        raise ValueError(f"Missing Auto-AVSR files in {repo}. Clone {UPSTREAM} and pass its directory to --repo.")
    return repo, python, checkpoint


def transcribe(clip, repo, python, checkpoint, device="cpu"):
    """Return draft text from a video file. Run model dependencies in a separate environment."""
    clip = _require_file(clip, "Video")
    repo, python, checkpoint = backend_paths(repo, python, checkpoint)
    worker = Path(__file__).with_name("silent_auto_avsr.py").resolve()
    with TemporaryDirectory(prefix="logiinsight-lip-") as temp:
        result_path = Path(temp) / "transcript.txt"
        result = subprocess.run(
            [str(python), "-I", str(worker), str(repo), str(clip), str(checkpoint), device, str(result_path)],
            cwd=repo, capture_output=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            details = result.stderr.strip() or result.stdout.strip()
            message = details.splitlines()[-1] if details else f"exit code {result.returncode}"
            raise RuntimeError(f"Auto-AVSR failed. {message}")
        if not result_path.is_file():
            raise RuntimeError("Auto-AVSR finished without returning a transcript.")
        transcript = result_path.read_text(encoding="utf-8").strip()
        if not transcript:
            raise RuntimeError("Auto-AVSR could not read any words from the clip.")
        return transcript


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Recorded video, one person facing the camera, at most 15 seconds")
    parser.add_argument("--repo", type=Path, required=True, help="Local checkout of mpc001/auto_avsr")
    parser.add_argument("--python", type=Path, required=True, help="Python executable in the separate Auto-AVSR environment")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted visual-only Auto-AVSR checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Write draft text to a new UTF-8 file")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        transcript = transcribe(args.clip, args.repo, args.python, args.checkpoint, args.device)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(transcript + "\n")
        print(transcript)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
