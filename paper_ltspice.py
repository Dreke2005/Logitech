"""Turn a circuit read from paper into an LTspice schematic (.asc) and run it.

Every part is placed on a grid with a net flag on each pin, so parts that share a net
label are connected by name. That keeps the schematic valid for any topology the reader
reports; rearrange it in LTspice if you want drawn wires.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

# Symbol name and the two pin offsets (first net, second net) of the unrotated LTspice symbol.
SYMBOLS = {
    "R": ("res", (16, 16), (16, 96)),
    "C": ("cap", (16, 0), (16, 64)),
    "L": ("ind", (16, 16), (16, 96)),
    "D": ("diode", (16, 0), (16, 64)),  # anode, cathode
    "V": ("voltage", (0, 16), (0, 96)),  # +, -
    "I": ("current", (0, 0), (0, 80)),  # current flows from the first net to the second
}
REFERENCE = re.compile(r"[RCLDVI][1-9][0-9]{0,2}")
NET = re.compile(r"[A-Za-z0-9_+\-]{1,20}")
VALUE = re.compile(r"[0-9A-Za-zµ.,+\-*/(){}= ]{1,48}")
DIRECTIVE = re.compile(r"\.(tran|ac|dc|op|step|param|meas|noise|tf)\b[0-9A-Za-zµ.,+\-*/(){}=<> ]{0,80}", re.IGNORECASE)
GROUND = {"gnd", "ground", "0", "vss", "earth"}


def _net(name):
    return "0" if name.strip().lower() in GROUND else name.strip()


def default_directive(parts):
    """A simulation command that fits the sources: transient for time signals, AC sweep, else operating point."""
    values = " ".join(str(part.get("value", "")) for part in parts if str(part.get("reference", ""))[:1] in "VI")
    if re.search(r"\b(sine|pulse|pwl|exp|sffm)\b", values, re.IGNORECASE):
        return ".tran 10m"
    if re.search(r"\bac\b", values, re.IGNORECASE):
        return ".ac dec 20 1 1Meg"
    return ".op"


def check_parts(parts):
    """Validated (reference, value, net, net) tuples, or ValueError naming what to fix on the sketch."""
    if not isinstance(parts, list) or not 1 <= len(parts) <= 24 or any(not isinstance(x, dict) for x in parts):
        raise ValueError("A circuit needs 1-24 clearly labelled components.")
    checked, seen = [], set()
    for part in parts:
        reference, value, nets = part.get("reference"), part.get("value"), part.get("nets")
        reference = reference.strip().upper() if isinstance(reference, str) else ""
        if REFERENCE.fullmatch(reference) is None or reference in seen:
            raise ValueError("Label each part with a distinct reference such as R1, C1, L1, D1, V1, or I1.")
        if (not isinstance(nets, list) or len(nets) != 2
                or any(not isinstance(net, str) or NET.fullmatch(net.strip()) is None for net in nets)):
            raise ValueError(f"{reference} needs two net labels (for example in and GND).")
        value = " ".join(value.split()) if isinstance(value, str) else ""
        if reference[0] == "D" and not value:
            value = "D"
        if VALUE.fullmatch(value) is None:
            raise ValueError(f"{reference} needs a short value such as 1k, 100n, 5, or SINE(0 1 1k).")
        seen.add(reference)
        checked.append((reference, value, _net(nets[0]), _net(nets[1])))
    if not any("0" in part[2:] for part in checked):
        raise ValueError("Mark the ground node (GND) on the sketch; LTspice needs a reference node.")
    return checked


def build_asc(parts, directive=None):
    """An LTspice schematic for two-terminal parts connected by net labels."""
    checked = check_parts(parts)
    directive = " ".join((directive or "").split())
    if DIRECTIVE.fullmatch(directive) is None:
        directive = default_directive(parts)
    columns = 4
    rows = (len(checked) + columns - 1) // columns
    lines = ["Version 4", f"SHEET 1 {max(880, 160 + columns * 240)} {max(680, 240 + rows * 240)}"]
    for index, (reference, value, first, second) in enumerate(checked):
        symbol, top, bottom = SYMBOLS[reference[0]]
        x, y = 160 + index % columns * 240, 160 + index // columns * 240
        for (px, py), net, lead in ((top, first, -32), (bottom, second, 32)):
            lines.append(f"WIRE {x + px} {y + py} {x + px} {y + py + lead}")
            lines.append(f"FLAG {x + px} {y + py + lead} {net}")
        lines.extend((f"SYMBOL {symbol} {x} {y} R0", f"SYMATTR InstName {reference}", f"SYMATTR Value {value}"))
    lines.append(f"TEXT 160 {160 + rows * 240} Left 2 !{directive}")
    return "\n".join(lines) + "\n"


def netlist_text(parts, directive=None):
    """The same circuit as a readable SPICE netlist, shown for review beside the drawing."""
    checked = check_parts(parts)
    directive = " ".join((directive or "").split())
    if DIRECTIVE.fullmatch(directive) is None:
        directive = default_directive(parts)
    body = [f"{reference} {first} {second} {value}" for reference, value, first, second in checked]
    return "\n".join(["* Circuit read from paper", *body, directive, ".end"]) + "\n"


def find_ltspice():
    programs = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs"
    system = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    candidates = [programs / "ADI/LTspice/LTspice.exe", system / "ADI/LTspice/LTspice.exe",
                  system / "LTC/LTspiceXVII/XVIIx64.exe"]
    found = shutil.which("LTspice.exe")
    for path in ([Path(found)] if found else []) + candidates:
        if path.is_file():
            return path
    raise OSError("LTspice was not found. Install it or open the saved .asc on another PC.")


def open_ltspice(path, run=False):
    """Open a schematic in LTspice; with ``run`` the simulation starts and the waveform viewer opens."""
    subprocess.Popen([str(find_ltspice()), *(["-Run"] if run else []), str(path)])


def simulate(path, timeout=90):
    """Run the schematic in batch mode. Returns (ok, lines worth showing from the log)."""
    path = Path(path)
    log = path.with_suffix(".log")
    log.unlink(missing_ok=True)
    try:
        subprocess.run([str(find_ltspice()), "-b", "-Run", str(path)], timeout=timeout, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, [f"The simulation did not finish within {timeout} s."]
    if not log.is_file():
        return False, ["LTspice did not write a log; open the schematic to see why."]
    raw = log.read_bytes()
    text = raw.decode("utf-16-le" if raw[1:2] == b"\x00" else "utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    problems = [line for line in lines if re.search(r"error|unknown|singular|fail|can't|cannot|missing", line,
                                                    re.IGNORECASE)]
    results = [line for line in lines if re.match(r"[A-Za-z_][\w()]*\s*[:=]", line)
               and not line.lower().startswith(("circuit", "date", "total elapsed", "tnom", "temp", "method"))]
    return not problems, (problems or results)[:12]
