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


PROBLEM = re.compile(r"error|unknown|undefined|singular|fail|can't|cannot|missing|expected|over-defined|floating|"
                     r"not found|unable|could not|convergence|time step too small|unrecognized|syntax", re.IGNORECASE)
# What LTspice's messages mean on a drawn circuit, in the order they are looked for.
EXPLANATIONS = (
    (r"paralleled|over-defined", "Two voltage sources (or a source and a short) sit across the same two nets. Check "
                                 "the net labels on the sources."),
    (r"singular matrix", "The circuit cannot be solved: a node has no DC path to ground, or sources fight each other. "
                         "Check the ground and the net labels."),
    (r"node (\S+) is floating", "A net is connected to nothing else or has no path to ground. Check its label on the "
                                "sketch."),
    (r"undefined model|unknown (?:model|subcircuit)|could not find|not found", "A part's value is not something "
     "LTspice knows. Write resistor, capacitor, and inductor values as numbers with a unit prefix (1k, 100n, 10m)."),
    (r"expected device|unknown syntax|unrecognized|syntax|missing|unknown parameter", "A line of the netlist is "
     "incomplete. Check the simulation command (for example .tran 10m) and that every part has a value."),
    (r"time step too small|convergence", "The simulation did not converge. Try a shorter .tran time or add a small "
                                         "series resistance."),
)


def _log_text(path):
    raw = Path(path).read_bytes()
    return raw.decode("utf-16-le" if raw[1:2] == b"\x00" else "utf-8", errors="replace")


def read_results(path):
    """What a finished run found, from the .raw file next to the schematic: the operating point as
    'V(out) = 3.84 V' lines, or the range of every node voltage over a transient or sweep. [] when the
    file is missing or in a layout this does not know."""
    import struct

    lines = []
    for raw_path in (Path(path).with_suffix(".raw"), Path(path).with_suffix(".op.raw")):
        try:
            data = raw_path.read_bytes()
        except OSError:
            continue
        marker = "Binary:\n".encode("utf-16-le")
        split = data.find(marker)
        if split < 0:
            continue
        header, body = data[:split].decode("utf-16-le", errors="replace"), data[split + len(marker):]
        plot = re.search(r"Plotname:\s*(.+)", header)
        count = re.search(r"No\. Points:\s*(\d+)", header)
        names = re.findall(r"^\t\d+\t(\S+)\t\S+", header, flags=re.M)
        if not plot or not count or not names or "complex" in header.split("Variables:")[0].lower():
            if plot and count:
                lines.append(f"{plot[1].strip()} finished with {int(count[1])} points.")
            continue
        points, variables = int(count[1]), len(names)
        if len(body) == points * (8 + 4 * (variables - 1)):
            layout = "<d" + "f" * (variables - 1)  # The first variable is a double, the others are floats.
        elif len(body) == points * 8 * variables:
            layout = "<" + "d" * variables
        else:
            continue
        size = struct.calcsize(layout)
        rows = [struct.unpack_from(layout, body, index * size) for index in range(points)]
        operating = plot[1].strip().lower().startswith("operating")
        if raw_path.name.endswith(".op.raw") and lines:
            continue  # The transient already describes the run.
        shown = [(name, [row[index] for row in rows]) for index, name in enumerate(names)
                 if name.upper().startswith(("V(", "I(")) and (operating or name.upper().startswith("V("))]
        for name, values in shown[:8]:
            unit = "V" if name.upper().startswith("V(") else "A"
            if operating or len(values) == 1:
                lines.append(f"{name} = {values[0]:.4g} {unit}")
            else:
                lines.append(f"{name}: {min(values):.4g} to {max(values):.4g} {unit}")
        if not operating:
            lines.insert(0, f"{plot[1].strip()}, {points} points.")
        if lines:
            break
    return lines


def explain(problems):
    """LTspice's own messages with a first line that says what to check on the sketch."""
    text = " ".join(problems)
    hint = next((message for pattern, message in EXPLANATIONS if re.search(pattern, text, re.IGNORECASE)), None)
    cleaned = []
    for line in problems:
        line = re.sub(r"^[A-Za-z]:\\\\?[^()]*\((\d+)\):\s*", r"Netlist line \1: ", line)  # Drop the file path.
        if line not in cleaned and not re.fullmatch(r"[\^ ]+", line):
            cleaned.append(line)
    return ([hint] if hint else []) + cleaned


def simulate(path, timeout=90):
    """Run the schematic in batch mode. Returns (ok, lines): on success what the run found (node voltages),
    preceded by any warnings; on failure what to check on the sketch, then LTspice's own messages."""
    path = Path(path)
    log = path.with_suffix(".log")
    for old in (log, path.with_suffix(".raw"), path.with_suffix(".op.raw")):
        old.unlink(missing_ok=True)
    try:
        subprocess.run([str(find_ltspice()), "-b", "-Run", str(path)], timeout=timeout, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, [f"The simulation did not finish within {timeout} s. Shorten the .tran time."]
    if not log.is_file():
        return False, ["LTspice did not write a log; open the schematic to see why."]
    lines = [line.strip() for line in _log_text(log).splitlines() if line.strip()]
    problems = []
    for index, line in enumerate(lines):
        if line.lower().startswith(("circuit:", "files loaded", "ltspice ")) or "succeeded" in line.lower():
            continue
        if PROBLEM.search(line) and not re.fullmatch(r"[A-Za-z]:\\[^():]*", line):
            problems.append(line)
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if re.search(r"\(\d+\):", line) and following and not PROBLEM.search(following):
                problems.append(f"in: {following}")  # LTspice echoes the netlist line it stopped at.
            elif following.lower().startswith("you will need"):
                problems.append(following)
    ran = path.with_suffix(".raw").is_file() or path.with_suffix(".op.raw").is_file()
    warnings = [line for line in problems if line.lower().startswith("warning")]
    if not ran or len(warnings) < len([line for line in problems if not line.startswith("in: ")]):
        return False, (explain(problems) or ["LTspice produced no result; open the schematic to see why."])[:12]
    return True, ([*explain(warnings), *read_results(path)] or ["The simulation ran."])[:12]
