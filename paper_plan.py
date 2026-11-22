"""Turn a handwritten project pipeline into a schedule: a Mermaid Gantt chart and a rendered figure.

The reader writes one line per item:

    task: Literature review, 2w
    task: Prototype, 3w, after Literature review
    milestone: Demo, after Prototype

A task without ``after`` follows the item above it. Durations use d, w, or m (days,
weeks, months of 30 days); a missing duration is one week.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

ITEM = re.compile(r"(task|milestone)\s*:\s*(.+)", re.IGNORECASE)
DURATION = re.compile(r"(\d+(?:[.,]\d+)?)\s*(d|days?|w|wk|wks|weeks?|m|mo|months?)\b", re.IGNORECASE)
UNIT_DAYS = {"d": 1, "w": 7, "m": 30}


@dataclass
class Item:
    name: str
    days: int  # 0 for a milestone
    after: str = ""
    start: date = None
    end: date = None

    @property
    def milestone(self):
        return self.days == 0


def _clean(name):
    return " ".join(re.sub(r"[:;#\"]", " ", name).split())[:60]


def parse_plan(text):
    """(items, problems) from the reader's lines. Unknown lines are reported, not guessed."""
    items, problems = [], []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        match = ITEM.fullmatch(line)
        if match is None:
            problems.append(f"Not a task or milestone: {line[:60]}")
            continue
        fields = [field.strip() for field in match[2].split(",")]
        name, days, after = _clean(fields[0]), None, ""
        for field in fields[1:]:
            duration = DURATION.fullmatch(field)
            if duration:
                days = max(1, round(float(duration[1].replace(",", ".")) * UNIT_DAYS[duration[2][0].lower()]))
            elif field.lower().startswith("after "):
                after = _clean(field[6:])
        if not name:
            problems.append(f"No name in: {line[:60]}")
            continue
        if any(item.name.lower() == name.lower() for item in items):
            problems.append(f"'{name}' appears twice; the second one was left out.")
            continue
        items.append(Item(name, 0 if match[1].lower() == "milestone" else (days or 7), after))
    return items, problems


def schedule(items, start=None):
    """Give every item a start and end date. Returns problems such as an unknown 'after'."""
    start = start or date.today()
    problems, done = [], {}
    for index, item in enumerate(items):
        anchor = None
        if item.after:
            anchor = done.get(item.after.lower())
            if anchor is None:
                problems.append(f"'{item.name}' comes after '{item.after}', which is not an earlier item; "
                                "it follows the item above instead.")
        if anchor is None and index:
            anchor = items[index - 1]
        item.start = anchor.end if anchor is not None else start
        item.end = item.start + timedelta(days=item.days)
        done[item.name.lower()] = item
    return problems


def mermaid(items, title="Project plan"):
    lines = ["gantt", f"    title {title}", "    dateFormat YYYY-MM-DD", "    section Plan"]
    for index, item in enumerate(items):
        if item.milestone:
            lines.append(f"    {item.name} :milestone, m{index}, {item.start.isoformat()}, 0d")
        else:
            lines.append(f"    {item.name} :t{index}, {item.start.isoformat()}, {item.days}d")
    return "\n".join(lines)


def render(items, path, title="Project plan"):
    """Draw the chart to ``path`` (.png) and the same name as .pdf. Returns the PNG path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 0.55 * len(items) + 1.6))
    for row, item in enumerate(items):
        if item.milestone:
            axis.plot(mdates.date2num(item.start), row, marker="D", markersize=11, color="#e0553a", zorder=3)
        else:
            axis.barh(row, item.days, left=mdates.date2num(item.start), height=0.55, color="#4f8fe6", zorder=2)
        axis.text(mdates.date2num(item.end) + (1.3 if item.milestone else 0.6), row, item.end.strftime("%d %b") if item.milestone
                  else f"{item.days} d", va="center", fontsize=8, color="#444444")
    axis.set_yticks(range(len(items)), [item.name for item in items])
    axis.invert_yaxis()
    axis.xaxis_date()
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.set_xlim(mdates.date2num(items[0].start) - 1, mdates.date2num(max(item.end for item in items)) + 6)
    axis.grid(axis="x", color="#dddddd", zorder=0)
    axis.set_title(title, loc="left", fontsize=12)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def check_plan(text):
    """Validator for the reader: at least two items in the expected form."""
    items, problems = parse_plan(text)
    if problems:
        return problems[0] + ". Write each line as 'task: name, 2w' or 'milestone: name'"
    if len(items) < 2:
        return "a plan needs at least two lines such as 'task: name, 2w'"
    return None


def summary(items):
    end = max(item.end for item in items)
    tasks = sum(not item.milestone for item in items)
    return (f"{tasks} tasks and {len(items) - tasks} milestones, {items[0].start.strftime('%d %b')} to "
            f"{end.strftime('%d %b %Y')} ({(end - items[0].start).days} days)")
