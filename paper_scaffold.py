"""Project files straight from a class sketch, without a model.

A class diagram on paper is already a specification: class names, their fields and
methods, and arrows between them. ``sketch_scaffold`` turns the transcribed lines into
one source file per class plus an entry file, in the selected language. It is instant,
always syntactically valid, and contains exactly what was drawn. Free-form project
ideas, which need interpreting, are left to the language model.
"""

import keyword
import re

CLASS = re.compile(r"(?:class|struct|interface)?\s*([A-Z][A-Za-z0-9_]*)\s*(?:[:({].*)?$")
METHOD = re.compile(r"[+\-#~]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")
FIELD = re.compile(r"[+\-#~]?\s*([a-z_][A-Za-z0-9_]*)\s*(?::\s*([A-Za-z_][A-Za-z0-9_\[\], ]*))?$")
INLINE = re.compile(r"(?:(?:class|struct|interface)\s+)?([A-Z][A-Za-z0-9_]*)\s+((?:[+\-#~]?\s*[A-Za-z_]\w*\s*\([^()]*\)[\s,;]*)+.*)")
ARROW = re.compile(r"([A-Z][A-Za-z0-9_]*)\s*(?:-+>|=+>|→|uses|has)\s*([A-Z][A-Za-z0-9_]*)")


def parse_sketch(source):
    """Classes as {name: {"methods": [(name, [args])], "fields": [(name, type)], "uses": [names]}}, or None
    when the lines do not look like a class sketch."""
    classes, current = {}, None
    lines = []
    for raw in source.splitlines():
        line = raw.strip().strip("|").strip()
        # A reader often joins a box's title and its first members: "class Sensor read() calibrate()".
        joined = INLINE.fullmatch(line)
        lines += [f"class {joined[1]}", joined[2]] if joined else [line]
    for line in lines:
        if not line:
            continue
        arrow = ARROW.fullmatch(line)
        if arrow:
            for name in arrow.groups():
                classes.setdefault(name, {"methods": [], "fields": [], "uses": []})
            if arrow[2] not in classes[arrow[1]]["uses"]:
                classes[arrow[1]]["uses"].append(arrow[2])
            continue
        methods = METHOD.findall(line)
        declared = re.match(r"(?:class|struct|interface)\s+", line)
        heading = CLASS.fullmatch(line)
        if heading and not methods and (declared or len(line.split()) == 1):
            current = heading[1]
            classes.setdefault(current, {"methods": [], "fields": [], "uses": []})
            continue
        if current is None:
            return None
        if methods:
            for name, arguments in methods:
                names = [re.split(r"[:=]", part)[0].strip() for part in arguments.split(",") if part.strip()]
                names = [name_ for name_ in names if re.fullmatch(r"[A-Za-z_]\w*", name_)]
                if (name, names) not in classes[current]["methods"]:
                    classes[current]["methods"].append((name, names))
            continue
        fields = [FIELD.fullmatch(part.strip()) for part in re.split(r"[,;]", line) if part.strip()]
        if fields and all(fields):
            classes[current]["fields"] += [(match[1], (match[2] or "").strip()) for match in fields]
            continue
        return None  # A line that is neither a class, a member, nor an arrow: this is prose.
    if not classes or not any(info["methods"] or info["fields"] or info["uses"] for info in classes.values()):
        return None
    return classes


def _snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _safe(name):
    return name + "_" if keyword.iskeyword(name) else name


def _python(classes):
    files = []
    for name, info in classes.items():
        imports = "".join(f"from {_snake(other)} import {other}\n" for other in info["uses"] if other in classes)
        lines = [f"class {name}:", f'    """{name} from the paper sketch."""', ""]
        arguments = "".join(f", {_snake(other)}: {other} = None" for other in info["uses"])
        lines.append(f"    def __init__(self{arguments}):")
        body = [f"        self.{_safe(field)} = None" + (f"  # {kind}" if kind else "") for field, kind in info["fields"]]
        body += [f"        self.{_snake(other)} = {_snake(other)}" for other in info["uses"]]
        lines += body or ["        pass"]
        for method, parameters in info["methods"]:
            if method == "__init__":
                continue
            signature = "".join(f", {_safe(parameter)}" for parameter in parameters)
            lines += ["", f"    def {_safe(method)}(self{signature}):", f"        # TODO: implement {name}.{method}.",
                      "        raise NotImplementedError"]
        files.append((f"{_snake(name)}.py", (imports + ("\n\n" if imports else "")) + "\n".join(lines) + "\n"))
    imports = "".join(f"from {_snake(name)} import {name}\n" for name in classes)
    created = "".join(f"    {_snake(name)} = {name}()\n" for name in classes)
    main = f"{imports}\n\ndef main():\n{created}    # TODO: connect the parts.\n\n\nif __name__ == \"__main__\":\n    main()\n"
    return files + [("main.py", main)]


def _script(classes, typed):
    extension = ".ts" if typed else ".js"
    files = []
    for name, info in classes.items():
        imports = "".join(f"import {{ {other} }} from './{other}';\n" for other in info["uses"] if other in classes)
        fields = [f"  {field}{': unknown' if typed else ''};" for field, _ in info["fields"]] if typed else []
        fields += [f"  {_snake(other)}{'?: ' + other if typed else ''};" for other in info["uses"]] if typed else []
        methods = []
        for method, parameters in info["methods"]:
            signature = ", ".join(f"{parameter}{': unknown' if typed else ''}" for parameter in parameters)
            methods += [f"  {method}({signature}){': void' if typed else ''} {{",
                        f"    // TODO: implement {name}.{method}.", "    throw new Error('Not implemented');", "  }", ""]
        body = "\n".join(fields + ([""] if fields else []) + methods).rstrip()
        files.append((f"{name}{extension}", f"{imports}{chr(10) if imports else ''}// {name} from the paper sketch.\n"
                                            f"export class {name} {{\n{body}\n}}\n"))
    imports = "".join(f"import {{ {name} }} from './{name}';\n" for name in classes)
    created = "".join(f"const {_snake(name)} = new {name}();\n" for name in classes)
    return files + [(f"main{extension}", f"{imports}\n{created}// TODO: connect the parts.\n")]


def _cpp(classes):
    files = []
    for name, info in classes.items():
        includes = "".join(f'#include "{other}.hpp"\n' for other in info["uses"] if other in classes)
        members = [f"    double {field}{{}};" for field, _ in info["fields"]]
        members += [f"    {other}* {_snake(other)} = nullptr;" for other in info["uses"]]
        methods = [f"    void {method}({', '.join('double ' + parameter for parameter in parameters)}) {{\n"
                   f"        // TODO: implement {name}::{method}.\n    }}" for method, parameters in info["methods"]]
        body = "\n".join(["public:"] + methods + members)
        files.append((f"{name}.hpp", f"#pragma once\n{includes}\n// {name} from the paper sketch.\nclass {name} {{\n"
                                     f"{body}\n}};\n"))
    includes = "".join(f'#include "{name}.hpp"\n' for name in classes)
    created = "".join(f"    {name} {_snake(name)};\n" for name in classes)
    return files + [("main.cpp", f"{includes}\nint main() {{\n{created}    // TODO: connect the parts.\n    return 0;\n}}\n")]


def _matlab(classes):
    files = []
    for name, info in classes.items():
        properties = [f"        {field}" for field, _ in info["fields"]] + [f"        {_snake(other)}" for other in
                                                                               info["uses"]]
        methods = []
        for method, parameters in info["methods"]:
            signature = ", ".join(["obj", *parameters])
            methods += [f"        function {method}({signature})", f"            % TODO: implement {name}.{method}.",
                        "            error('Not implemented');", "        end"]
        block = ["    properties", *properties, "    end"] if properties else []
        files.append((f"{name}.m", "\n".join([f"classdef {name} < handle", f"    % {name} from the paper sketch.",
                                              *block, "    methods", *methods, "    end", "end"]) + "\n"))
    created = "".join(f"{_snake(name)} = {name}();\n" for name in classes)
    return files + [("main.m", f"% Entry point for the project from the paper sketch.\n{created}% TODO: connect the parts.\n")]


def sketch_scaffold(source, language):
    """(path, content) files for a class sketch in ``language``, or None when it is not a class sketch."""
    classes = parse_sketch(source)
    if classes is None or len(classes) > 7:
        return None
    if language in ("Python", "PyTorch"):
        return _python(classes)
    if language in ("JavaScript", "TypeScript"):
        return _script(classes, language == "TypeScript")
    if language == "C++":
        return _cpp(classes)
    if language == "MATLAB":
        return _matlab(classes)
    return None
