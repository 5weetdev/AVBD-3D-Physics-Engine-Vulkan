#!/usr/bin/env python3
"""Link-completeness check for the avbd3d C ABI.

Every function declared with AVBD3D_API in include/avbd3d/*.h must have a definition
in src/api/*.cpp, otherwise code ported from Box3D fails at link time. This script
collects both sets with a regex pass over comment-stripped source and prints the declared
names that have no definition, grouped by header. Together with the exports of the built library this is the link
completeness gate. It exits 1 if any are undefined.

Declarations are recognised as AVBD3D_API ... name( ... ); with no body. Inline functions
(a body in the header) are skipped. Definitions are recognised as name( ... ) followed by '{'
in the .cpp sources. Names are matched only by name, not by signature.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER_DIR = os.path.join(ROOT, "include", "avbd3d")
SOURCE_DIR = os.path.join(ROOT, "src", "api")

NAME = re.compile(r"\b(avbd3d_\w+)\s*\(")
API = re.compile(r"\bAVBD3D_API\b")
# Typed joint accessors are defined by macro invocations in avbd3d_props.cpp, e.g.
# JOINT_SETF(avbd3d_DistanceJoint_SetLength, avbd3d_distanceJoint, "length"). The first
# argument of such an invocation at the start of a line is the function it defines.
MACRO_DEF = re.compile(r"^[ \t]*JOINT_[A-Z0-9_]+\(\s*(avbd3d_\w+)", re.M)


def strip_comments(text):
    """Remove // and /* */ comments and blank out string and char literals.

    Newlines are kept so that line structure survives. Preprocessor lines are blanked too,
    which drops the AVBD3D_API macro definition.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    stripped = "".join(out)
    lines = []
    for line in stripped.split("\n"):
        lines.append("" if line.lstrip().startswith("#") else line)
    return "\n".join(lines)


def declared_names(text):
    """Names of AVBD3D_API functions declared (not defined) in one header."""
    names = []
    for m in API.finditer(text):
        end = len(text)
        for ch in ";{}":
            k = text.find(ch, m.end())
            if k >= 0:
                end = min(end, k)
        if end >= len(text) or text[end] != ";":
            continue  # a body follows, so this is an inline definition
        decl = text[m.end():end]
        if re.search(r"\binline\b", decl):
            continue
        n = NAME.search(decl)
        if n and n.group(1).startswith("avbd3d_"):
            names.append(n.group(1))
    return names


def defined_names(text):
    """Names of avbd3d_ functions with a definition (a '{' after the parameter list)."""
    names = set()
    for m in NAME.finditer(text):
        before = text[: m.start()].rstrip()
        if before.endswith(".") or before.endswith("->"):
            continue
        depth = 0
        i = m.end() - 1  # at '('
        n = len(text)
        while i < n:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i >= n:
            continue
        j = i + 1
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j < n and text[j] == "{":
            names.add(m.group(1))
    return names


def main():
    defined = set()
    for path in sorted(glob.glob(os.path.join(SOURCE_DIR, "*.cpp"))):
        with open(path, encoding="utf-8") as f:
            defined |= defined_names(strip_comments(f.read()))

    undefined_by_header = {}
    total = 0
    for path in sorted(glob.glob(os.path.join(HEADER_DIR, "*.h"))):
        with open(path, encoding="utf-8") as f:
            names = declared_names(strip_comments(f.read()))
        missing = sorted(set(n for n in names if n not in defined))
        total += len(set(names))
        if missing:
            undefined_by_header[os.path.relpath(path, ROOT).replace(os.sep, "/")] = missing

    count = sum(len(v) for v in undefined_by_header.values())
    for header, missing in undefined_by_header.items():
        print(f"{header}: {len(missing)} undefined")
        for name in missing:
            print(f"    {name}")
    print(f"avbd3d declared: {total}, undefined: {count}")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
