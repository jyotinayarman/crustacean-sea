#!/usr/bin/env python3
"""Remove leading license/comment blocks from all .py files under this directory."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def is_special_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith("#!") or ("coding" in s and s.startswith("#"))


def remove_header(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        return False
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if is_special_comment(line):
            out.append(line)
            i += 1
            continue
        if s.startswith("#"):
            i += 1
            continue
        out.extend(lines[i:])
        break
    else:
        return False
    if len(out) == len(lines):
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return True


def main():
    changed = 0
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        if "remove_header_comments" in dirpath:
            continue
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if path == os.path.join(ROOT, "remove_header_comments.py"):
                continue
            try:
                if remove_header(path):
                    changed += 1
                    print(path)
            except Exception as e:
                print(f"Error {path}: {e}")
    print(f"Stripped header from {changed} file(s).")


if __name__ == "__main__":
    main()
