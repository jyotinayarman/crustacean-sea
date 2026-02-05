#!/usr/bin/env python3
"""
Post-process existing GLB files to normalize them to unit cube [-0.5, 0.5]^3.
Use this on already-generated GLBs so you don't need to regenerate.
"""
import argparse
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_glb import normalize_glb_to_unit_cube


def main():
    p = argparse.ArgumentParser(description="Normalize GLBs in a directory to unit cube.")
    p.add_argument("--data-dir", default="../generations", help="Directory containing .glb files")
    p.add_argument("--output-dir", default=None, help="If set, write normalized GLBs here; else overwrite in place")
    p.add_argument("--dry-run", action="store_true", help="Only list files that would be processed")
    args = p.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else data_dir
    if args.output_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    glbs = [f for f in os.listdir(data_dir) if f.lower().endswith(".glb")]
    if not glbs:
        print(f"No .glb files in {data_dir}")
        return
    if args.dry_run:
        for f in sorted(glbs):
            print(f)
        print(f"Would process {len(glbs)} file(s).")
        return
    ok, err = 0, 0
    for name in sorted(glbs):
        src = os.path.join(data_dir, name)
        dst = os.path.join(out_dir, name)
        try:
            buf = BytesIO()
            normalize_glb_to_unit_cube(src, buf)
            with open(dst, "wb") as f:
                f.write(buf.getvalue())
            print(f"OK {name}")
            ok += 1
        except Exception as e:
            print(f"FAIL {name}: {e}", file=sys.stderr)
            err += 1
    print(f"Done: {ok} OK, {err} failed.")


if __name__ == "__main__":
    main()
