import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--overwrite", action="store_true")
args = vars(parser.parse_args())

file = args["file"]
overwrite = args["overwrite"]
base, ext = os.path.splitext(file)
ext = ext.lstrip(".")
assert ext in ("json", "jsonl"), f"{base=} {ext=} Extension is not 'json' or 'jsonl'"

alt_ext = "jsonl" if ext == "json" else "json"
alt_path = f"{base}.{alt_ext}"
if not overwrite and os.path.exists(alt_path):
    raise Exception(f"Path {alt_path} already exists! Overwrite with --overwrite flag")

with open(file, "r") as f:
    print(f"Converting .{ext} to .{alt_ext}")
    if ext == "json":
        obj = json.load(f)
        with open(alt_path, "w") as g:
            g.write("\n".join([json.dumps(v) for v in obj]))
    else:
        obj = [json.loads(l) for line in f.readlines() if (l := line.strip())]
        with open(alt_path, "w") as g:
            json.dump(obj, g, indent=2)