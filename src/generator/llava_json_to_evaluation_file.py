import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = vars(parser.parse_args())
file = args["file"]
base, ext = os.path.splitext(file)
ext = ext.lstrip(".")
assert ext == "json"
write_file = f"{base}_inference.{ext}"

with open(file, "r") as f:
    obj = json.load(f)
with open(write_file, "w") as f:
    out = [
        {
            "retrieved_finding": [v["text"]]
        } for v in obj
    ]
    json.dump(out, f)