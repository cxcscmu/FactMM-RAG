import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = vars(parser.parse_args())
file = args["file"]

with open(file, "r") as f:
    obj = [json.loads(l) for line in f.readlines() if (l := line.strip())]
with open(os.path.splitext(file)[0]+".json", "w") as f:
    out = [
        {
            "retrieved_finding": [v["text"]]
        } for i, v in enumerate(obj)
    ]
    json.dump(out, f)