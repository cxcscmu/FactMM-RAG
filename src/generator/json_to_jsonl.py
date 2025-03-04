import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = vars(parser.parse_args())
file = args["file"]

with open(file, "r") as f:
    obj = json.load(f)
with open(file.rstrip(".json") + ".jsonl", "w") as f:
    f.write("\n".join([json.dumps(v) for v in obj]))