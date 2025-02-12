import numpy as np
import numpy.ma as ma
import argparse
import json
import math
import tqdm
import os
import time
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--from_folder", type=str, required=True)
parser.add_argument("--file_prefix", type=str, required=True)
parser.add_argument("--num_chunks", type=int, required=True)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--add_back_self", action="store_true")
args = parser.parse_args()
print(vars(args), flush=True)

statistics = defaultdict(float)
positive_list = []
bad_list = []

for chunk_idx in tqdm.tqdm(range(args.num_chunks)):
  to_load = os.path.join(args.from_folder, args.file_prefix.format(i=chunk_idx))
  with open(to_load, "rb") as f:
    obj = pickle.load(f)
  for k, v in obj["statistics"].items():
    statistics[k] += v
  positive_list.extend(obj["positive_list"])
  bad_list.extend(obj["bad_list"])
print(json.dumps(statistics, indent=4))
print(f"{len(positive_list)=}, {positive_list[:5]=}")

output_file = args.output_file
if output_file is None:
  output_file = os.path.join(args.from_folder, "reduction.pkl")
print(f"Write to {output_file}", flush=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "wb") as f:
  pickle.dump(positive_list, f)