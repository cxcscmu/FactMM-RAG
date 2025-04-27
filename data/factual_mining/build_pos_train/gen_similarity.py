import numpy as np
import argparse
import json
import math
import tqdm
import os
import time

from utils import chexbert_similarity, radgraph_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_file", type=str,
                    default="FactMM-RAG/data/mimic/train_labeled.json")
parser.add_argument("--output_folder", type=str,
                    default="FactMM-RAG/data/mimic/scoring_chunks_train")
parser.add_argument("--num_chunks", type=int, default=64)
parser.add_argument("--chunk_id", type=int, required=True)
args = parser.parse_args()

with open(args.train_data_file, "r") as f:
  obj = json.load(f)
os.makedirs(args.output_folder, exist_ok=True)
n = len(obj)
chunk_size = (n + args.num_chunks - 1) // args.num_chunks
start = chunk_size * args.chunk_id
end = min(chunk_size * (args.chunk_id + 1), n)
print(
  f"Chunking({n=}) [{args.chunk_id}] ({args.chunk_id + 1}/{args.num_chunks}): [{start}, {end})", flush=True)

out_chexbert_sims = np.zeros((end - start, n))
out_radgraph_sims = np.zeros((end - start, n))
a = time.time()
for i in tqdm.tqdm(range(start, end), desc="outer"):
  query_i = obj[i]
  for j in range(n):
    doc_j = obj[j]
    chex_sim = chexbert_similarity(query_i, doc_j)
    radg_sim = radgraph_similarity(query_i, doc_j)
    out_chexbert_sims[i - start, j] = chex_sim
    out_radgraph_sims[i - start, j] = radg_sim
b = time.time()
print(f"Time Taken: {(b-a):0.4f} s", flush=True)
print(f"Output: {out_chexbert_sims.shape=}, {out_radgraph_sims.shape=}")
output_file = os.path.join(args.output_folder)
print(f"Saving to: {args.output_folder}", flush=True)
np.save(os.path.join(args.output_folder,
        f"chex_{args.chunk_id}"), out_chexbert_sims)
np.save(os.path.join(args.output_folder,
        f"radg_{args.chunk_id}"), out_radgraph_sims)