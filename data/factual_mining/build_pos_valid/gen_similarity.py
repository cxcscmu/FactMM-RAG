import numpy as np
import argparse
import json
import math
import tqdm
import os
import time

from utils import chexbert_similarity, radgraph_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--query_data_file", type=str,
                    default="FactMM-RAG/data/mimic/valid_labeled.json")
parser.add_argument("--corpus_data_file", type=str,
                    default="FactMM-RAG/data/mimic/train_labeled.json")
parser.add_argument("--output_folder", type=str,
                    default="FactMM-RAG/data/mimic/scoring_chunks_valid")
parser.add_argument("--num_chunks", type=int, default=16)
parser.add_argument("--chunk_id", type=int, required=True)
args = parser.parse_args()

with open(args.query_data_file, "r") as q, open(args.corpus_data_file, "r") as c:
  obj_query = json.load(q)
  obj_corpus = json.load(c)
os.makedirs(args.output_folder, exist_ok=True)
n = len(obj_query)
n_corpus = len(obj_corpus)
chunk_size = (n + args.num_chunks - 1) // args.num_chunks
start = chunk_size * args.chunk_id
end = min(chunk_size * (args.chunk_id + 1), n)
print(
  f"Chunking({n=}) [{args.chunk_id}] ({args.chunk_id + 1}/{args.num_chunks}): [{start}, {end})", flush=True)

out_chexbert_sims = np.zeros((end - start, n_corpus))
out_radgraph_sims = np.zeros((end - start, n_corpus))
a = time.time()
for i in tqdm.tqdm(range(start, end), desc="outer"):
  query_i = obj_query[i]["finding"]
  for j in range(n_corpus):
    doc_j = obj_corpus[j]["finding"]
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