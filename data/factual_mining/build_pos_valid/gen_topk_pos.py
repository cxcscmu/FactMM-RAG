import numpy as np
import numpy.ma as ma
import argparse
import json
import math
import tqdm
import os
import time
import pickle

# Does not support removing self
parser = argparse.ArgumentParser()
parser.add_argument("--from_folder", type=str, required=True)
parser.add_argument("--do_chex", action="store_true")
parser.add_argument("--do_radg", action="store_true")
parser.add_argument("--skip_bad_sample", action="store_true",
                    help="Activate to return empty sample list, in case radgraph score with self is 0 (usually indicates bad data sample)")
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--num_chunks", type=int, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--pre_mask", action="store_true",
                    help="Filter all rows by chexbert or radgraph value, before doing top-k operation")
parser.add_argument("--pre_mask_chex", type=float,
                    help="Lower bound for chexbert filtering", default=0)
parser.add_argument("--pre_mask_radg", type=float,
                    help="Lower bound for radgraph filtering", default=0)
parser.add_argument("--top_k", type=int)
np.random.seed(42)

args = parser.parse_args()
print(vars(args), flush=True)
if not args.do_chex and not args.do_radg:
  raise Exception("Must have at least radgraph or chexbert mode")

k = args.top_k
positive_list = []
statistics = {
  "Self is retrieved": 0,
  "A score of 2 is achieved": 0,
  "Diagonal Radgraph is not 1": 0,
  "DropSelf - didn't use self": 0,
  "DropSelf - used self": 0
}
bad_list = []

n = args.n
chunk_size = (n + args.num_chunks - 1) // args.num_chunks
processed = 0

for chunk_id in range(args.num_chunks):
  start = chunk_size * chunk_id
  end = min(chunk_size * (chunk_id + 1), n)

  # Obtain Statistics
  tensor = None
  chexbert = None
  radgraph = None
  if args.do_chex:
    chexbert = np.load(os.path.join(args.from_folder, f"chex_{chunk_id}.npy"))
    tensor = chexbert
    if chunk_id == 0:
      print("Loading Chexbert Tensor", flush=True)
  if args.do_radg:
    radgraph = np.load(os.path.join(
      args.from_folder, f"radg_{chunk_id}.npy"))
    if tensor is None:
      tensor = radgraph
      if chunk_id == 0:
        print("Loading Radgraph Tensor", flush=True)
    else:
      assert tensor.shape == radgraph.shape
      tensor = tensor + radgraph
      if chunk_id == 0:
        print("Adding Radgraph Tensor", flush=True)

  # Masking Operation before top-k
  ind = None  # index of top-k elements
  ind_mask = None  # if pre_mask, then this defines a mask of valid elements
  if not args.pre_mask:
    ind = np.argpartition(tensor, -k, axis=1)[:, -k:]
    ind_mask = np.ones_like(ind)
  else:
    ind = []
    assert args.do_chex
    assert args.do_radg
    chex_mask = chexbert >= args.pre_mask_chex
    radg_mask = radgraph >= args.pre_mask_radg
    mask = np.logical_and(chex_mask, radg_mask)
    # set things that don't satisfy conditions to -1
    masked_tensor = np.where(mask, tensor, -1)
    # things that don't satisfy filter will be the smallest
    ind = np.argpartition(masked_tensor, -k, axis=1)[:, -k:]
    # get a mask of all indices that yield a (radg + chex) != -1
    ind_mask = np.take_along_axis(masked_tensor, ind, axis=1) != -1

  # Filtering Rows
  for row_idx, (row, row_mask) in enumerate(zip(ind, ind_mask)):
    if start + row_idx in row:
      statistics["Self is retrieved"] += 1
    if np.allclose(np.max(tensor[row_idx, row]), 2):
      statistics["A score of 2 is achieved"] += 1
    # if diagonal radgraph is not zero, it's invalid. Skip sample
    if args.skip_bad_sample and \
            radgraph is not None and \
            not np.allclose(radgraph[row_idx][start + row_idx], 1):
      statistics["Diagonal Radgraph is not 1"] += 1
      bad_list.append(start + row_idx)
      positive_list.append([])
      continue
    # extract only the elements that are allowed by indicies mask
    extracted_row = row[np.nonzero(row_mask)[0]]
    positive_list.append(extracted_row.tolist())
  processed += len(ind)

statistics["n_set"] = n
statistics["n_actual"] = processed
print(json.dumps(statistics, indent=2), flush=True)
print(f"Bad List: {bad_list}", flush=True)

os.makedirs(args.output_folder, exist_ok=True)
print(f"Saving to: {args.output_folder}", flush=True)

mapped = {
  "statistics": statistics,
  "bad_list": bad_list
}
with open(os.path.join(args.output_folder, "positive_list.pkl"), "wb") as f:
  pickle.dump(positive_list, f)
with open(os.path.join(args.output_folder, "extraneous_data.pkl"), "wb") as f:
  pickle.dump(mapped, f)