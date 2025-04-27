import argparse, json, os, pickle, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder", type=str, default="./data/mimic/scoring_chunks_train/top_30_c0.0_r0.0"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./data/rag/2025_03_09_end_to_end/knn_tr2tr_ideal.pkl",
)
parser.add_argument("--num_chunks", type=int, default=64)
args = vars(parser.parse_args())

folder, output_file = args["folder"], args["output_file"]
num_chunks = args["num_chunks"]

out = []

i = 0
for chunk in range(num_chunks):
    file = os.path.join(folder, f"chunk_{chunk}.pkl")
    with open(file, "rb") as f:
        obj = pickle.load(f)
    for row in obj["positive_list"]:
        if len(row) == 0:
            print(f"Missing on {i=}")
        out.append({"key": i, "knn_index": row})
        i += 1
print(len(out))
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(out, f)
