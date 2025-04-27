import numpy as np
import pickle
import os
import json
import argparse
import sys
import tqdm

parser = argparse.ArgumentParser(
    description="Takes in an index file, and produces a llava-formatted dataset."
)
# pickle file of embeddings (output from marvel ance). obj[1] should be the array of embeddings
parser.add_argument("--queries_data_path", type=str, default="./data/mimic/test.json")
parser.add_argument(
    "--is_conversational",
    action="store_true",
    help="""Set to true if used for llava training / evaluation, Set to False if used for inference""",
)
parser.add_argument(
    "--output_data_mode",
    type=str,
    default="finding",
    help="the data type that is used for ground truth",
)
parser.add_argument("--output_path", type=str, default="./data/rag/vqa/vqa_test.json")
args = parser.parse_args()
print(f"{vars(args)=}")

queries_data_path = args.queries_data_path
is_conversational = args.is_conversational
output_path = args.output_path
output_data_mode = args.output_data_mode

with open(queries_data_path, "r") as f_que:
    query_data = json.load(f_que)
print(f"{len(query_data)=}", flush=True)

output = []

for i, knn_sample_i in tqdm.tqdm(enumerate(query_data), desc="queries..."):
    query_data_i = query_data[
        i
    ]  # correspondent sample. Dict with keys "image", "finding", "impression"
    query_data_image_path = query_data_i["image"]

    if is_conversational:
        obj = {
            "id": i,
            "image": query_data_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"Generate a radiology report from this image:<image>",
                },
                {"from": "gpt", "value": f"{query_data_i[output_data_mode]}"},
            ],
        }
    else:
        obj = {
            "question_id": i,
            "image": query_data_image_path,
            "text": f"\nGenerate a radiology report from this image:",
        }
    output.append(obj)

print(f"Saving to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(os.path.join(output_path), "w") as f:
    json.dump(output, f, indent=2)
