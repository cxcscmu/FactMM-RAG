import argparse
import json
import pickle
import os
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_file", type=str, default="./data/mimic/train.json")
parser.add_argument("--knn_index", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--is_test", action="store_true")
args = vars(parser.parse_args())

train_data_file = args["train_data_file"]
knn_index, output_file = args["knn_index"], args["output_file"]
is_test = args["is_test"]

with open(train_data_file, "r") as tr, open(knn_index, "rb") as knn:
    train_data = json.load(tr)
    knn_data = pickle.load(knn)

output = [
    {
        "retrieved_finding": [train_data[knn_data_i["knn_index"][0]]["finding"]]
    } for knn_data_i in knn_data
]


with open(output_file, "w") as f:
    json.dump(output, f, indent=2)
    print(f"Writing {len(output)} to {output_file}")