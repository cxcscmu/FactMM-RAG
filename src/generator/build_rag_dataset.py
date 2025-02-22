import numpy as np
import pickle
import os
import json
import argparse
import sys
import tqdm

parser = argparse.ArgumentParser(
  description="Takes in an index file, and produces a llava-formatted dataset.")
# pickle file of embeddings (output from marvel ance). obj[1] should be the array of embeddings
parser.add_argument('--faiss_knn_path', type=str)
parser.add_argument('--queries_data_path', type=str)
parser.add_argument('--corpus_data_path', type=str)
parser.add_argument('--rag_data_mode', type=str, default="finding",
                    help="the data type that is plugged into the input's prompt")
parser.add_argument('--output_data_mode', type=str, default="finding",
                    help="the data type that is used for ground truth")
parser.add_argument('--test_short', action="store_true",
                    help="Skip KNN on things that are short (< 5 words)")
parser.add_argument('--is_conversational', action="store_true",
                    help="""Set to true if used for llava training / evaluation, Set to False if used for inference""")
parser.add_argument('--output_path', type=str)
args = parser.parse_args()
print(vars(args))
sys.stdout.flush()
faiss_knn_path = args.faiss_knn_path
queries_data_path, corpus_data_path = args.queries_data_path, args.corpus_data_path
rag_data_mode, output_data_mode = args.rag_data_mode, args.output_data_mode
test_short = args.test_short
is_conversational = args.is_conversational
output_path = args.output_path

with \
  open(queries_data_path, "r") as f_que, \
  open(corpus_data_path, "r") as f_cor, \
  open(faiss_knn_path, "rb") as f_knn:
    corpus_data = json.load(f_cor)
    query_data = json.load(f_que)
    faiss_knn_data = pickle.load(f_knn)
print(f"{len(corpus_data)=} {len(query_data)=}")
sys.stdout.flush()


def extract_patient(image_path):
    print(image_path)
    return image_path.split("/")[-3]


def extract_study(image_path):
    return image_path.split("/")[-3]


def idx_to_patient_index(data_json):
    return [extract_patient(v["image"]) for v in data_json]


def idx_to_study_index(data_json):
    return [extract_study(v["image"]) for v in data_json]


query_patient_index = idx_to_patient_index(query_data)
query_study_index = idx_to_study_index(query_data)
corpus_patient_index = idx_to_patient_index(corpus_data)
corpus_study_index = idx_to_study_index(corpus_data)


def build_dataset(knn_query_data, query_data, corpus_data,
                  corpus_patient_index, corpus_study_index, is_conversational=True):
    output = []
    ct_patient = 0
    ct_self_study = 0
    ct_short = 0
    anomalies = []
    for i, v in tqdm.tqdm(enumerate(knn_query_data)):
        yielded_index = None
        knn_self_key = v["key"]  # idx of train_data
        knn_data_i = query_data[knn_self_key]  # correspondent sample
        patient_id = extract_patient(knn_data_i["image"][0])
        study_id = extract_study(knn_data_i["image"][0])
        for iteration_order, knn_idx in enumerate(v["knn_index"]):
            # make sure aren't self-retrieving
            if corpus_study_index[knn_idx] == study_id:
                # these things just check how often the nearest-neighbor fails a certain filter
                if iteration_order == 0:
                    ct_self_study += 1
            # make sure aren't self-patient retrieving
            elif corpus_patient_index[knn_idx] == patient_id:
                if iteration_order == 0:
                    ct_patient += 1
            # findings sometimes have corrupted data, e.x "a.m." or "___"
            # test_short shouldn't be active for impression data mode, since impressions
            # can be short normally
            elif test_short and len(corpus_data[knn_idx][rag_data_mode].split()) < 5:
                if iteration_order == 0:
                    ct_short += 1
            # if all filters pass, yield the index
            else:
                yielded_index = knn_idx
                break
        if yielded_index is None:
            anomalies.append(knn_self_key)
            yielded_index = v["knn_index"][0]
        if is_conversational:
            obj = {
              "id": i,
              "image": knn_data_i["image"][0],
              "conversations": [
                {
                  "from": "human",
                  "value": f"Here is a report of a related patient: \"{corpus_data[yielded_index][rag_data_mode]}\"\nGenerate a radiology report from this image:<image>"
                },
                {
                  "from": "gpt",
                  "value": f"{knn_data_i[output_data_mode]}"
                }
              ]
            }
        else:
            obj = {
              "question_id": i,
              "image": knn_data_i["image"][0],
              "text": f"Here is a report of a related patient: \"{corpus_data[yielded_index][rag_data_mode]}\"\nGenerate a radiology report from this image:"
            }
        output.append(obj)
    print(f"{ct_patient=}, {ct_self_study=}, {ct_short=}")
    # Anomalies are data samples that weren't able to yield valid results
    print(f"(n={len(anomalies)}) {anomalies=}")
    return output


dataset = build_dataset(faiss_knn_data, query_data, corpus_data,
                        corpus_patient_index, corpus_study_index, is_conversational=is_conversational)

print(f"Saving to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(os.path.join(output_path), "w") as f:
    json.dump(dataset, f, indent=2)
