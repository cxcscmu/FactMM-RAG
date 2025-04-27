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
print(f"{vars(args)=}")

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


def extract_paths(image_path):
    patient, study, file = image_path.split("/")[-3:]
    return patient, study, file

def indexes(data_json):
    patient_index, study_index, _ = zip(*[extract_paths(v["image"]) for v in data_json])
    return patient_index, study_index

# These indexes are arrays, where arr[i] contains the patient id of patient_i
query_patient_index, query_study_index = indexes(query_data)
corpus_patient_index, corpus_study_index = indexes(corpus_data)


output = []

# Track how many times filters fail for top-1 result
ct_self_study = 0
ct_self_patient = 0
ct_short = 0
anomalies = []

for i, knn_sample_i in tqdm.tqdm(enumerate(faiss_knn_data), desc="queries..."):
    query_idx = knn_sample_i["key"]  # idx of query
    query_knn_rankings = knn_sample_i["knn_index"]
    query_data_i = query_data[query_idx]  # correspondent sample. Dict with keys "image", "finding", "impression"
    query_data_image_path = query_data_i["image"]
    patient_id, study_id, _ = extract_paths(query_data_image_path)

    chosen_document_idx = None
    for rank, document_idx in enumerate(query_knn_rankings):
        # make sure aren't self-study retrieving
        if corpus_study_index[document_idx] == study_id:
            # these things just check how often the nearest-neighbor fails a certain filter
            ct_self_study += rank == 0

        # make sure aren't self-patient retrieving
        elif corpus_patient_index[document_idx] == patient_id:
            ct_self_patient += rank == 0

        # findings sometimes have corrupted data, e.x "a.m." or "___"
        # test_short shouldn't be active for impression data mode, since impressions can be short normally
        elif test_short and len(corpus_data[document_idx][rag_data_mode].split()) < 5:
            ct_short += rank == 0

        # if all filters pass, yield the index
        else:
            chosen_document_idx = document_idx
            break
    
    # Choose first document if none pass filters
    if chosen_document_idx is None:
        anomalies.append(query_idx)
        chosen_document_idx = query_knn_rankings[0]
    
    retrieved_doc = corpus_data[chosen_document_idx][rag_data_mode]
    if is_conversational:
        obj = {
            "id": i,
            "image": query_data_image_path,
            "conversations": [
            {
                "from": "human",
                "value": f"Here is a report of a related patient: \"{retrieved_doc}\"\nGenerate a radiology report from this image:<image>"
            },
            {
                "from": "gpt",
                "value": f"{query_data_i[output_data_mode]}"
            }
            ]
        }
    else:
        obj = {
            "question_id": i,
            "image": query_data_image_path,
            "text": f"Here is a report of a related patient: \"{retrieved_doc}\"\nGenerate a radiology report from this image:"
        }
    output.append(obj)

# Anomalies are data samples that weren't able to yield valid results
print(f"Data Quality: ")
print(f"{ct_self_study=}, {ct_self_patient=}, {ct_short=}")
print(f"n={len(anomalies)} anomalies, occuring at query id's: {anomalies}")


print(f"Saving to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(os.path.join(output_path), "w") as f:
    json.dump(output, f, indent=2)
