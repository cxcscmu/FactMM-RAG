import argparse
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm

def main(args):
    # Load test image embeddings
    with open(args.test_image_embedding_path, 'rb') as fin_test:
        test_image_embeddings = pickle.load(fin_test)

    # Load actual training data (JSON)
    with open(args.train_json_path, 'r') as fin_train_actual:
        train_actual = json.load(fin_train_actual)

    query_idx = list(range(len(test_image_embeddings)))
    test_image_embeddings = np.array(test_image_embeddings, dtype=np.float32)

    # Load train embeddings
    with open(args.train_embedding_path, 'rb') as fin_train:
        train_embedding = pickle.load(fin_train)

    # FAISS Index
    size = test_image_embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(size)
    cpu_index.add(np.array(train_embedding, dtype=np.float32))
    
    # Search for nearest neighbors
    D, I = cpu_index.search(test_image_embeddings, 3)

    # Retrieve findings & impressions
    ctx_findings_impressions = []
    for step, qid in enumerate(tqdm(query_idx)):
        cur = {"associated_impression": [], "retrieved_finding": []}
        for idx in I[step]:
            cur["associated_impression"].append(train_actual[idx]['impression'])
            cur["retrieved_finding"].append(train_actual[idx]['finding'])
        ctx_findings_impressions.append(cur)

    # Write all entries to JSON
    with open(args.output_json_path, 'w') as json_file:
        json.dump(ctx_findings_impressions, json_file, indent=4)

    print(f"Saved retrieved findings & impressions to {args.output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_image_embedding_path", type=str, required=True, 
                        help="Path to test image embeddings (pkl file)")
    parser.add_argument("--train_json_path", type=str, required=True, 
                        help="Path to training dataset JSON file")
    parser.add_argument("--train_embedding_path", type=str, required=True, 
                        help="Path to training embeddings (pkl file)")
    parser.add_argument("--output_json_path", type=str, required=True, 
                        help="Path to output JSON file")

    args = parser.parse_args()
    main(args)
