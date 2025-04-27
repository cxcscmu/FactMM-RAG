import faiss
import pickle
import argparse
import numpy as np
import json
import sys
import os

faiss.omp_set_num_threads(16)

parser = argparse.ArgumentParser(
  description="""
  Constructs a knn-index file. For every query embedding, 
  obtains the related knn indices from the training corpus.
  """)
# pickle file of embeddings (output from marvel ance). obj[1] should be the array of embeddings
parser.add_argument('--query_embedding_file', type=str,
                    help="""
                    A pickle file containing embeddings. In format
                    {
                      idx: {
                        "image_embedding": nd.array(d),
                        "finding_embedding": nd.array(d),
                        "impression_embedding": nd.array(d)
                      }
                    }
                    
                    Also can be a tensor of size: (n, d)
                    """)
parser.add_argument('--corpus_embedding_file', type=str)
parser.add_argument('--query_data_file', type=str,
                    help="""
                    A path to a json file containing data in format, likely train.json:
                    [
                      {
                        "image": str,
                        "finding": str,
                        "impression": str
                      }
                    ]""", default="./data/mimic/train.json")
parser.add_argument('--corpus_data_file', type=str,
                    help="The json path to corpus data (probably train.json)", default="./data/mimic/test.json")
parser.add_argument("--query_key", type=str, default=None,
                    help="The key to use to lookup in query_embedding_file. Ignored if query is a numpy array")
parser.add_argument("--corpus_key", type=str, default=None,
                    help="The key to use to lookup in corpus_embedding_file. Ignored if query is a numpy array")
parser.add_argument('--nlist', type=int, default=100,
                    help="faiss nlist value. If -1, will do dense indexing")
parser.add_argument("--nprobe", type=int, default=10, help="faiss nprobe value")
parser.add_argument("--k", type=int, default=20,
                    help="faiss 'k'-nearest-neighbor value")
parser.add_argument("--results_k", type=int, default=5,
                    help="how many (k) text results (e.x. finding / impression) from corpus to save")
parser.add_argument("--data_type", type=str, default=None,
                    help="the type of data (e.x. query's finding) that is output in the resulting file")
parser.add_argument("--output_path", type=str,
                    help="output json file path", required=True)


args = parser.parse_args()

print("args", vars(args))


def process_pickle(loaded_object, key_type=None):
    if type(loaded_object) == np.ndarray:
        print(f"Loaded an ndarray of shape: {loaded_object.shape}")
        keys = list(range(len(loaded_object)))
        embeddings = loaded_object
    else:
        print(f"Assuming loaded a dictionary")
        assert key_type
        keys, embeddings = zip(
            *[(i, v[key_type]) for i, v in loaded_object.items()])
        keys, embeddings = list(keys), np.vstack(embeddings)
    return keys, embeddings


with open(args.query_embedding_file, "rb") as f_query_embedding_file, \
        open(args.corpus_embedding_file, "rb") as f_corpus_embedding_file, \
        open(args.query_data_file, "r") as f_query_data_file, \
        open(args.corpus_data_file, "r") as f_corpus_data_file:
    query_embeddings = pickle.load(f_query_embedding_file)
    corpus_embeddings = pickle.load(f_corpus_embedding_file)
    corpus_keys, corpus_embeddings = process_pickle(
      corpus_embeddings, args.corpus_key)
    query_keys, query_embeddings = process_pickle(
      query_embeddings, args.query_key)
    query_data = json.load(f_query_data_file)
    corpus_data = json.load(f_corpus_data_file)

    print(
        f"Query Keys Sequential({len(query_keys)})? {query_keys == list(range(len(query_keys)))}")
    print(
      f"Corpus Keys Sequential({len(corpus_keys)})? {corpus_keys == list(range(len(corpus_keys)))}")
    print(
      f"Query Embedding Keys match range? {set(query_keys) == set(range(len(query_keys)))}")
    print(
      f"Corpus Embedding Keys match range? {set(corpus_keys) == set(range(len(corpus_keys)))}")
    print(f"{query_embeddings.shape=} {corpus_embeddings.shape=}")

d = query_embeddings.shape[1]
if args.nlist == -1:
    print(f"Using Exhaustive Search, {args.k=}")
    cpu_index = faiss.IndexFlatIP(d)
    cpu_index.add(corpus_embeddings)
    D, I = cpu_index.search(query_embeddings, args.k)
else:
    print(f"Using IVFFlat, {args.nlist=}, {args.nprobe=}, {args.k=}")
    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, args.nlist)
    cpu_index.nprobe = args.nprobe
    cpu_index.train(corpus_embeddings)
    cpu_index.add(corpus_embeddings)
    D, I = cpu_index.search(query_embeddings, args.k)

output = []
for query_number, query_key in enumerate(query_keys):
    obj = {
      "key": query_key,
      "knn_index": I[query_number],
      "similarities": D[query_number],
    }
    if args.data_type:
        obj[args.data_type] = query_data[query_key][args.data_type],
    if args.results_k > 0 and args.data_type:
        qty = min(args.k, args.results_k)
        obj["results"] = [corpus_data[corpus_keys[corpus_index]][args.data_type]
                          for corpus_index in I[query_number][:qty]]
    output.append(obj)

print(f"Saving to: {args.output_path}. Items: {len(output)}")
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "wb") as f:
    pickle.dump(output, f)
