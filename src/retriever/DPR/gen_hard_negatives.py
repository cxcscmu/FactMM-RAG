from tqdm import tqdm
import pickle
import faiss
import pytrec_eval
import argparse
import json
import numpy as np
import pandas as pd
import numpy as np
import pickle
from IPython import embed
import os
from utils import exact_entity_token_if_rel_exists_reward
from collections import defaultdict
def chexbert_similarity(report,ret_report):
    report_label = report['label']
    ret_report_label = ret_report['label']
    # distance = manhattan_distance(report_label, ret_report_label)
    # # Calculate the similarity as the inverse of the distance plus one
    # # to prevent division by zero when the distance is zero.
    return sum(1 for true, pred in zip(report_label, ret_report_label) if true == pred)/len(report_label)
    
    
def radgraph_similarity(report,ret_report):
    report_entities = report['entities']
    ret_report_entities = ret_report['entities']
    partial_reward = exact_entity_token_if_rel_exists_reward(ret_report_entities,report_entities)
    return partial_reward






if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_path")
    parser.add_argument("--corpus_path")
    parser.add_argument("--query_embed_path")
    parser.add_argument("--txt_embed_path")
    parser.add_argument("--chexbert_threshold",type=float,default=1)
    parser.add_argument("--radgraph_threshold",type=float,default=0.4)
    parser.add_argument("--result_path")
    parser.add_argument("--topN",type=int,default=100)
    parser.add_argument("--num_top_neg",type=int,default=2)
    
    
    args = parser.parse_args()
    faiss.omp_set_num_threads(16)

    

    with open(args.query_embed_path, 'rb') as fin:
        query_embeds = pickle.load(fin)
        query_embeds = np.array(query_embeds, np.float32)

    cpu_index = faiss.IndexFlatIP(query_embeds.shape[1])
         
 
    print("load data from {}".format(args.txt_embed_path))
    with open(args.txt_embed_path, 'rb') as fin:
        txt_embeds = pickle.load(fin)
    cpu_index.add(np.array(txt_embeds, np.float32))


    with open(args.query_path,"r") as f:
        query_data = json.load(f)
    with open(args.corpus_path,"r") as f:
        corpus_data = json.load(f)
    
    
    D, I = cpu_index.search(query_embeds, args.topN)    
    
    query_hard_negative_id = []
    for qid, query_results in tqdm(enumerate(I)):
        cur_query_hard_negative_id = []
        for ret_id in query_results:
            if ret_id != qid :
                if chexbert_similarity(query_data[qid],corpus_data[ret_id]) < args.chexbert_threshold and radgraph_similarity(query_data[qid],corpus_data[ret_id]) < args.radgraph_threshold:
                    cur_query_hard_negative_id.append((chexbert_similarity(query_data[qid],corpus_data[ret_id])+radgraph_similarity(query_data[qid],corpus_data[ret_id]),ret_id))      
        cur_query_hard_negative_id.sort(key = lambda x:x[0])            
        query_hard_negative_id.append([  ret_idx for _,ret_idx in cur_query_hard_negative_id[:args.num_top_neg]])      
    del cpu_index

    print("Save file!")
    pickle.dump(query_hard_negative_id,open(args.result_path,'wb'))
    
    
    
    
    