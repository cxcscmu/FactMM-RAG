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

import re

def compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MaxMRRRank=10):
    MRR = 0
    ranking = []
    for qid in tqdm(qids_to_ranked_candidate_passages,desc=f"Evaluate MRR@{MaxMRRRank}"):
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_relevant_passageids)
    return MRR


def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def EvalDevQuery(query_positive_id, ctx_idxs):
    prediction = {}  # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)


    qids_to_ranked_candidate_passages = {}
    for query_id, top_pid in tqdm(ctx_idxs.items(), total=len(ctx_idxs),desc="Convert prediction results"):
        prediction[query_id] = {}
        rank = 0

        tmp = [0] * 1000
        qids_to_ranked_candidate_passages[query_id] = tmp

        for idx in top_pid:
            pred_pid = idx
            qids_to_ranked_candidate_passages[query_id][rank] = pred_pid
            rank += 1
            prediction[query_id][pred_pid] = -rank


    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(query_positive_id), {'ndcg_cut', 'recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))

    qids_to_relevant_passageids = {}
    for qid in tqdm(query_positive_id,desc="Convert ground truth results"):
        qids_to_relevant_passageids[qid] = []
        for pid in query_positive_id[qid]:
            qids_to_relevant_passageids[qid].append(pid)


    
    # Initialize MRR values
    mrr_100 = compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 100)
    mrr_200 = compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 200)
    mrr_500 = compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 500)
    mrr_1000 = compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, 1000)
        
    # Initialize NDCG and Recall values
    recall_100 = 0
    recall_200 = 0
    recall_500 = 0
    recall_1000 = 0
    
    ndcg_100 = 0
    ndcg_200 = 0
    ndcg_500 = 0
    ndcg_1000 = 0
    
    for k in tqdm(result.keys(), desc="Report results"):
        eval_query_cnt += 1
        
        recall_100 += result[k]["recall_100"]
        recall_200 += result[k]["recall_200"]
        recall_500 += result[k]["recall_500"]
        recall_1000 += result[k]["recall_1000"]
        
        ndcg_100 += result[k]["ndcg_cut_100"]
        ndcg_200 += result[k]["ndcg_cut_200"]
        ndcg_500 += result[k]["ndcg_cut_500"]
        ndcg_1000 += result[k]["ndcg_cut_1000"]
    
    # Calculate average values
    recall_100 /= eval_query_cnt
    recall_200 /= eval_query_cnt
    recall_500 /= eval_query_cnt   
    recall_1000 /= eval_query_cnt
    
    ndcg_100 /= eval_query_cnt
    ndcg_200 /= eval_query_cnt
    ndcg_500 /= eval_query_cnt
    ndcg_1000 /= eval_query_cnt
    
    return recall_100, recall_200, recall_500, recall_1000, mrr_100, mrr_200, mrr_500, mrr_1000, ndcg_100, ndcg_200, ndcg_500, ndcg_1000





if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_embed_path")
    parser.add_argument("--txt_embed_path")
    parser.add_argument("--query_positive_matrix_path")    
    parser.add_argument("--chexbert_threshold",type=float,default=1)
    parser.add_argument("--radgraph_threshold",type=float,default=0.4)
    parser.add_argument("--result_path")
    parser.add_argument("--topN",type=int,default=1000)



    args = parser.parse_args()
    faiss.omp_set_num_threads(16)

    

    with open(args.query_embed_path, 'rb') as fin:
        query_embeds = pickle.load(fin)
        query_embeds = np.array(query_embeds, np.float32)

    cpu_index = faiss.IndexFlatIP(query_embeds.shape[1])
         
    if args.txt_embed_path:
        print("load data from {}".format(args.txt_embed_path))
        with open(args.txt_embed_path, 'rb') as fin:
            txt_embeds = pickle.load(fin)
        cpu_index.add(np.array(txt_embeds, np.float32))
        model_name = "image2finding"

            

    query_positive_matrix = pickle.load(open(args.query_positive_matrix_path,"rb"))


    D, I = cpu_index.search(query_embeds, args.topN)
    ctx_idxs = {}
    query_positive_id = {}
    for qid, np_query_results in enumerate(I):
        query_results = np_query_results.tolist()        

        ctx_idxs[qid] = query_results
    
    del cpu_index
    
 

    for qid, query_positive_results in enumerate(query_positive_matrix):
        query_positive_id.setdefault(qid, {})
        for ret_id in query_positive_results:
            query_positive_id[qid][ret_id] = 1
            
        if len(query_positive_id[qid]) == 0:
            del query_positive_id[qid]
            del ctx_idxs[qid]
    
    
            
    
    result = EvalDevQuery(query_positive_id, ctx_idxs)
    recall_100, recall_200,recall_500,recall_1000, mrr_100, mrr_200,mrr_500,mrr_1000, ndcg_100, ndcg_200,ndcg_500,ndcg_1000 = result
    
      
    result_path = os.path.join(args.result_path,f"chexbert_{args.chexbert_threshold}_radgraph_{args.radgraph_threshold}_top1000.txt")
        
    if not os.path.exists(result_path):
        with open(result_path, 'w') as fout:
            fout.write("Model Name\tRecall@100\tRecall@200\tRecall@500\tRecall@1000\tMRR@100\tMRR@200\tMRR@500\tMRR@1000\tNDCG@100\tNDCG@200\tNDCG@500\tNDCG@1000\n")
    with open(result_path, 'a') as fout:  
        fout.write(f"{model_name}\t"
                f"{recall_100 * 100:.2f}\t"
                f"{recall_200 * 100:.2f}\t"
                f"{recall_500 * 100:.2f}\t"
                f"{recall_1000 * 100:.2f}\t"
                f"{mrr_100 * 100:.2f}\t"
                f"{mrr_200 * 100:.2f}\t"
                f"{mrr_500 * 100:.2f}\t"
                f"{mrr_1000 * 100:.2f}\t"
                f"{ndcg_100 * 100:.2f}\t"
                f"{ndcg_200 * 100:.2f}\t"
                f"{ndcg_500 * 100:.2f}\t"
                f"{ndcg_1000 * 100:.2f}\n")
        
    print(mrr_100)