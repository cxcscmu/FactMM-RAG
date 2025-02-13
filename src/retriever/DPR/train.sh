#!/bin/bash
#SBATCH --job-name=healthcare
#SBATCH --output=healthcare_1.0_0.4_clueweb.out
#SBATCH --error=healthcare_1.0_0.4_clueweb.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00


chex_thresh=1.0
top_k=3
radg_thresh=0.4

expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}

python train.py  --out_path /FactMM-RAG/src/retriever/output/dpr.best.pt \
--train_path /FactMM-RAG/data/mimic/train.json \
--valid_path /FactMM-RAG/data/mimic/valid.json \
--train_pos_path /FactMM-RAG/data/mimic/scoring_chunks_train/$expname/reduction.pkl \
--valid_pos_path /FactMM-RAG/data/mimic/scoring_chunks_valid/$expname/positive_list.pkl \
--wandb_name "Finetuning with positives from exhaustive with reranked top_$top_k chexbert $chex_thresh radgraph $radg_thresh" \
--pretrained_model_path /FactMM-RAG/src/retriever/checkpoint/model.best.pt
