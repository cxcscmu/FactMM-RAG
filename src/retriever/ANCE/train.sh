
chex_thresh=1.0
top_k=3
radg_thresh=0.4

expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}

python train.py  --out_path /FactMM-RAG/src/retriever/ANCE/output/ance.best.pt \
--train_path /FactMM-RAG/data/mimic/train.json \
--valid_path /FactMM-RAG/data/mimic/valid.json \
--train_pos_path /FactMM-RAG/data/mimic/scoring_chunks_train/$expname/reduction.pkl \
--valid_pos_path /FactMM-RAG/data/mimic/scoring_chunks_valid/$expname/positive_list.pkl \
--train_neg_path  /FactMM-RAG/src/retriever/ANCE/train_hard_negatives.pkl \
--valid_neg_path  /FactMM-RAG/src/retriever/ANCE/valid_hard_negatives.pkl \
--pretrained_model_path /FactMM-RAG/src/retriever/DPR/output/dpr.best.pt \
--wandb_name "Finetuning with positives from exhaustive with reranked top_$top_k chexbert $chex_thresh radgraph $radg_thresh ance"
# --freeze_vision_model
