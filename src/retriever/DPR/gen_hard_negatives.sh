python3 gen_hard_negatives.py  --query_embed_path /FactMM-RAG/DPR/embedding/train_embedding_image.pkl\
--txt_embed_path /FactMM-RAG/DPR/embedding/train_embedding_finding.pkl \
--result_path  /FactMM-RAG/src/retriever/ANCE/train_hard_negatives.pkl \
--query_path /home/liwens/healthcare/Lightning-Pretrain/chest/data/mimic/train_labeled.json \
--corpus_path /home/liwens/healthcare/Lightning-Pretrain/chest/data/mimic/train_labeled.json \
--chexbert_threshold 1 \
--radgraph_threshold 0.4 \
--topN 100

python3 gen_hard_negatives.py  --query_embed_path /FactMM-RAG/DPR/embedding/valid_embedding_image.pkl \
--txt_embed_path /FactMM-RAG/DPR/embedding/train_embedding_finding.pkl \
--result_path  /FactMM-RAG/src/retriever/ANCE/valid_hard_negatives.pkl \
--query_path /home/liwens/healthcare/Lightning-Pretrain/chest/data/mimic/valid_labeled.json \
--corpus_path /home/liwens/healthcare/Lightning-Pretrain/chest/data/mimic/train_labeled.json \
--chexbert_threshold 1 \
--radgraph_threshold 0.4 \
--topN 100