#!/bin/bash

# Set default values
REF_PATH="./data/mimic/test.json"
PRED_PATH="./data/rag/llava_output/test/merge_test_eval.json"
DEVICE="cuda"
RADGRAPH_LEVEL="partial"
BERT_MODEL="distilbert-base-uncased"

# Run Python evaluation script
python src/evaluation.py --ref_path $REF_PATH \
               --pred_path $PRED_PATH \
               --device $DEVICE \
               --radgraph_level $RADGRAPH_LEVEL \
               --bert_model $BERT_MODEL
