#!/bin/bash

# Set default values
REF_PATH="/FactMM-RAG/data/mimic/test.json"
PRED_PATH="/FactMM-RAG/data/mimic/test_generated.json"
DEVICE="cuda"
RADGRAPH_LEVEL="partial"
BERT_MODEL="distilbert-base-uncased"

# Run Python evaluation script
python eval.py --ref_path $REF_PATH \
               --pred_path $PRED_PATH \
               --device $DEVICE \
               --radgraph_level $RADGRAPH_LEVEL \
               --bert_model $BERT_MODEL
