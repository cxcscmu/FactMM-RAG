#!/bin/bash

# Set default values
REF_PATH="./data/mimic/test.json"
PRED_PATH="./data/rag/llava_output/test/merge_test_eval.json"
PRED_PATH_INF="./data/rag/llava_output/test/merge_test_eval_inference.json" &&
DEVICE="cuda"
RADGRAPH_LEVEL="partial"
BERT_MODEL="distilbert-base-uncased"

# Run Python evaluation script
python src/generator/convert_json_or_jsonl.py --file ${PRED_PATH}l --overwrite &&
python src/generator/llava_json_to_evaluation_file.py --file ${PRED_PATH} &&
python src/evaluation.py --ref_path $REF_PATH \
               --pred_path $PRED_PATH_INF \
               --device $DEVICE \
               --radgraph_level $RADGRAPH_LEVEL \
               --bert_model $BERT_MODEL
