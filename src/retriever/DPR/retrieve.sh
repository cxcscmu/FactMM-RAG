#!/bin/bash

# Set paths
TEST_IMAGE_EMBEDDING_PATH="/FactMM-RAG/DPR/embedding/test_embedding_image.pkl"
TRAIN_JSON_PATH="/FactMM-RAG/data/mimic/train.json"
TRAIN_EMBEDDING_PATH="/FactMM-RAG/DPR/embedding/train_embedding_finding.pkl"
OUTPUT_JSON_PATH="/FactMM-RAG/src/retriever/DPR/retrieval/retrieved_reports.json"

# Run Python script
python retrieve.py --test_image_embedding_path $TEST_IMAGE_EMBEDDING_PATH \
                   --train_json_path $TRAIN_JSON_PATH \
                   --train_embedding_path $TRAIN_EMBEDDING_PATH \
                   --output_json_path $OUTPUT_JSON_PATH
