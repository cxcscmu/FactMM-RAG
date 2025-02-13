#!/bin/bash

# Set default values for arguments
T5_MODEL_NAME="OpenMatch/t5-ance"
CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
SAVED_CKPT="/FactMM-RAG/src/retriever/output/ance.best.pt"

TRAIN_PATH="/FactMM-RAG/data/mimic/train.json"
VALID_PATH="/FactMM-RAG/data/mimic/valid.json"
TEST_PATH="/FactMM-RAG/data/mimic/test.json"

OUTPUT_TRAIN_IMAGE_PATH="/FactMM-RAG/ANCE/embedding/train_embedding_image.pkl"
OUTPUT_TRAIN_FINDING_PATH="/FactMM-RAG/ANCE/embedding/train_embedding_finding.pkl"
OUTPUT_VALID_IMAGE_PATH="/FactMM-RAG/ANCE/embedding/valid_embedding_image.pkl"
OUTPUT_TEST_IMAGE_PATH="/FactMM-RAG/ANCE/embedding/test_embedding_image.pkl"

# Run Python script with arguments
python gen_embeddings.py --t5_model_name $T5_MODEL_NAME \
                 --clip_model_name $CLIP_MODEL_NAME \
                 --saved_ckpt $SAVED_CKPT \
                 --train_path $TRAIN_PATH \
                 --valid_path $VALID_PATH \
                 --test_path $TEST_PATH \
                 --output_train_image_path $OUTPUT_TRAIN_IMAGE_PATH \
                 --output_train_finding_path $OUTPUT_TRAIN_FINDING_PATH \
                 --output_valid_image_path $OUTPUT_VALID_IMAGE_PATH \
                 --output_test_image_path $OUTPUT_TEST_IMAGE_PATH
