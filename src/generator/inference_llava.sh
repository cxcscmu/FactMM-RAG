cd LLaVA
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llava

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
TEST_PATH=../data/rag
CKPT=llava_output
SPLIT="test"
CHUNKS=${#GPULIST[@]}

Q_FILE=../data/rag/llava_data_te.jsonl
OUTPUT_FILE=$TEST_PATH/$CKPT/$SPLIT/merge_test_eval.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    mkdir -p $TEST_PATH/$CKPT/$SPLIT
    touch $TEST_PATH/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl
done

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $TEST_PATH/$CKPT \
        --question-file $Q_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $TEST_PATH/$CKPT/$SPLIT/$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

# Clear out the output file if it exists.
> "$OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $TEST_PATH/$CKPT/$SPLIT/$CHUNKS_$IDX.jsonl >> "$OUTPUT_FILE"
done