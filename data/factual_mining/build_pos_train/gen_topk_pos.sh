#!/bin/bash
#SBATCH --job-name="gen_pos_train.sh"
#SBATCH -o ./tr_reports/%x-%a.out
#SBATCH -e ./tr_reports/%x-%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-63%8

tr_chunks=64
chex_thresh=1.0
top_k=3
radg_thresh=0.4

start=$(date +%s)
expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}
echo "=== topk: $top_k, radg_thresh: $radg_thresh ==="
python gen_topk_pos.py \
    --from_folder "/FactMM-RAG/data/mimic/scoring_chunks_train" \
    --do_chex \
    --do_radg \
    --skip_bad_sample \
    --num_chunks $tr_chunks \
    --n 125417 \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --output_file "/FactMM-RAG/data/mimic/scoring_chunks_train/$expname/chunk_$SLURM_ARRAY_TASK_ID.pkl" \
    --pre_mask \
    --pre_mask_chex $chex_thresh \
    --pre_mask_radg $radg_thresh \
    --top_k $top_k

end=$(date +%s) &&
runtime=$((end-start)) &&
echo "Time Taken: $runtime s"