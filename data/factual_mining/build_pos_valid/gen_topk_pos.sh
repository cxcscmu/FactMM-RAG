#!/bin/bash
#SBATCH --job-name="gen_merge_pos_valid.sh"
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

va_chunks=4
chex_thresh=1.0
top_k=3
radg_thresh=0.4

echo "$SLURM_NODELIST" &&

start=$(date +%s)

expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}
echo "=== topk: $topk, radg_thresh: $radg_thresh ==="
python gen_topk_pos.py \
    --from_folder "FactMM-RAG/data/mimic/scoring_chunks_valid" \
    --do_chex \
    --do_radg \
    --num_chunks $va_chunks \
    --n 991 \
    --output_folder "FactMM-RAG/data/mimic/scoring_chunks_valid/$expname" \
    --pre_mask \
    --pre_mask_chex $chex_thresh \
    --pre_mask_radg $radg_thresh \
    --top_k $top_k

end=$(date +%s) &&
runtime=$((end-start)) &&
echo "Time Taken: $runtime s"
