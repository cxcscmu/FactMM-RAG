#!/bin/bash
#SBATCH --job-name="merge_pos_train.sh"
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00


top_k=3
tr_chunks=64
chex_thresh=1.0
radg_thresh=0.4
expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}

echo "===  Reduce ${expname} (tr) ==="
echo "$SLURM_NODELIST" &&

start=$(date +%s)

expname=top_${top_k}_c${chex_thresh}_r${radg_thresh}
echo "=== topk: $topk, radg_thresh: $radg_thresh ==="

python merge_topk_pos.py \
    --from_folder "/FactMM-RAG/data/mimic/scoring_chunks_train/$expname" \
    --file_prefix "chunk_{i}.pkl" \
    --num_chunks $tr_chunks

end=$(date +%s) &&
runtime=$((end-start)) &&
echo "Time Taken: $runtime s"