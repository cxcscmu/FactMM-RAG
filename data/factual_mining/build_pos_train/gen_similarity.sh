#!/bin/bash
#SBATCH --job-name="gen_sim_train"
#SBATCH -o %x-%a.out
#SBATCH -e %x-%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-63

python3 gen_similarity.py \
  --train_data_file "/FactMM-RAG/data/mimic/train_labeled.json" \
  --output_folder "/FactMM-RAG/data/mimic/scoring_chunks_train" \
  --num_chunks 64 \
  --chunk_id $SLURM_ARRAY_TASK_ID


  