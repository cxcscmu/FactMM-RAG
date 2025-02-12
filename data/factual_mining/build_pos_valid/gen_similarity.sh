#!/bin/bash
#SBATCH --job-name="gen_sim_valid"
#SBATCH -o %x-%a.out
#SBATCH -e %x-%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-4

python3 gen_similarity_valid.py \
  --query_data_file "FactMM-RAG/data/mimic/valid_labeled.json" \
  --corpus_data_file "FactMM-RAG/data/mimic/train_labeled.json" \
  --output_folder "FactMM-RAG/data/mimic/scoring_chunks_valid" \
  --num_chunks 4 \
  --chunk_id $SLURM_ARRAY_TASK_ID


  