# [NAACL 2025] FactMM-RAG: Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation
In this work, we present FactMM-RAG, a fact-aware multimodal retrieval-augmented pipeline for generating accurate radiology reports. [[Paper Link](https://arxiv.org/abs/2407.15268)]

![Pipeline](assets/overview.png)

## 📅 Schedule

- [x] Release the data preprocessing code
- [x] Release the factual report pair mining code
- [x] Release the retriever training code
- [ ] Release the generator training code


## 📦 Requirements
1. Clone this repository and navigate to FactMM-RAG folder
```bash
git clone https://github.com/cxcscmu/FactMM-RAG.git
cd FactMM-RAG
```

2. Install Package: Create conda environment

```Shell
conda create -n FactMM-RAG python=3.10 -y
conda activate FactMM-RAG
pip install -r requirements.txt
```

3. Download the required dataset and checkpoint
   - Dataset: [MIMIC-CXR](https://vilmedic.app/papers/acl2023/) and [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
   - Checkpoint: [MARVEL](https://huggingface.co/OpenMatch/marvel-ance-clueweb/tree/main) 

## 📖 Data Preprocessing
1. Place the downloaded datasets in `./data/mimic` and `./data/chexpert`. We follow the official splitting and parse them into train, valid, and train files. To process the radiology dataset and generate the output JSON file, run the following command (e.g. train file parsing):
```sh
python ./data/parse.py --image_paths_file ./data/mimic/train.image.tok \
                 --findings_file ./data/mimic/train.findings.tok \
                 --impressions_file ./data/mimic/train.impression.tok \
                 --output_json_file ./data/mimic/train.json
```
2. Annotate reports with radiological entities, clinical relations, and diagnostic labels using RadGraph and CheXbert:
```sh
python ./data/label.py --input_path ./data/mimic/train.json \
                --output_path ./data/mimic/train_labeled.json \
                --device cuda   
```

## 📖 Factual Report Pairs Mining
1. Generate factual similarity scores using annotations from RadGraph and CheXbert. Before running the scripts, ensure that you update the data paths accordingly. Since the training corpus is large, we utilize parallel processing with SLURM array jobs for efficiency. Run the following commands:
```bash
#Query: training reports | Corpus: training reports
cd ./data/factual_mining/build_pos_train/
sbatch gen_similarity.sh
#Query: validation reports | Corpus: training reports
cd ./data/factual_mining/build_pos_valid/
sbatch gen_similarity.sh
```
2. Construct query and Top-K reference report pairs based on factual similarity thresholds. Run the following command:
```bash
cd ./data/factual_mining/build_pos_train/
sbatch gen_topk_pos.sh
sh merge_topk_pos.sh

cd ./data/factual_mining/build_pos_valid/
sh gen_topk_pos.sh
```

## 🚀 Training

1. Place the downloaded MARVEL ckpt into `./src/checkpoint/`. Train the multimodal retriever using constructed query-image and reference-report pairs, incorporating in-batch negative sampling. Additionally, an optional training stage with hard negatives can be included to further enhance performance. Run the following command:
```bash
cd ./src/retriever/DPR
sh train.sh
sh gen_embeddings.sh

#Optional ANCE Training
sh gen_hard_negatives.sh
cd ./src/retriever/ANCE
sh train.sh
sh gen_embeddings.sh
```
## 🚀 RAG

Using a trained retriever, follow instructions in `install_llava.sh` and also set the following environment variables

```bash
export IMAGE_FOLDER="path_to_image_folder"
export PROJECTOR_PATH="path_to_llava_projector"
```

Build the RAG training and test datasets using the generated query and document embeddings

```bash
python ./src/generator/build_rag_dataset.py \
  --faiss_knn_path ./data/rag/knn_te2tr.pkl \
  --queries_data_path ./data/mimic/test.json \
  --corpus_data_path ./data/mimic/train.json \
  --rag_data_mode finding \
  --output_data_mode finding \
  --test_short \
  --output_path ./data/rag/llava_data_te.json

python ./src/generator/build_rag_dataset.py \
  --faiss_knn_path ./data/rag/knn_tr2tr.pkl \
  --queries_data_path ./data/mimic/train.json \
  --corpus_data_path ./data/mimic/train.json \
  --rag_data_mode finding \
  --output_data_mode finding \
  --test_short \
  --is_conversational \
  --output_path ./data/rag/llava_data_tr.json

python src/generator/json_to_jsonl.py ./data/rag/llava_data_te.json
```

Then, train a LLaVA Model and run inference & scoring

```bash
./src/generator/train_llava.sh
./src/generator/inference_llava.sh
python src/generator/inference_jsonl_to_json.py ./data/rag/llava_output/test/merge_test_eval.jsonl
./src/generator/evaluate_llava.sh
```

## 📚Citation
```bibtex
@misc{sun2025factawaremultimodalretrievalaugmentation,
      title={Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation}, 
      author={Liwen Sun and James Zhao and Megan Han and Chenyan Xiong},
      year={2025},
      eprint={2407.15268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.15268}, 
}
```

## 🙏Acknowledgement
We use code from [LLaVA](https://github.com/haotian-liu/LLaVA) and [MARVEL](https://github.com/OpenMatch/MARVEL). We thank the authors for releasing their code.