cd FactMM-RAG
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# Follow install instructions from
# https://github.com/haotian-liu/LLaVA.git
# at the current time, commit hash c121f0432da27facab705978f83c4ada465e46fd
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation
# If flash-attn gets 404 error, set environment variable export `HUGGINGFACE_CO_TIMEOUT=60`


# some additional upgrades
pip install transformers[torch]
pip install peft==0.10.0 transformers==4.36.2 accelerate==0.21.0 tokenizers==0.15.1