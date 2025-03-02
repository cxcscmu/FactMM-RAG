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

# some additional upgrades
pip install transformers[torch]