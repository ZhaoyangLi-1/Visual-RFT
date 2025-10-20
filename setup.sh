pip uninstall -y transformers huggingface-hub
pip install "transformers>=4.51.0,<5.0" "huggingface-hub>=0.23,<1.0"
cd src/virft
pip install -e ".[dev]"

# Additional modules
pip install wandb
pip install tensorboardx
pip install qwen_vl_utils torchvision

pip uninstall -y torch torchvision torchaudio

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -v flash-attn --no-build-isolation

