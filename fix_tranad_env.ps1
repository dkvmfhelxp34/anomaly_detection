# Run inside Anaconda Prompt (Windows)

# 1) Remove mismatched CPU/legacy packages
conda run -n tranad pip uninstall -y torch torchvision torchaudio

# 2) Install matching CUDA builds (PyTorch cu121)
conda run -n tranad pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 3) Optional: resolve pip check warnings
conda run -n tranad pip install smmap ansicon

# 4) Verify
conda run -n tranad python codex_scr/check_env.py
