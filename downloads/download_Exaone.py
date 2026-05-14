from huggingface_hub import snapshot_download

# Download all files into a folder named "Exaone"
snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-4.0-1.2B",
    local_dir="Exaone",  # Folder where files will be saved
    local_dir_use_symlinks=False,  # Download actual files (not symlinks)
    ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.onnx"],  # Skip unnecessary files (optional)
)
