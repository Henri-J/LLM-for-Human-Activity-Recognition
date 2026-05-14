#!/usr/bin/env bash

mkdir -p llama

REPO="huggyllama/llama-7b"
FILES=(
    "config.json"
    "tokenizer.json"
		"generation_config.json"
		"tokenizer.model"
		"tokenizer_config.json"
		"pytorch_model-00001-of-00002.bin"
		"pytorch_model-00002-of-00002.bin"
		"pytorch_model.bin.index.json"
)

for FILE in "${FILES[@]}"; do
    URL="https://huggingface.co/$REPO/resolve/main/$FILE"
    echo "Downloading $FILE..."
    curl -L "$URL" -o "./llama/$FILE" --fail
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $FILE"
    else
        echo "Failed to download $FILE" >&2
    fi
done
