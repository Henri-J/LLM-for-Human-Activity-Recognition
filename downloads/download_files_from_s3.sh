#!/usr/bin/env bash

# Create directories
mkdir -p ./dataset ./llama

# Download dataset
aws s3 cp s3://ml-har-bucket/dataset.zip .
unzip dataset.zip -d .

# Download all LLaMA model files
aws s3 cp s3://ml-har-bucket/config.json ./llama/
aws s3 cp s3://ml-har-bucket/generation_config.json ./llama/
aws s3 cp s3://ml-har-bucket/pytorch_model-00001-of-00002.bin ./llama/
aws s3 cp s3://ml-har-bucket/pytorch_model-00002-of-00002.bin ./llama/
aws s3 cp s3://ml-har-bucket/pytorch_model.bin.index.json ./llama/
aws s3 cp s3://ml-har-bucket/tokenizer.json ./llama/
aws s3 cp s3://ml-har-bucket/tokenizer.model ./llama/
aws s3 cp s3://ml-har-bucket/tokenizer_config.json ./llama/
