#!/bin/bash
# Download SentenceTransformer offline model to cache directory

echo "Downloading all-MiniLM-L6-v2 for offline usage..."

python3 -c "
from sentence_transformers import SentenceTransformer
import os

cache_dir = os.path.join(os.getcwd(), 'cache', 'all-MiniLM-L6-v2')
os.makedirs(cache_dir, exist_ok=True)
print(f'Downloading model to {cache_dir}')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save(cache_dir)
print('✅ Download successful!')
"

echo "Model downloaded and saved to ./cache/all-MiniLM-L6-v2"
