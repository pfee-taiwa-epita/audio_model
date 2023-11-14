
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
import os
import pandas as pd
import torchaudio

# Step 1: Download the dataset
base_path = snapshot_download(repo_id="PFEE-TxE/audio_sampler", repo_type="dataset")

# Define the paths for each sub-folder
accacia_path = os.path.join(base_path, 'data/accacia')
bouleau_path = os.path.join(base_path, 'data/bouleau')
chene_path = os.path.join(base_path, 'data/chene')
sapin_path = os.path.join(base_path, 'data/sapin')

# Load the metadata
metadata_df = pd.read_csv(os.path.join(base_path, 'data/data.csv'))
metadata_df['file_path'] = metadata_df.apply(lambda row: os.path.join(base_path, 'data', row['label'], row['filename']), axis=1)
dataset_df = metadata_df[metadata_df['is_in_dataset']]
print(dataset_df.head())



