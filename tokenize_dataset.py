from transformers import AutoTokenizer
from datasets import load_dataset

# Memuat tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Memuat dataset dari file JSON
dataset = load_dataset('json', data_files='chatbot_dataset.json')

# Fungsi untuk tokenisasi dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], padding="max_length", truncation=True, max_length=512)

# Tokenisasi dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Menyimpan dataset tokenized
tokenized_datasets.save_to_disk('path/to/save/tokenized_dataset')

print("Dataset telah berhasil ditokenisasi dan disimpan.")
