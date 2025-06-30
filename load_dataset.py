from datasets import load_dataset
from transformers import AutoTokenizer

# Memuat tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Memuat dataset dari file JSON
dataset = load_dataset('json', data_files='chatbot_dataset.json')

# Menampilkan dataset untuk verifikasi
print("Dataset yang dimuat:")
print(dataset)

# Fungsi untuk tokenisasi dataset
def tokenize_function(examples):
    # Tokenisasi teks pada kolom 'question'
    return tokenizer(examples['question'], padding="max_length", truncation=True, max_length=512)

# Tokenisasi dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Menyimpan dataset yang telah ditokenisasi ke disk
tokenized_datasets.save_to_disk('path/to/save/tokenized_dataset')

# Menampilkan beberapa data untuk memastikan berhasil
print("Contoh dataset yang telah ditokenisasi:")
print(tokenized_datasets['train'][0])  # Menampilkan data pertama pada 'train' split
