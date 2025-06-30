from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Memuat tokenizer dan model DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Inisialisasi Flask app
app = Flask(__name__)

# Riwayat percakapan chatbot
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")  # Ambil input pesan dari form
    response = get_Chat_response(msg)  # Dapatkan respons dari chatbot
    return jsonify({"response": response})  # Kirimkan sebagai JSON


def get_Chat_response(text):
    global chat_history_ids

    # Jawaban berbasis aturan untuk pertanyaan pendaftaran mahasiswa baru
    if "syarat pendaftaran" in text.lower():
        return "Syarat pendaftaran adalah memiliki ijazah SMA dan mengisi formulir pendaftaran di situs kami."
    elif "biaya pendaftaran" in text.lower():
        return "Biaya pendaftaran adalah Rp. 500.000."
    elif "dokumen yang dibutuhkan" in text.lower():
        return "Dokumen yang diperlukan termasuk fotokopi ijazah, KTP, dan pas foto terbaru."
    elif "jadwal pendaftaran" in text.lower():
        return "Apa saja yang harus disiapkan untuk melakukan pendaftaran online?"
    elif "jadwal pendaftaran" in text.lower():
        return "Bagaimana calon mahasiswa dapat mengetahui tahapan berikut setelah melakukan pendaftaran?"
    elif "jadwal pendaftaran" in text.lower():
        return "Bagaimana jika dalam proses pengisian nilai rapor atau unggah rapor tidak berhasil?"
    elif "jadwal pendaftaran" in text.lower():
        return "Nilai rapor pada semester berapa yang perlu diinput dan mata pelajaran apa saja?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apakah saya bebas memilih program studi yang saya inginkan atau sesuai dengan jurusan di SMA/MA/SMK?"
    elif "jadwal pendaftaran" in text.lower():
        return "Berapa jumlah program studi yang bisa saya pilih dari daftar prodi yang sesuai dengan jurusan sekolah?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apakah saya bisa pindah program studi setelah saya memutuskan satu program studi berikut pelunasan blanko pembayaran kuliahnya?"
    elif "jadwal pendaftaran" in text.lower():
        return " Apakah saya bisa pindah lokasi kampus?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apa tahap selanjutnya setelah saya melakukan daftar ulang online?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apakah ada jalur seleksi lain selain jalur online/daring?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apakah ada keringanan/potongan biaya kuliah?"
    elif "jadwal pendaftaran" in text.lower():
        return "Berapa besar file rapor yang akan diunggah?"
    elif "jadwal pendaftaran" in text.lower():
        return "Apakah tersedia pedoman lengkap mengenai tata cara pendaftaran, pembayaran serta daftar ulang?"
    elif "jadwal pendaftaran" in text.lower():
        return "Bagaimana melakukan proses pengunduran diri sebagai mahasiswa baru ?"
    # Jika pertanyaan tidak relevan, gunakan model DialoGPT untuk respons lainnya
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Mengembalikan respons sebagai string
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
def get_Chat_response(text):
    global chat_history_ids

    # Jika pertanyaan tidak relevan, gunakan model DialoGPT untuk respons lainnya
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Mengembalikan respons sebagai string
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


    
if __name__ == "__main__":
    app.run(debug=True)

