from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi model dan tokenizer T5
tokenizer = T5Tokenizer.from_pretrained("panggi/t5-base-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-base-indonesian-summarization-cased")

def valid_url(url):
    """Validasi apakah URL valid."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def extract_article_text(url):
    """Mengambil teks utama dari halaman berita."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise error untuk status HTTP yang tidak sukses
    bs = BeautifulSoup(response.text, "lxml")

    # Mencoba menemukan elemen teks utama
    article_text = " ".join([p.text for p in bs.find_all('p') if len(p.text) > 50])
    if not article_text.strip():
        raise ValueError("Konten utama tidak ditemukan.")
    return article_text

def summarize_text(text, max_input_length=512, max_summary_length=100):
    """Merangkum teks menggunakan model T5."""
    tokens = tokenizer.encode(text, return_tensors='pt', max_length=max_input_length, truncation=True)
    summary_ids = model.generate(
        tokens,
        max_length=max_summary_length,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    error = None

    if request.method == "POST":
        # Tentukan mode input: URL atau teks langsung
        mode = request.form.get("mode")
        if mode == "url":
            url = request.form.get("url")
            if not valid_url(url):
                error = "URL tidak valid. Harap masukkan URL yang benar."
            else:
                try:
                    article_text = extract_article_text(url)
                    summary = summarize_text(article_text)
                except Exception as e:
                    error = f"Terjadi kesalahan saat memproses artikel: {e}"
        elif mode == "text":
            input_text = request.form.get("input_text")
            if not input_text.strip():
                error = "Teks tidak boleh kosong."
            else:
                try:
                    summary = summarize_text(input_text)
                except Exception as e:
                    error = f"Terjadi kesalahan saat meringkas teks: {e}"
        else:
            error = "Mode input tidak dikenal."

    return render_template("index.html", summary=summary, error=error)

if __name__ == "__main__":
    app.run(debug=True)
