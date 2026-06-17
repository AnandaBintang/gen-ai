# ✍️ UMKM Copywriting Generator

Aplikasi **RAG (Retrieval-Augmented Generation)** berbasis Streamlit untuk menghasilkan copywriting produk UMKM Indonesia secara otomatis.

> ⚠️ Implementasi AI **tidak** menggunakan closed-source provider (OpenAI / Anthropic / Gemini).  
> Model yang digunakan adalah **LLaMA 3.1 8B** (Meta, open-source) via Groq API.

---

## Fitur

| Output | Deskripsi |
|---|---|
| **Deskripsi Produk** | 3–4 kalimat informatif dan persuasif |
| **Caption Instagram** | Engaging, ada emoji, 4–5 hashtag relevan |
| **Tagline** | Singkat, maksimal 8 kata, catchy |

---

## Tech Stack

| Komponen | Teknologi |
|---|---|
| LLM | `llama-3.1-8b-instant` via **Groq API** |
| Embedding | `all-MiniLM-L6-v2` (lokal, ~90 MB) |
| Vector DB | **FAISS** (`faiss-cpu`) |
| Knowledge Base | JSON — 100 contoh copywriting UMKM |
| Interface | **Streamlit** |
| Evaluasi | ROUGE-score, response time, retrieval similarity |

---

## Struktur Folder

```
umkm-copywriting-generator/
├── app.py                  # Streamlit UI (entry point)
├── rag_pipeline.py         # Core RAG logic
├── embedder.py             # Embedding + FAISS indexing
├── evaluator.py            # Evaluasi ROUGE & response time
│
├── data/
│   └── knowledge_base.json # 100 contoh copywriting UMKM
│
├── index/                  # Di-generate otomatis
│   ├── faiss.index
│   └── chunks.pkl
│
├── outputs/                # Di-generate saat evaluasi
│   ├── eval_results.csv
│   └── topk_experiment.csv
│
├── notebooks/
│   └── experiment.ipynb    # Notebook eksperimen & visualisasi
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Instalasi

### 1. Clone & buat virtual environment

```bash
git clone https://github.com/kelompokxx/umkm-copywriting-generator
cd umkm-copywriting-generator

python -m venv venv
source venv/bin/activate        # Mac / Linux
# atau
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi API Key

```bash
cp .env.example .env
```

Edit `.env` dan isi `GROQ_API_KEY` dengan API key kamu.  
Dapatkan gratis di: <https://console.groq.com>

---

## Cara Menjalankan

### Langkah 1 — Build FAISS Index (lakukan sekali saja)

```bash
python embedder.py
```

Output yang diharapkan:
```
Loading embedding model...
Encoding 100 documents...
Index built: 100 vectors, dim=384
```

### Langkah 2 — Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser di <http://localhost:8501>

### Langkah 3 — Evaluasi (opsional)

```bash
# Evaluasi penuh (20 test cases, default k=3)
python evaluator.py --mode full

# Evaluasi dengan top-k tertentu
python evaluator.py --mode full --k 5

# Eksperimen pengaruh top-k (k=1,3,5)
python evaluator.py --mode topk
```

---

## Alur Sistem

```
knowledge_base.json
       │
       ▼
Load & parse JSON entries
       │
       ▼
Encode via all-MiniLM-L6-v2
       │
       ▼
Simpan ke faiss.index + chunks.pkl
       │
       ─────────────────────────────── SETUP selesai


User Input (nama produk, kategori, keunggulan)
       │
       ▼
Encode query → query vector
       │
       ▼
FAISS similarity search → top-k (k=3) results
       │
       ▼
Format retrieved docs sebagai few-shot context
       │
       ▼
Build prompt (system + context + user input)
       │
       ▼
Groq API → llama-3.1-8b-instant
       │
       ▼
Parse output → deskripsi + caption + tagline
       │
       ▼
Tampilkan di Streamlit UI
```

---

## Evaluasi

### Metrik

| Metrik | Tool | Keterangan |
|---|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | `rouge-score` | Overlap n-gram dengan referensi |
| Response Time | `time` module | Latensi per generate (detik) |
| Retrieval Similarity | FAISS cosine score | Relevansi dokumen yang diambil |

### Eksperimen Top-k

| top-k | Relevansi Context | Kualitas Output | Response Time |
|---|---|---|---|
| k=1 | Sangat terbatas | Kurang bervariasi | Paling cepat |
| k=3 | Balanced | Optimal | Sedang |
| k=5 | Lebih banyak noise | Bisa menurun | Lebih lambat |

---

## Dataset

**`data/knowledge_base.json`** berisi **100 entri** copywriting produk UMKM Indonesia, terdistribusi ke 5 kategori:

| Kategori | Jumlah |
|---|---|
| makanan | 20 |
| minuman | 20 |
| fashion | 20 |
| kosmetik | 20 |
| kerajinan | 20 |

Setiap entri memiliki field: `id`, `kategori`, `nama_produk`, `keunggulan`, `deskripsi_produk`, `caption_ig`, `tagline`.

---

## Anggota Kelompok

| Nama | NIM | Tugas |
|---|---|---|
| - | - | Backend RAG (`embedder.py`, `rag_pipeline.py`) |
| - | - | UI Streamlit (`app.py`) |
| - | - | Dataset & Knowledge Base |
| - | - | Evaluasi & Laporan (`evaluator.py`) |

---

## Checklist UAS

| Requirement | Status |
|---|---|
| Topik Generative AI | ✅ Text generation copywriting |
| Tidak pakai closed API sebagai inti | ✅ LLaMA via Groq (open-source model) |
| Model open source | ✅ LLaMA 3.1 (Meta) |
| Ada alur program jelas | ✅ Flowchart di README |
| Ada evaluasi output | ✅ ROUGE, response time, retrieval similarity |
| Interface bisa dijalankan | ✅ Streamlit app |
| Reproducible | ✅ `requirements.txt` + README |
| Dataset dijelaskan | ✅ `knowledge_base.json` + dokumentasi |
