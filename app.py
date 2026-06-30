# app.py — UMKM Copywriting Generator
# Streamlit UI (Entry Point)
#
# ── Integrasi dengan rag_pipeline.py ─────────────────────────────────────────
#  Memanggil: RAGPipeline().generate(nama_produk, kategori, keunggulan)
#  Return dict:
#    "deskripsi"          : str
#    "caption_ig"         : str
#    "tagline"            : str
#    "mode"               : "rag" | "partial" | "zeroshot"
#    "retrieved_docs"     : list
#    "relevant_docs"      : list
#    "similarity_threshold": float
#    "raw_output"         : str  ← dipakai fallback parser di app.py
# ─────────────────────────────────────────────────────────────────────────────

import html
import re
import time

import streamlit as st
import streamlit.components.v1 as components

# ── INTEGRASI RAG PIPELINE ───────────────────────────────────────────────────
from rag_pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UMKM Copywriting Generator",
    page_icon="✍️",
    layout="centered",  # Single-column terpusat
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Tema Putih / Biru / Hitam, Minimalis & Bersih
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset & Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #F8FAFC 100%) !important;
    color: #111111 !important;
    min-height: 100vh;
}

/* ── Sembunyikan Sidebar Toggle ── */
section[data-testid="stSidebar"],
button[data-testid="baseButton-header"] {
    display: none !important;
}
div[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Lebar konten utama ── */
.block-container {
    max-width: 880px !important;
    padding: 3rem 2rem 5rem !important;
}

/* ── Smooth scroll ── */
html {
    scroll-behavior: smooth;
}

/* ── Header / Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(239,246,255,0.9) 100%);
    border-radius: 24px;
    border: 1px solid rgba(21,101,192,0.08);
    box-shadow: 0 4px 24px rgba(21,101,192,0.06);
}
.hero::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    border-radius: 24px;
    opacity: 0.03;
    z-index: -1;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #EBF3FF 0%, #DBEAFE 100%);
    color: #1565C0;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 0.4rem 1.1rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(21,101,192,0.15);
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);
    animation: fadeInDown 0.6s ease-out;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #0D1117;
    margin: 0 0 0.8rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
    animation: fadeInUp 0.7s ease-out;
}
.hero-title span {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #475569;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.7;
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1565C0;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    width: 3px;
    height: 16px;
    background: linear-gradient(180deg, #1565C0 0%, #0D47A1 100%);
    border-radius: 2px;
}

/* ── Streamlit Input override ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.9) !important;
    border: 2px solid #E2E8F0 !important;
    border-radius: 12px !important;
    color: #0D1117 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}
div[data-testid="stTextInput"] input:hover,
div[data-testid="stTextArea"] textarea:hover {
    border-color: #CBD5E1 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #1565C0 !important;
    box-shadow: 0 0 0 4px rgba(21,101,192,0.1), 0 4px 16px rgba(21,101,192,0.15) !important;
    outline: none !important;
    background: #FFFFFF !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Generate Button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2.5rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(21,101,192,0.3) !important;
    position: relative;
    overflow: hidden;
}
div[data-testid="stButton"] > button[kind="primary"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(21,101,192,0.4) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover::before {
    left: 100%;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Output section divider ── */
.out-divider {
    border: none;
    border-top: 2px solid #E8EEF6;
    margin: 2rem 0 1.5rem;
}

/* ── Output Card ── */
.out-card {
    background: rgba(255,255,255,0.95);
    border: 2px solid rgba(226,232,240,0.8);
    border-radius: 16px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    position: relative;
    overflow: hidden;
}
.out-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
}
.out-card:hover {
    border-color: rgba(21,101,192,0.3);
    box-shadow: 0 8px 32px rgba(21,101,192,0.12);
    transform: translateY(-2px);
}
.out-card:hover::before {
    transform: scaleX(1);
}
.out-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #E8EEF6;
}
.out-card-badge {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #EBF3FF 0%, #DBEAFE 100%);
    color: #1565C0;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    border: 1px solid rgba(21,101,192,0.12);
    box-shadow: 0 2px 4px rgba(21,101,192,0.05);
}
.out-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0D1117;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
}
.out-card-body {
    font-size: 0.93rem;
    color: #1C2A3A;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
}
.out-card-body-tagline {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-style: italic;
    text-align: center;
    padding: 0.85rem 0 0.35rem;
    line-height: 1.5;
    letter-spacing: -0.01em;
}

/* ── Readonly text area (caption) ── */
div[data-testid="stTextArea"] textarea[disabled] {
    background: #FAFBFC !important;
    border: 1px solid #E0E8F4 !important;
    border-radius: 8px !important;
    color: #1C2A3A !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    resize: none !important;
}

/* ── Placeholder text styling ── */
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #9CA3AF !important;
    font-style: italic !important;
    opacity: 1 !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #5A6473 !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 8px 8px 0 0 !important;
    opacity: 1 !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #1565C0 !important;
    background: #EBF3FF !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #1565C0 !important;
    border-bottom: 3px solid #1565C0 !important;
    background: transparent !important;
}
div[data-testid="stTabs"] [data-testid="stTabContent"] {
    padding-top: 1rem !important;
}

/* ── Copy button (HTML native) ── */
.copy-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.75rem;
    padding: 0.5rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1565C0;
    background: linear-gradient(135deg, #EBF3FF 0%, #DBEAFE 100%);
    border: 1.5px solid rgba(21,101,192,0.2);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);
}
.copy-btn:hover {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    color: #FFFFFF;
    border-color: #1565C0;
    box-shadow: 0 4px 16px rgba(21,101,192,0.25);
    transform: translateY(-1px);
}
.copy-btn:active {
    transform: translateY(0);
}

/* ── Metadata strip ── */
.meta-strip {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin: 1.5rem 0;
    font-size: 0.82rem;
    color: #5A6473;
    flex-wrap: wrap;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    border: 1px solid rgba(226,232,240,0.8);
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.meta-pill {
    background: #F1F5FB;
    border: 1.5px solid #D0D9E8;
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-weight: 600;
    color: #374151;
    font-size: 0.8rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    transition: transform 0.2s ease;
}
.meta-pill:hover {
    transform: scale(1.05);
}
.meta-pill.rag    { background: linear-gradient(135deg, #E6F9F0 0%, #D1FAE5 100%); border-color:#A7DFC0; color:#166534; }
.meta-pill.partial{ background: linear-gradient(135deg, #FEF9E6 0%, #FEF3C7 100%); border-color:#F0D88A; color:#854D0E; }
.meta-pill.zero   { background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); border-color:#FCA5A5; color:#991B1B; }

/* ── Alert override ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] > div {
    border-color: #1565C0 !important;
    border-right-color: transparent !important;
}
div[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(21,101,192,0.12) !important;
    border: 1px solid rgba(21,101,192,0.1) !important;
}

/* ── Expander ── */
details summary {
    font-size: 0.82rem !important;
    color: #5A6473 !important;
}

/* ── Footer caption ── */
.footer-note {
    text-align: center;
    margin-top: 4rem;
    padding: 1.8rem 0;
    border-top: 2px solid rgba(226,232,240,0.8);
    font-size: 0.8rem;
    color: #94A3B8;
    background: rgba(255,255,255,0.5);
    border-radius: 12px;
}
.footer-note::before {
    content: '⚡';
    margin-right: 0.5rem;
    font-size: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inisialisasi RAGPipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """
    Inisialisasi RAGPipeline sekali dan simpan di cache Streamlit.
    Memuat FAISS index + Groq client dari .env.
    Pastikan sudah menjalankan `python embedder.py` sebelum streamlit run.
    """
    return RAGPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Emoji Post-processing
# ─────────────────────────────────────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f9ff"
    "\U00002600-\U000027bf"
    "\U0001fa00-\U0001faff"
    "\U00002702-\U000027b0"
    "\U0000fe00-\U0000fe0f"
    "\U0001f1e0-\U0001f1ff"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """Hapus semua emoji — untuk Deskripsi & Tagline agar teks bersih."""
    out = _EMOJI_RE.sub("", text)
    out = re.sub(r" {2,}", " ", out)
    return out.strip()


def limit_emoji(text: str, max_count: int = 2) -> str:
    """
    Batasi emoji di Caption IG maks `max_count`.
    Teks dan hashtag tetap utuh; hanya kelebihan emoji yang dihilangkan.
    """
    matches = _EMOJI_RE.findall(text)
    if len(matches) <= max_count:
        return text
    # Hapus semua emoji, sisipkan kembali hanya sejumlah max_count di depan
    base = _EMOJI_RE.sub("", text)
    base = re.sub(r" {2,}", " ", base).strip()
    prefix = " ".join(matches[:max_count])
    return f"{prefix} {base}"


# ─────────────────────────────────────────────────────────────────────────────
# Parser — Ekstrak Deskripsi / Caption IG / Tagline dari raw output LLM
#
# Format aktual yang dikeluarkan LLM (berdasarkan observasi):
#   Deskripsi : <teks>
#   Caption IG: <teks>
#   Tagline   : <teks>
#
# Masalah sebelumnya:
#   - Regex per-baris hanya cocok dengan "CAPTION_IG" (underscore)
#   - "Caption IG :" atau "CAPTION IG :" (spasi + spasi sebelum titik dua)
#     tidak tertangkap → seluruh output menumpuk di kolom Deskripsi
#
# Solusi: re.split + re.search per-seksi
#   1. Pecah raw_output berdasarkan header seksi (re.split)
#   2. Ekstrak konten dari setiap potongan (re.search)
#   Pendekatan ini tidak bergantung pada posisi label di baris tersendiri.
# ─────────────────────────────────────────────────────────────────────────────

# Delimiter: mendeteksi baris yang berisi label seksi, toleran terhadap:
#   - Huruf besar/kecil campur        : "deskripsi", "DESKRIPSI", "Deskripsi"
#   - Spasi sebelum/sesudah titik dua : "Caption IG :" atau "Caption IG:"
#   - Spasi atau underscore            : "Caption IG" atau "CAPTION_IG"
#   - Markdown bold **                 : "**TAGLINE**:"
#   Grup 1 = nama label seksi (digunakan untuk routing)
_SECTION_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*\*{0,2}"  # awal baris, opsional **
    r"(DESKRIPSI|CAPTION[\s_]IG|TAGLINE)"  # nama seksi
    r"\*{0,2}"  # opsional **
    r"\s*:\s*",  # titik dua dengan spasi opsional di kiri/kanan
    re.IGNORECASE,
)


def _normalize_key(label: str) -> str:
    """Mapping nama label LLM → kunci dict standar."""
    label = label.strip().lower()
    if label.startswith("caption"):
        return "caption_ig"
    if label.startswith("deskripsi"):
        return "deskripsi"
    if label.startswith("tagline"):
        return "tagline"
    return label


def robust_parse(raw_output: str) -> dict:
    """
    Parse raw output LLM menggunakan re.split + re.search.

    Cara kerja:
      1. re.split() dengan _SECTION_SPLIT_RE memotong teks di setiap
         kemunculan header seksi (DESKRIPSI / CAPTION IG / TAGLINE).
         Karena pakai capturing group, hasil split menyertakan nama label.
      2. Iterasi hasil split berpasangan: (label, konten)
      3. Konten di-strip dari whitespace dan disimpan ke dict.

    Toleran terhadap:
      - "Deskripsi :" / "DESKRIPSI:" / "deskripsi  :"
      - "Caption IG:" / "CAPTION_IG:" / "Caption IG :"
      - "Tagline:" / "TAGLINE   :"
      - Bold markdown **DESKRIPSI**:
      - Teks setelah label di baris yang sama (inline)
      - Konten multi-baris
    """
    result = {"deskripsi": "", "caption_ig": "", "tagline": ""}

    # re.split dengan capturing group → daftar: [teks_sebelum, label1, konten1, label2, ...]
    # Contoh hasil split dari "Deskripsi: A\nCaption IG: B\nTagline: C":
    #   ["", "Deskripsi", "A\n", "Caption IG", "B\n", "Tagline", "C"]
    parts = _SECTION_SPLIT_RE.split(raw_output)

    # parts[0] = teks sebelum seksi pertama (biasanya kosong / bisa diabaikan)
    # parts[1], parts[2] = label seksi 1, konten seksi 1
    # parts[3], parts[4] = label seksi 2, konten seksi 2, dst.
    i = 1  # mulai dari indeks 1 (lewati teks pra-seksi di indeks 0)
    while i + 1 < len(parts):
        label = parts[i].strip()  # nama seksi dari capturing group
        content = parts[i + 1].strip()  # isi konten seksi
        key = _normalize_key(label)
        if key in result:
            result[key] = content
        i += 2

    return result


def extract_fields(result: dict) -> tuple[str, str, str]:
    """
    Ambil (deskripsi, caption_ig, tagline) dari result dict pipeline.

    Strategi berlapis:
      1. Gunakan nilai dari pipeline.generate() jika semua field terisi.
      2. Jika ada field kosong, jalankan robust_parse() terhadap raw_output
         agar dapat menangkap format aktual LLM yang bervariasi.
      3. Selalu parse raw_output jika deskripsi mengandung penanda seksi lain
         (tanda bahwa pipeline.parse_output() gagal memisahkan seksi).

    ── Kunci result dict dari rag_pipeline.generate() ───────────────────
      result["deskripsi"]    → parsed oleh rag_pipeline.parse_output()
      result["caption_ig"]   → sama
      result["tagline"]      → sama
      result["raw_output"]   → teks mentah LLM (fallback)
    ────────────────────────────────────────────────────────────────────
    """
    deskripsi = (result.get("deskripsi", "") or "").strip()
    caption_ig = (result.get("caption_ig", "") or "").strip()
    tagline = (result.get("tagline", "") or "").strip()
    raw_output = result.get("raw_output", "") or ""

    # Deteksi kegagalan parser pipeline: caption atau tagline kosong,
    # ATAU deskripsi mengandung label seksi lain (output menumpuk)
    _section_marker = re.compile(r"(caption[\s_]ig|tagline)", re.IGNORECASE)
    pipeline_failed = (
        not caption_ig or not tagline or bool(_section_marker.search(deskripsi))
    )

    if pipeline_failed and raw_output:
        fb = robust_parse(raw_output)
        # Ambil nilai dari fallback hanya jika lebih baik dari pipeline
        if fb.get("deskripsi"):
            deskripsi = fb["deskripsi"]
        if fb.get("caption_ig"):
            caption_ig = fb["caption_ig"]
        if fb.get("tagline"):
            tagline = fb["tagline"]

    return deskripsi or "—", caption_ig or "—", tagline or "—"


# ─────────────────────────────────────────────────────────────────────────────
# Copy to Clipboard (JavaScript)
# ─────────────────────────────────────────────────────────────────────────────


def copy_button(text: str, btn_id: str, label: str = "Salin") -> None:
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html_content = f"""
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: transparent; }}
        .copy-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 1.2rem;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            color: #1565C0;
            background: linear-gradient(135deg, #EBF3FF 0%, #DBEAFE 100%);
            border: 1.5px solid rgba(21,101,192,0.2);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(21,101,192,0.08);
        }}
        .copy-btn:hover {{
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            color: #FFFFFF;
            border-color: #1565C0;
            box-shadow: 0 4px 16px rgba(21,101,192,0.25);
            transform: translateY(-1px);
        }}
        .copy-btn:active {{
            transform: translateY(0);
        }}
    </style>
    <button class="copy-btn" id="{btn_id}" onclick="
        navigator.clipboard.writeText(`{escaped}`)
            .then(()=>{{
                var b=document.getElementById('{btn_id}');
                b.innerHTML='&#10003;&nbsp;Tersalin!';
                b.style.background='linear-gradient(135deg, #166534 0%, #15803D 100%)';
                b.style.color='#FFFFFF';
                b.style.borderColor='#166534';
                b.style.transform='translateY(0)';
                setTimeout(()=>{{
                    b.innerHTML='&#128203;&nbsp;{label}';
                    b.style.background='';
                    b.style.color='';
                    b.style.borderColor='';
                    b.style.transform='';
                }}, 2000);
            }})
            .catch(()=>{{
                var b=document.getElementById('{btn_id}');
                b.innerHTML='&#10007;&nbsp;Gagal';
                b.style.background='linear-gradient(135deg, #991B1B 0%, #DC2626 100%)';
                b.style.color='#FFFFFF';
                setTimeout(()=>{{
                    b.innerHTML='&#128203;&nbsp;{label}';
                    b.style.background='';
                    b.style.color='';
                }}, 2000);
            }});
    ">&#128203;&nbsp;{label}</button>
    """
    components.html(html_content, height=50)


# ─────────────────────────────────────────────────────────────────────────────
# Validasi Input
# ─────────────────────────────────────────────────────────────────────────────


def validate_inputs(nama: str, unggulan: str) -> list[str]:
    errors = []
    if not nama.strip():
        errors.append("**Nama Produk** tidak boleh kosong.")
    elif len(nama.strip()) < 3:
        errors.append("**Nama Produk** terlalu pendek (minimal 3 karakter).")
    if not unggulan.strip():
        errors.append("**Keunggulan Produk** tidak boleh kosong.")
    elif len(unggulan.strip()) < 10:
        errors.append("**Keunggulan Produk** terlalu singkat. Berikan detail lebih.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
    <div class="hero-eyebrow">RAG + LLaMA 3.1 · Groq API · FAISS</div>
    <h1 class="hero-title">UMKM <span>Copywriting</span> Generator</h1>
    <p class="hero-sub">
        Buat deskripsi produk, caption Instagram, dan tagline yang menarik
        untuk usaha kamu — otomatis, cepat, dan berbasis AI open-source.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Form Input
# ─────────────────────────────────────────────────────────────────────────────

KATEGORI_OPTIONS = ["makanan", "minuman", "fashion", "kosmetik", "kerajinan"]

st.markdown('<p class="section-label">Informasi Produk</p>', unsafe_allow_html=True)

col_nama, col_kat = st.columns([2, 1], gap="medium")

with col_nama:
    nama_produk = st.text_input(
        label="Nama Produk",
        placeholder="Contoh: Keripik Singkong Renyah Bu Sri",
        help="Nama lengkap produk UMKM kamu.",
        key="input_nama",
    )

with col_kat:
    kategori = st.selectbox(
        label="Kategori",
        options=KATEGORI_OPTIONS,
        format_func=lambda k: k.capitalize(),
        help="Pilih kategori yang paling sesuai.",
        key="input_kategori",
    )

keunggulan = st.text_area(
    label="Keunggulan Produk",
    placeholder="Contoh: Renyah, tanpa MSG, dibuat dari singkong segar pilihan, tersedia dalam 3 varian rasa",
    help="Jelaskan keunggulan dan keunikan produk. Semakin detail, semakin baik hasilnya.",
    height=110,
    key="input_keunggulan",
)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

generate_clicked = st.button(
    "Generate Copywriting",
    type="primary",
    key="btn_generate",
)


# ─────────────────────────────────────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────────────────────────────────────

if generate_clicked:
    errors = validate_inputs(nama_produk, keunggulan)

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")

    else:
        # ── Load pipeline ──────────────────────────────────────────────────
        try:
            pipeline = load_pipeline()
        except Exception as exc:
            st.error(
                f"**Gagal memuat RAG Pipeline.**\n\n"
                f"Pastikan:\n"
                f"- FAISS index sudah dibuild: jalankan `python embedder.py`\n"
                f"- `GROQ_API_KEY` sudah diisi di `.env`\n\n"
                f"Detail: `{exc}`"
            )
            st.stop()

        # ── Generate ───────────────────────────────────────────────────────
        st.markdown('<hr class="out-divider">', unsafe_allow_html=True)

        with st.spinner("Sedang membuat copywriting…"):
            t0 = time.time()
            try:
                # ── INTEGRASI: panggil pipeline.generate() ─────────────────
                # Jika signature berubah di rag_pipeline.py, sesuaikan di sini
                result = pipeline.generate(
                    nama_produk=nama_produk.strip(),
                    kategori=kategori,
                    keunggulan=keunggulan.strip(),
                )
                # ───────────────────────────────────────────────────────────

            except Exception as exc:
                st.error(
                    f"**Gagal memanggil Groq API.**\n\n"
                    f"Kemungkinan penyebab: API key tidak valid, rate limit, "
                    f"atau koneksi internet terputus.\n\n"
                    f"Detail: `{exc}`"
                )
                st.stop()

        elapsed = time.time() - t0

        # ── Ekstrak & bersihkan field ───────────────────────────────────────
        # extract_fields() akan menjalankan fallback_parse() jika perlu
        deskripsi_raw, caption_raw, tagline_raw = extract_fields(result)

        # Post-processing emoji
        deskripsi = strip_emoji(deskripsi_raw) or "—"
        tagline = strip_emoji(tagline_raw) or "—"
        caption_ig = limit_emoji(caption_raw, 2) if caption_raw != "—" else "—"

        # ── Metadata strip ─────────────────────────────────────────────────
        mode = result.get("mode", "zeroshot")
        n_rel = len(result.get("relevant_docs", []))
        n_ret = len(result.get("retrieved_docs", []))
        thr = result.get("similarity_threshold", 0.45)

        mode_label = {
            "rag": ("RAG — Few-shot", "rag"),
            "partial": ("Partial — 1 Ref", "partial"),
            "zeroshot": ("Zero-shot", "zero"),
        }.get(mode, ("Zero-shot", "zero"))

        st.markdown(
            f"""
            <div class="meta-strip">
                <span class="meta-pill {mode_label[1]}">{mode_label[0]}</span>
                <span>{n_rel}/{n_ret} dok relevan &nbsp;·&nbsp; threshold {thr}</span>
                <span style="margin-left:auto; color:#9CA3AF;">{elapsed:.2f} dtk</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Output Tabs — Deskripsi / Tagline / Caption ──────────────────
        # html.escape() mencegah HTML injection dari output LLM.
        tab_desk, tab_tag, tab_cap = st.tabs(
            ["📝 Deskripsi", "✨ Tagline", "📸 Caption IG"]
        )

        # Tab 1 — Deskripsi Produk
        with tab_desk:
            desk_safe = html.escape(deskripsi)
            st.markdown(
                f"""
            <div class="out-card">
                <div class="out-card-header">
                    <span class="out-card-badge">01</span>
                    <span class="out-card-title">Deskripsi Produk</span>
                </div>
                <div class="out-card-body">{desk_safe}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            copy_button(deskripsi, btn_id="copy_desk", label="Salin Deskripsi")

        # Tab 2 — Tagline
        with tab_tag:
            tag_safe = html.escape(tagline)
            st.markdown(
                f"""
            <div class="out-card">
                <div class="out-card-header">
                    <span class="out-card-badge">02</span>
                    <span class="out-card-title">Tagline</span>
                </div>
                <div class="out-card-body-tagline">&ldquo;{tag_safe}&rdquo;</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            copy_button(tagline, btn_id="copy_tag", label="Salin Tagline")

        # Tab 3 — Caption Instagram
        with tab_cap:
            cap_safe = html.escape(caption_ig).replace("\n", "<br>")
            st.markdown(
                f"""
            <div class="out-card">
                <div class="out-card-header">
                    <span class="out-card-badge">03</span>
                    <span class="out-card-title">Caption Instagram</span>
                </div>
                <div class="out-card-body" style="white-space: pre-wrap;">{cap_safe}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            copy_button(caption_ig, btn_id="copy_cap", label="Salin Caption")

        # ── Expander: Raw Output (debugging) ──────────────────────────────
        with st.expander("Lihat raw output LLM (untuk debugging)", expanded=False):
            st.code(result.get("raw_output", ""), language=None)

# ── Placeholder sebelum generate ──────────────────────────────────────────────
else:
    st.markdown(
        """
    <div style="
        text-align:center;
        padding: 3.5rem 1rem;
        color: #9CA3AF;
    ">
        <div style="font-size:2.8rem; margin-bottom:0.75rem;">✍️</div>
        <p style="font-size:0.9rem; margin:0;">
            Isi formulir di atas lalu klik
            <strong style="color:#1565C0;">Generate Copywriting</strong>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="footer-note">
    Proyek UAS Generative AI &nbsp;·&nbsp;
    LLaMA 3.1 8B (Meta, open-source) via Groq API &nbsp;·&nbsp;
    FAISS · Streamlit
</div>
""",
    unsafe_allow_html=True,
)
