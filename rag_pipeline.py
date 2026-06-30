# rag_pipeline.py
import os
from groq import Groq
from dotenv import load_dotenv
from embedder import UMKMEmbedder

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3

SIMILARITY_THRESHOLD = 0.45

# Mode label yang dikembalikan ke caller
MODE_RAG        = "rag"        # >= 2 dokumen relevan  → few-shot
MODE_PARTIAL    = "partial"    # == 1 dokumen relevan  → 1 reference + adapt
MODE_ZEROSHOT   = "zeroshot"   # 0 dokumen relevan     → full generate


# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────

_BASE_FORMAT = """
Buat copywriting dalam format TEPAT berikut (jangan tambahkan teks lain di luar format):

DESKRIPSI:
[Deskripsi produk 3-4 kalimat, informatif dan persuasif]

CAPTION_IG:
[Caption Instagram dengan emoji, engaging, dan 4-5 hashtag relevan]

TAGLINE:
[Tagline singkat maksimal 8 kata, memorable dan catchy]

Gunakan bahasa Indonesia yang natural dan sesuai gaya UMKM lokal."""

SYSTEM_PROMPT_RAG = (
    "Kamu adalah ahli copywriting untuk UMKM Indonesia.\n"
    "Kamu akan menerima beberapa contoh copywriting produk serupa sebagai referensi gaya penulisan.\n"
    "Ikuti gaya dan struktur dari contoh-contoh tersebut, lalu sesuaikan dengan informasi produk baru.\n"
    + _BASE_FORMAT
)

SYSTEM_PROMPT_PARTIAL = (
    "Kamu adalah ahli copywriting untuk UMKM Indonesia.\n"
    "Kamu akan menerima SATU contoh copywriting produk yang mungkin hanya sebagian relevan.\n"
    "Gunakan contoh tersebut sebatas inspirasi gaya penulisan, "
    "namun tetap prioritaskan keunggulan spesifik produk baru yang diberikan.\n"
    + _BASE_FORMAT
)

SYSTEM_PROMPT_ZEROSHOT = (
    "Kamu adalah ahli copywriting untuk UMKM Indonesia.\n"
    "Tidak ada contoh referensi yang tersedia. "
    "Buat copywriting yang menarik, autentik, dan sesuai pasar Indonesia "
    "hanya berdasarkan informasi produk yang diberikan.\n"
    + _BASE_FORMAT
)


class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedder = UMKMEmbedder()
        self.embedder.load_index()

    # ─────────────────────────────────────────────────────────
    # Filtering
    # ─────────────────────────────────────────────────────────

    def filter_by_threshold(
        self, retrieved_docs: list, threshold: float = SIMILARITY_THRESHOLD
    ) -> list:
        """Kembalikan hanya dokumen dengan similarity >= threshold."""
        return [d for d in retrieved_docs if d["score"] >= threshold]

    # ─────────────────────────────────────────────────────────
    # Prompt builders
    # ─────────────────────────────────────────────────────────

    def build_rag_prompt(
        self, nama_produk: str, kategori: str, keunggulan: str, docs: list
    ) -> str:
        """≥2 referensi relevan → few-shot penuh."""
        context = "=== CONTOH REFERENSI COPYWRITING ===\n\n"
        for i, doc in enumerate(docs, 1):
            d = doc["document"]
            context += f"Contoh {i} (kategori: {d['kategori']}):\n"
            context += f"  Produk    : {d['nama_produk']}\n"
            context += f"  Keunggulan: {d['keunggulan']}\n"
            context += f"  Deskripsi : {d['deskripsi_produk']}\n"
            context += f"  Caption IG: {d['caption_ig']}\n"
            context += f"  Tagline   : {d['tagline']}\n"
            context += "---\n\n"

        return (
            f"{context}"
            f"=== PRODUK BARU ===\n\n"
            f"Nama Produk: {nama_produk}\n"
            f"Kategori   : {kategori}\n"
            f"Keunggulan : {keunggulan}\n\n"
            f"Buatkan copywriting sesuai format, terinspirasi dari contoh di atas."
        )

    def build_partial_prompt(
        self, nama_produk: str, kategori: str, keunggulan: str, docs: list
    ) -> str:
        """Tepat 1 referensi relevan → gunakan sebagai inspirasi gaya saja."""
        d = docs[0]["document"]
        score = docs[0]["score"]
        context = (
            f"=== SATU REFERENSI INSPIRASI (similarity: {score:.3f}) ===\n\n"
            f"Produk    : {d['nama_produk']} (kategori: {d['kategori']})\n"
            f"Keunggulan: {d['keunggulan']}\n"
            f"Deskripsi : {d['deskripsi_produk']}\n"
            f"Caption IG: {d['caption_ig']}\n"
            f"Tagline   : {d['tagline']}\n\n"
            f"Catatan: referensi di atas mungkin tidak sepenuhnya serupa. "
            f"Pelajari GAYA penulisannya, tapi fokus pada keunggulan produk baru berikut.\n"
        )
        return (
            f"{context}"
            f"=== PRODUK BARU ===\n\n"
            f"Nama Produk: {nama_produk}\n"
            f"Kategori   : {kategori}\n"
            f"Keunggulan : {keunggulan}\n\n"
            f"Buatkan copywriting sesuai format."
        )

    def build_zeroshot_prompt(
        self, nama_produk: str, kategori: str, keunggulan: str
    ) -> str:
        """Tidak ada referensi relevan → generate murni dari informasi produk."""
        return (
            f"Tidak ada contoh referensi yang tersedia untuk produk ini.\n\n"
            f"=== INFORMASI PRODUK ===\n\n"
            f"Nama Produk: {nama_produk}\n"
            f"Kategori   : {kategori}\n"
            f"Keunggulan : {keunggulan}\n\n"
            f"Gunakan keahlianmu sebagai copywriter UMKM Indonesia untuk membuat "
            f"copywriting yang menarik dan sesuai pasar lokal."
        )

    # ─────────────────────────────────────────────────────────
    # Routing — pilih mode berdasarkan jumlah dokumen relevan
    # ─────────────────────────────────────────────────────────

    def route(
        self,
        nama_produk: str,
        kategori: str,
        keunggulan: str,
        relevant_docs: list,
    ) -> tuple[str, str, str]:
        """
        Tentukan mode dan bangun (system_prompt, user_prompt).

        Returns:
            (mode, system_prompt, user_prompt)
        """
        n = len(relevant_docs)

        if n >= 2:
            mode = MODE_RAG
            system_prompt = SYSTEM_PROMPT_RAG
            user_prompt = self.build_rag_prompt(
                nama_produk, kategori, keunggulan, relevant_docs
            )
        elif n == 1:
            mode = MODE_PARTIAL
            system_prompt = SYSTEM_PROMPT_PARTIAL
            user_prompt = self.build_partial_prompt(
                nama_produk, kategori, keunggulan, relevant_docs
            )
        else:
            mode = MODE_ZEROSHOT
            system_prompt = SYSTEM_PROMPT_ZEROSHOT
            user_prompt = self.build_zeroshot_prompt(
                nama_produk, kategori, keunggulan
            )

        return mode, system_prompt, user_prompt

    # ─────────────────────────────────────────────────────────
    # Main generate
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        nama_produk: str,
        kategori: str,
        keunggulan: str,
        k: int = TOP_K,
        temperature: float = 0.7,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict:
        """
        Hybrid RAG + Zero-shot pipeline.

        Returns dict dengan keys:
            deskripsi, caption_ig, tagline,
            mode, retrieved_docs, relevant_docs,
            similarity_threshold, raw_output
        """
        # Retrieve top-k kandidat dari FAISS
        query = f"{kategori} {nama_produk} {keunggulan}"
        retrieved_docs = self.embedder.retrieve(query, k=k)

        # Filter berdasarkan similarity threshold
        relevant_docs = self.filter_by_threshold(retrieved_docs, similarity_threshold)

        # Routing → pilih mode + bangun prompt
        mode, system_prompt, user_prompt = self.route(
            nama_produk, kategori, keunggulan, relevant_docs
        )

        # Call Groq API
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        raw_output = response.choices[0].message.content

        # 5. Parse output terstruktur
        parsed = self.parse_output(raw_output)
        parsed.update({
            "mode": mode,
            "retrieved_docs": retrieved_docs,      # semua kandidat (sebelum filter)
            "relevant_docs": relevant_docs,         # yang benar-benar dipakai
            "similarity_threshold": similarity_threshold,
            "raw_output": raw_output,
        })
        return parsed

    # ─────────────────────────────────────────────────────────
    # Output parser
    # ─────────────────────────────────────────────────────────

    def parse_output(self, text: str) -> dict:
        """
        Parse structured output DESKRIPSI / CAPTION_IG / TAGLINE dari LLM.

        Toleran terhadap variasi format yang umum dihasilkan LLM:
          - **DESKRIPSI:**  (markdown bold)
          - DESKRIPSI      (tanpa titik dua)
          - deskripsi:     (huruf kecil)
          - spasi/tab di awal baris
        """
        import re

        SECTION_RE = re.compile(
            r"^\s*\*{0,2}(DESKRIPSI|CAPTION_IG|TAGLINE)\*{0,2}:?\s*$",
            re.IGNORECASE,
        )

        INLINE_RE = re.compile(
            r"^\s*\*{0,2}(DESKRIPSI|CAPTION_IG|TAGLINE)\*{0,2}:\s*(.+)$",
            re.IGNORECASE,
        )

        KEY_MAP = {
            "deskripsi": "deskripsi",
            "caption_ig": "caption_ig",
            "tagline": "tagline",
        }

        result = {"deskripsi": "", "caption_ig": "", "tagline": ""}
        lines = text.strip().split("\n")
        current_key = None
        buffer = []

        def flush():
            if current_key and buffer:
                result[current_key] = "\n".join(buffer).strip()

        for line in lines:
            m_inline = INLINE_RE.match(line)
            m_section = SECTION_RE.match(line)

            if m_inline:
                flush()
                current_key = KEY_MAP[m_inline.group(1).lower()]
                buffer = [m_inline.group(2).strip()]
            elif m_section:
                flush()
                current_key = KEY_MAP[m_section.group(1).lower()]
                buffer = []
            else:
                if current_key is not None:
                    buffer.append(line)

        flush()
        return result


# ─────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_cases = [
        # Produk yang pasti ada banyak referensi > RAG
        {
            "nama_produk": "Keripik Tempe Renyah",
            "kategori": "makanan",
            "keunggulan": "tempe segar, bumbu bawang putih, renyah tahan lama, tanpa pengawet",
        },
        # Produk unik / sangat niche > mungkin Partial / Zero-shot
        {
            "nama_produk": "Sabun Lumpur Vulkanik Ijen",
            "kategori": "kosmetik",
            "keunggulan": "lumpur vulkanik Ijen asli, mineral tinggi, detox kulit, handmade",
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        result = pipeline.generate(**tc)
        mode_label = {
            MODE_RAG: "🟢 RAG (few-shot penuh)",
            MODE_PARTIAL: "🟡 Partial (1 referensi)",
            MODE_ZEROSHOT: "🔴 Zero-shot (tanpa referensi)",
        }[result["mode"]]

        print(f"Produk : {tc['nama_produk']}")
        print(f"Mode   : {mode_label}")
        print(f"Relevan: {len(result['relevant_docs'])}/{len(result['retrieved_docs'])} "
              f"(threshold={result['similarity_threshold']})")
        print(f"\nDESKRIPSI:\n{result['deskripsi']}")
        print(f"\nCAPTION IG:\n{result['caption_ig']}")
        print(f"\nTAGLINE:\n{result['tagline']}")
