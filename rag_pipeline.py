# rag_pipeline.py
import os
from groq import Groq
from dotenv import load_dotenv
from embedder import UMKMEmbedder

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3

SYSTEM_PROMPT = """Kamu adalah ahli copywriting untuk UMKM Indonesia.
Tugasmu adalah membuat copywriting produk yang menarik, autentik, dan sesuai pasar Indonesia.

Kamu akan diberikan:
1. Informasi produk dari pengguna
2. Beberapa contoh copywriting produk serupa sebagai referensi gaya penulisan

Berdasarkan informasi tersebut, buat copywriting dalam format TEPAT berikut:

DESKRIPSI:
[Deskripsi produk 3-4 kalimat, informatif dan persuasif]

CAPTION_IG:
[Caption Instagram dengan emoji, engaging, dan 4-5 hashtag relevan]

TAGLINE:
[Tagline singkat maksimal 8 kata, memorable dan catchy]

Gunakan bahasa Indonesia yang natural dan sesuai gaya UMKM lokal."""


class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedder = UMKMEmbedder()
        self.embedder.load_index()

    def format_context(self, retrieved_docs: list) -> str:
        """Format retrieved docs sebagai few-shot examples"""
        context = "=== CONTOH REFERENSI COPYWRITING ===\n\n"

        for i, doc in enumerate(retrieved_docs, 1):
            d = doc["document"]
            context += f"Contoh {i} ({d['kategori']}):\n"
            context += f"Produk: {d['nama_produk']}\n"
            context += f"Keunggulan: {d['keunggulan']}\n"
            context += f"Deskripsi: {d['deskripsi_produk']}\n"
            context += f"Caption IG: {d['caption_ig']}\n"
            context += f"Tagline: {d['tagline']}\n"
            context += "---\n\n"

        return context

    def build_prompt(self, nama_produk: str, kategori: str,
                     keunggulan: str, context: str) -> str:
        """Build user prompt dengan context"""
        return f"""{context}
=== PRODUK YANG HARUS DIBUAT COPYWRITING-NYA ===

Nama Produk: {nama_produk}
Kategori: {kategori}
Keunggulan: {keunggulan}

Buatkan copywriting sesuai format yang diminta."""

    def generate(self, nama_produk: str, kategori: str,
                 keunggulan: str, k: int = TOP_K) -> dict:
        """
        Main generation function
        Returns: dict dengan keys deskripsi, caption_ig, tagline, retrieved_docs
        """
        # Step 1: Retrieve relevant examples
        query = f"{kategori} {nama_produk} {keunggulan}"
        retrieved = self.embedder.retrieve(query, k=k)

        # Step 2: Build context dan prompt
        context = self.format_context(retrieved)
        user_prompt = self.build_prompt(nama_produk, kategori, keunggulan, context)

        # Step 3: Call Groq API
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        raw_output = response.choices[0].message.content

        # Step 4: Parse output
        parsed = self.parse_output(raw_output)
        parsed["retrieved_docs"] = retrieved
        parsed["raw_output"] = raw_output

        return parsed

    def parse_output(self, text: str) -> dict:
        """Parse structured output dari LLM"""
        result = {
            "deskripsi": "",
            "caption_ig": "",
            "tagline": ""
        }

        lines = text.strip().split("\n")
        current_key = None
        buffer = []

        for line in lines:
            if line.startswith("DESKRIPSI:"):
                if current_key and buffer:
                    result[current_key] = "\n".join(buffer).strip()
                current_key = "deskripsi"
                buffer = [line.replace("DESKRIPSI:", "").strip()]
            elif line.startswith("CAPTION_IG:"):
                if current_key and buffer:
                    result[current_key] = "\n".join(buffer).strip()
                current_key = "caption_ig"
                buffer = [line.replace("CAPTION_IG:", "").strip()]
            elif line.startswith("TAGLINE:"):
                if current_key and buffer:
                    result[current_key] = "\n".join(buffer).strip()
                current_key = "tagline"
                buffer = [line.replace("TAGLINE:", "").strip()]
            else:
                if current_key:
                    buffer.append(line)

        # Flush buffer terakhir
        if current_key and buffer:
            result[current_key] = "\n".join(buffer).strip()

        return result


if __name__ == "__main__":
    pipeline = RAGPipeline()

    result = pipeline.generate(
        nama_produk="Keripik Tempe Renyah",
        kategori="makanan",
        keunggulan="tempe segar, bumbu bawang putih, renyah tahan lama, tanpa pengawet"
    )

    print("=== HASIL GENERATE ===")
    print(f"\nDESKRIPSI:\n{result['deskripsi']}")
    print(f"\nCAPTION IG:\n{result['caption_ig']}")
    print(f"\nTAGLINE:\n{result['tagline']}")
    print(f"\nRetrieved {len(result['retrieved_docs'])} docs:")
    for doc in result['retrieved_docs']:
        print(f"  - {doc['document']['nama_produk']} (score: {doc['score']:.4f})")
