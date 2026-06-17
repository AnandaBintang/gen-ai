# evaluator.py
"""
Modul evaluasi untuk UMKM Copywriting Generator.
Menghitung ROUGE score dan response time per test case.
Hasil disimpan ke outputs/eval_results.csv.
"""

import os
import time
import json
import pathlib
import pandas as pd
from rouge_score import rouge_scorer
from rag_pipeline import RAGPipeline

# ============================================================
# 20 TEST CASES — satu per sub-kategori / variasi produk
# reference_deskripsi diambil / diinspirasi dari knowledge base
# ============================================================
TEST_CASES = [
    # ─── MAKANAN (8 cases) ───────────────────────────────────
    {
        "nama_produk": "Keripik Tempe Bawang",
        "kategori": "makanan",
        "keunggulan": "tempe segar, bumbu bawang goreng, renyah, tanpa pengawet",
        "reference_deskripsi": (
            "Keripik tempe segar dengan bumbu bawang goreng yang melimpah dan tekstur renyah yang tahan lama. "
            "Dibuat tanpa pengawet buatan sehingga aman untuk seluruh keluarga. "
            "Camilan lokal favorit yang cocok dinikmati kapan saja."
        ),
    },
    {
        "nama_produk": "Sambal Ijo Padang",
        "kategori": "makanan",
        "keunggulan": "cabai hijau segar, bawang merah, pedas segar, tanpa MSG",
        "reference_deskripsi": (
            "Sambal ijo khas Padang menggunakan cabai hijau segar pilihan yang menghasilkan rasa pedas segar dan aromatik. "
            "Dimasak dengan bawang merah dan rempah tanpa MSG untuk menjaga keaslian rasa. "
            "Cocok sebagai pelengkap nasi, ayam, maupun ikan goreng."
        ),
    },
    {
        "nama_produk": "Brownies Kukus Kopi",
        "kategori": "makanan",
        "keunggulan": "kopi arabika, coklat premium, kukus, tekstur lembut",
        "reference_deskripsi": (
            "Brownies kukus dengan sentuhan kopi arabika yang memberikan aroma dan rasa kopi khas yang menggoda. "
            "Menggunakan coklat premium dan proses pengukusan untuk menghasilkan tekstur yang sangat lembut. "
            "Perpaduan sempurna bagi pecinta kopi dan coklat sekaligus."
        ),
    },
    {
        "nama_produk": "Rendang Ayam Kampung",
        "kategori": "makanan",
        "keunggulan": "ayam kampung asli, bumbu rempah lengkap, tahan 5 hari, autentik",
        "reference_deskripsi": (
            "Rendang ayam kampung asli dengan bumbu rempah lengkap yang dimasak perlahan hingga bumbu meresap sempurna. "
            "Menggunakan ayam kampung pilihan yang empuk namun tidak mudah hancur. "
            "Tahan hingga 5 hari di kulkas tanpa mengurangi cita rasa autentik."
        ),
    },
    {
        "nama_produk": "Onde-onde Wijen Premium",
        "kategori": "makanan",
        "keunggulan": "wijen tebal, isian kacang hijau, kenyal, digoreng segar",
        "reference_deskripsi": (
            "Onde-onde premium berlapis wijen tebal dengan isian kacang hijau manis yang lembut dan harum. "
            "Dibuat dari tepung ketan pilihan untuk tekstur kenyal yang sempurna. "
            "Digoreng segar setiap hari tanpa pengawet untuk menjaga kualitas terbaik."
        ),
    },
    {
        "nama_produk": "Baso Aci Pedas Level",
        "kategori": "makanan",
        "keunggulan": "kenyal, kuah gurih, level pedas 1-5, bahan alami",
        "reference_deskripsi": (
            "Baso aci dengan tekstur kenyal sempurna dan kuah gurih dari kaldu ayam asli. "
            "Tersedia pilihan level pedas 1 hingga 5 yang bisa disesuaikan selera. "
            "Dibuat dari bahan alami tanpa pengawet untuk kelezatan yang segar dan aman."
        ),
    },
    {
        "nama_produk": "Peyek Kacang Mede",
        "kategori": "makanan",
        "keunggulan": "kacang mede premium, renyah, bumbu gurih, tanpa MSG",
        "reference_deskripsi": (
            "Peyek kacang mede premium dengan butiran kacang mede berkualitas yang melimpah di setiap lembarnya. "
            "Bumbu gurih yang meresap sempurna tanpa MSG memberikan cita rasa autentik yang khas. "
            "Tekstur renyah yang tahan lama, cocok sebagai camilan maupun lauk pendamping."
        ),
    },
    {
        "nama_produk": "Bika Ambon Pandan",
        "kategori": "makanan",
        "keunggulan": "pandan segar, bersarang sempurna, fermentasi alami, resep turun-temurun",
        "reference_deskripsi": (
            "Bika Ambon pandan dengan pori-pori bersarang sempurna hasil fermentasi alami yang tepat. "
            "Menggunakan perasan pandan segar untuk aroma dan warna hijau yang alami dan menggugah selera. "
            "Resep turun-temurun yang menghasilkan cita rasa autentik dan tekstur yang lembut."
        ),
    },
    # ─── MINUMAN (4 cases) ───────────────────────────────────
    {
        "nama_produk": "Kopi Robusta Toraja",
        "kategori": "minuman",
        "keunggulan": "single origin Toraja, dark roast, bold flavor, tanpa campuran",
        "reference_deskripsi": (
            "Kopi robusta single origin dari dataran tinggi Toraja dengan proses dark roast yang menghasilkan rasa bold dan penuh. "
            "Aroma kuat dan rasa pahit yang seimbang menjadi ciri khas kopi Toraja yang otentik. "
            "Ditanam oleh petani lokal tanpa campuran varietas lain untuk kemurnian rasa terbaik."
        ),
    },
    {
        "nama_produk": "Teh Herbal Jahe Kayu Manis",
        "kategori": "minuman",
        "keunggulan": "jahe merah, kayu manis, menghangatkan, tanpa gula tambahan",
        "reference_deskripsi": (
            "Teh herbal dengan perpaduan jahe merah dan kayu manis yang menghasilkan kehangatan dari dalam. "
            "Formula tanpa gula tambahan menjadikannya pilihan sehat untuk program hidup bersih. "
            "Aroma rempah yang menenangkan cocok dinikmati di pagi maupun malam hari."
        ),
    },
    {
        "nama_produk": "Minuman Sinom Kunyit Asam",
        "kategori": "minuman",
        "keunggulan": "kunyit segar, asam jawa, menyegarkan, tradisional Jawa",
        "reference_deskripsi": (
            "Minuman sinom kunyit asam tradisional Jawa yang menyegarkan dengan perpaduan kunyit segar dan asam jawa pilihan. "
            "Kaya manfaat antioksidan alami yang baik untuk kesehatan tubuh sehari-hari. "
            "Rasa manis asam yang seimbang menjadikannya minuman jamu yang disukai semua kalangan."
        ),
    },
    {
        "nama_produk": "Cold Brew Arabika Flores",
        "kategori": "minuman",
        "keunggulan": "arabika Flores, cold brew 18 jam, fruity notes, tanpa gula",
        "reference_deskripsi": (
            "Cold brew dibuat dari biji arabika single origin Flores yang diseduh dingin selama 18 jam. "
            "Proses cold brew mengekstrak profil rasa fruity dan floral yang unik tanpa rasa pahit berlebih. "
            "Tanpa tambahan gula, murni kenikmatan kopi yang bersih dan menyegarkan."
        ),
    },
    # ─── FASHION (3 cases) ───────────────────────────────────
    {
        "nama_produk": "Kemeja Batik Kontemporer",
        "kategori": "fashion",
        "keunggulan": "motif batik modern, bahan rayon adem, slim fit, formal kasual",
        "reference_deskripsi": (
            "Kemeja batik kontemporer dengan motif modern yang memadukan estetika tradisional dan desain masa kini. "
            "Bahan rayon viscose premium yang lembut dan sangat adem untuk kenyamanan seharian. "
            "Potongan slim fit yang versatile, cocok untuk acara formal maupun kasual."
        ),
    },
    {
        "nama_produk": "Tas Anyam Rotan Bali",
        "kategori": "fashion",
        "keunggulan": "rotan asli Bali, handwoven, desain bohemian, tahan lama",
        "reference_deskripsi": (
            "Tas anyam dari rotan asli Bali yang dianyam tangan oleh pengrajin lokal berpengalaman. "
            "Desain bohemian yang timeless cocok untuk berbagai gaya dan kesempatan. "
            "Bahan rotan alami yang kuat dan tahan lama menjadikan tas ini investasi fashion yang bermakna."
        ),
    },
    {
        "nama_produk": "Sandal Kulit Sapi Handmade",
        "kategori": "fashion",
        "keunggulan": "kulit sapi asli, handmade Yogyakarta, adjustable strap, tahan lama",
        "reference_deskripsi": (
            "Sandal kulit sapi asli yang dikerjakan tangan oleh pengrajin kulit berpengalaman di Yogyakarta. "
            "Menggunakan full grain leather berkualitas tinggi yang makin indah seiring waktu pemakaian. "
            "Tali adjustable strap yang fleksibel memastikan kenyamanan untuk berbagai ukuran kaki."
        ),
    },
    # ─── KOSMETIK (3 cases) ─────────────────────────────────
    {
        "nama_produk": "Serum Vitamin C Niacinamide",
        "kategori": "kosmetik",
        "keunggulan": "vitamin C 15%, niacinamide 5%, mencerahkan, BPOM terdaftar",
        "reference_deskripsi": (
            "Serum wajah dengan kandungan Vitamin C 15% dan Niacinamide 5% yang bekerja sinergis untuk mencerahkan dan meratakan warna kulit. "
            "Formula ringan yang cepat meresap cocok untuk semua jenis kulit termasuk kulit sensitif. "
            "Sudah terdaftar BPOM dan aman untuk pemakaian rutin pagi dan malam."
        ),
    },
    {
        "nama_produk": "Sabun Mandi Charcoal Detox",
        "kategori": "kosmetik",
        "keunggulan": "activated charcoal, membersihkan pori, aroma mint segar, halal",
        "reference_deskripsi": (
            "Sabun mandi dengan kandungan activated charcoal yang efektif menyerap kotoran dan racun dari pori-pori kulit. "
            "Aroma mint segar memberikan sensasi dingin yang menyegarkan dan rasa bersih yang tahan lama. "
            "Formula halal dan bebas paraben yang aman untuk pemakaian sehari-hari seluruh keluarga."
        ),
    },
    {
        "nama_produk": "Masker Lumpur Volcanique",
        "kategori": "kosmetik",
        "keunggulan": "lumpur vulkanik asli, menyerap minyak, pori bersih, natural",
        "reference_deskripsi": (
            "Masker wajah dari lumpur vulkanik asli yang kaya mineral alami untuk mengangkat kotoran dan mengecilkan pori-pori. "
            "Formula natural tanpa bahan kimia berbahaya yang efektif menyerap minyak berlebih di wajah. "
            "Kulit terasa segar, bersih, dan cerah setelah pemakaian rutin 2-3 kali seminggu."
        ),
    },
    # ─── KERAJINAN (2 cases) ────────────────────────────────
    {
        "nama_produk": "Pot Tanaman Keramik Motif Etnik",
        "kategori": "kerajinan",
        "keunggulan": "keramik tangan, motif etnik, anti pecah, ukuran variatif",
        "reference_deskripsi": (
            "Pot tanaman keramik buatan tangan dengan motif etnik Indonesia yang kaya warna dan detail indah. "
            "Menggunakan tanah liat pilihan yang dibakar pada suhu tinggi untuk kekuatan dan ketahanan jangka panjang. "
            "Tersedia dalam berbagai ukuran untuk tanaman indoor maupun outdoor kesayangan."
        ),
    },
    {
        "nama_produk": "Lampu Hias Bambu Rotan",
        "kategori": "kerajinan",
        "keunggulan": "bambu dan rotan alami, handmade, desain unik, eco-friendly",
        "reference_deskripsi": (
            "Lampu hias dari bambu dan rotan alami yang dianyam tangan menciptakan pola cahaya yang indah dan hangat. "
            "Setiap produk merupakan karya tangan yang unik dengan desain yang tidak pernah sama persis. "
            "Material eco-friendly yang berkelanjutan menjadikannya pilihan dekorasi rumah yang bertanggung jawab."
        ),
    },
]


def evaluate(top_k: int = 3, save_csv: bool = True) -> pd.DataFrame:
    """
    Jalankan evaluasi ROUGE untuk semua test cases.

    Args:
        top_k: Jumlah dokumen referensi yang diambil saat retrieval.
        save_csv: Simpan hasil ke outputs/eval_results.csv jika True.

    Returns:
        DataFrame berisi semua hasil evaluasi.
    """
    pipeline = RAGPipeline()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    pathlib.Path("outputs").mkdir(exist_ok=True)
    results = []

    print(f"\n{'='*60}")
    print(f"  EVALUASI UMKM COPYWRITING GENERATOR  (top_k={top_k})")
    print(f"{'='*60}\n")

    for i, tc in enumerate(TEST_CASES, 1):
        start = time.time()
        output = pipeline.generate(
            nama_produk=tc["nama_produk"],
            kategori=tc["kategori"],
            keunggulan=tc["keunggulan"],
            k=top_k,
        )
        elapsed = time.time() - start

        scores = scorer.score(tc["reference_deskripsi"], output["deskripsi"])

        # Ambil similarity score tertinggi dari retrieved docs
        top_similarity = (
            max(d["score"] for d in output["retrieved_docs"])
            if output["retrieved_docs"]
            else 0.0
        )

        results.append(
            {
                "no": i,
                "nama_produk": tc["nama_produk"],
                "kategori": tc["kategori"],
                "rouge1_f": round(scores["rouge1"].fmeasure, 4),
                "rouge2_f": round(scores["rouge2"].fmeasure, 4),
                "rougeL_f": round(scores["rougeL"].fmeasure, 4),
                "top_retrieval_score": round(top_similarity, 4),
                "response_time_s": round(elapsed, 2),
                "generated_deskripsi": output["deskripsi"],
                "generated_caption_ig": output["caption_ig"],
                "generated_tagline": output["tagline"],
            }
        )

        print(
            f"[{i:02d}/{len(TEST_CASES)}] {tc['nama_produk']:<35} "
            f"R-1:{scores['rouge1'].fmeasure:.3f}  "
            f"R-L:{scores['rougeL'].fmeasure:.3f}  "
            f"sim:{top_similarity:.3f}  "
            f"t:{elapsed:.1f}s"
        )

    df = pd.DataFrame(results)

    if save_csv:
        csv_path = "outputs/eval_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nHasil disimpan ke: {csv_path}")

    # ─── Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total test cases    : {len(df)}")
    print(f"  ROUGE-1  avg        : {df['rouge1_f'].mean():.4f}")
    print(f"  ROUGE-2  avg        : {df['rouge2_f'].mean():.4f}")
    print(f"  ROUGE-L  avg        : {df['rougeL_f'].mean():.4f}")
    print(f"  Retrieval sim avg   : {df['top_retrieval_score'].mean():.4f}")
    print(f"  Response time avg   : {df['response_time_s'].mean():.2f}s")
    print(f"\nPer kategori (ROUGE-L avg):")
    per_cat = df.groupby("kategori")["rougeL_f"].mean().sort_values(ascending=False)
    for cat, val in per_cat.items():
        print(f"  {cat:<15}: {val:.4f}")
    print(f"{'='*60}\n")

    return df


def evaluate_topk_experiment(k_values: list = None) -> pd.DataFrame:
    """
    Eksperimen pengaruh top-k terhadap kualitas output.
    Menggunakan 5 test case pertama untuk efisiensi.

    Args:
        k_values: Daftar nilai k yang akan diuji. Default [1, 3, 5].

    Returns:
        DataFrame berisi perbandingan hasil per nilai k.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    pipeline = RAGPipeline()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    sample_cases = TEST_CASES[:5]  # subset untuk efisiensi
    rows = []

    print(f"\n{'='*60}")
    print("  EKSPERIMEN TOP-K")
    print(f"{'='*60}\n")

    for k in k_values:
        r1_list, rl_list, time_list = [], [], []

        for tc in sample_cases:
            start = time.time()
            output = pipeline.generate(
                nama_produk=tc["nama_produk"],
                kategori=tc["kategori"],
                keunggulan=tc["keunggulan"],
                k=k,
            )
            elapsed = time.time() - start

            scores = scorer.score(tc["reference_deskripsi"], output["deskripsi"])
            r1_list.append(scores["rouge1"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)
            time_list.append(elapsed)

        avg_r1 = sum(r1_list) / len(r1_list)
        avg_rl = sum(rl_list) / len(rl_list)
        avg_time = sum(time_list) / len(time_list)

        rows.append(
            {
                "top_k": k,
                "rouge1_avg": round(avg_r1, 4),
                "rougeL_avg": round(avg_rl, 4),
                "response_time_avg_s": round(avg_time, 2),
            }
        )
        print(f"  k={k} | ROUGE-1: {avg_r1:.4f} | ROUGE-L: {avg_rl:.4f} | Time: {avg_time:.2f}s")

    df_topk = pd.DataFrame(rows)

    pathlib.Path("outputs").mkdir(exist_ok=True)
    df_topk.to_csv("outputs/topk_experiment.csv", index=False)
    print(f"\nHasil eksperimen top-k disimpan ke: outputs/topk_experiment.csv")

    return df_topk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluasi UMKM Copywriting Generator")
    parser.add_argument(
        "--mode",
        choices=["full", "topk"],
        default="full",
        help="'full' = evaluasi semua test cases, 'topk' = eksperimen top-k",
    )
    parser.add_argument("--k", type=int, default=3, help="Nilai top-k untuk mode full (default: 3)")
    args = parser.parse_args()

    if args.mode == "full":
        evaluate(top_k=args.k)
    elif args.mode == "topk":
        evaluate_topk_experiment()
