# Homework 2 — Prompt Engineering ve RAG ile Makine Çevirisi — Yol Haritası

> **Ders:** LLM Driven Software Development
> **Teslim Tarihi:** 21/05/2026
> **Konu:** Küçük instruction-tuned LLM'ler ile İngilizce ↔ Türkçe makine çevirisi; prompt engineering ve RAG karşılaştırması
> **Bu dokümanın amacı:** Ödevin uygulanma stratejisini, kullanılacak teknolojileri ve adımları onayınıza sunmaktır. Onay sonrası implementasyon bu plana göre yürütülecektir.

---

## 1. Ödevin Genel Çerçevesi (Analiz)

Ödev, küçük instruction-tuned bir LLM ile WMT16 EN↔TR makine çevirisi üzerinde üç farklı prompting yaklaşımının COMET metriğine göre karşılaştırılmasını istemektedir:

| # | Yaklaşım | Açıklama |
|---|----------|----------|
| 1 | **Zero-shot prompting** | Modelin yalnızca direktif ile çeviri yapması (baseline) |
| 2 | **Makale stratejisi** | "Exploring Human-Like Translation Strategy with LLMs" makalesindeki çok adımlı prompting (görev ayrıştırma + LLM-as-Judge) |
| 3 | **RAG tabanlı dinamik 5-shot** | Test cümlesine semantik olarak benzer 5 örneğin vektör veritabanından çekilip few-shot olarak prompt'a eklenmesi |

Ödev 5 ana bölümden oluşuyor:
- **Part 1:** Dataset hazırlığı (WMT16 EN↔TR)
- **Part 2:** Model seçimi ve donanım gerekçesi (Qwen 2.5 7B Instruct / Mistral 7B Instruct)
- **Part 3:** Prompt engineering — literatür özeti + iki prompting deseni (Task Decomposition, LLM-as-Judge) + COMET tanımı + zero-shot vs. makale stratejisi karşılaştırması
- **Part 4:** RAG mimarisi tasarımı + implementasyonu
- **Part 5:** Üç yaklaşımın COMET ile karşılaştırılması ve tartışma

**Teslim edilecekler:** Çalışan kod, deney sonuçları (COMET skorları), sistem diyagramı, raporda istenen tüm bölümlerin yazılı açıklamaları.

---

## 2. Önerilen Teknoloji Yığını (Stack)

### 2.1 Çekirdek
| Bileşen | Tercih | Gerekçe |
|---------|--------|---------|
| Programlama dili | **Python 3.10+** | ML/NLP ekosisteminin merkezi |
| Notebook/Script | **Jupyter + .py modüller** | Bölümler bağımsız çalıştırılabilir |
| Paket yönetimi | **uv** veya **conda** | Hızlı, reproducible env (CUDA uyumu için conda da olabilir) |
| Versiyon kontrolü | **git** | Deney varyantlarını branch'lerle yönetmek |

### 2.2 Model & Çıkarım (Inference)
| Bileşen | Tercih | Fallback | Gerekçe |
|---------|--------|----------|---------|
| LLM | **Qwen 2.5 7B Instruct** | — | Onaylandı |
| Inference engine | **Hugging Face `transformers` + `accelerate`** | vLLM (yalnızca süre çok uzarsa) | Kullanıcı tercihi: tanıdık API ile başla; performans problemi olursa vLLM'e geçiş hazır tut |
| Quantization | **bitsandbytes 4-bit (NF4)** | AWQ/GPTQ 4-bit | Colab T4'te transformers ile sorunsuz çalışır, ek model dosyası indirme gerektirmez |
| Batching | `transformers.pipeline` veya `model.generate` batch | — | Çok cümle aynı anda; padding-left aktif edilecek |
| Tokenizer | `transformers` AutoTokenizer | — | Qwen 2.5'in default tokenizer'ı (chat template ile) |

### 2.3 Dataset & Veri İşleme
| Bileşen | Tercih |
|---------|--------|
| Dataset yükleme | **`datasets`** (Hugging Face) — `wmt16` config: `tr-en` |
| Veri işleme | **pandas**, **numpy** |
| Cümle temizleme | regex, `unicodedata`, opsiyonel **sacremoses** (detokenization) |

### 2.4 Değerlendirme (Evaluation)
| Bileşen | Tercih | Gerekçe |
|---------|--------|---------|
| COMET | **`unbabel-comet`** (PyPI: `unbabel-comet`) | Resmi paket, `wmt22-comet-da` referans model |
| (Opsiyonel) BLEU | **`sacrebleu`** | Karşılaştırma raporunda yardımcı olabilir |

### 2.5 RAG Yığını
| Bileşen | Tercih | Alternatif | Gerekçe |
|---------|--------|------------|---------|
| Vektör DB | **FAISS** (CPU index) | ChromaDB | FAISS daha hızlı, lokal dosya tabanlı, dependency hafif |
| Embedding modeli | **`intfloat/multilingual-e5-base`** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `BAAI/bge-m3` | E5 multilingual çok dilli görevlerde güçlü, EN↔TR için uygun |
| Embedding kütüphanesi | **`sentence-transformers`** | — | Hugging Face uyumlu, kolay API |

### 2.6 Yardımcı
| Bileşen | Tercih |
|---------|--------|
| Loglama | Python `logging` + tqdm progress |
| Konfigürasyon | YAML (`config.yaml`) veya `pydantic-settings` |
| Sistem diyagramı | **Mermaid** (Markdown içinde) veya **draw.io** |
| Rapor | Markdown + PDF export (Pandoc) |

---

## 3. Donanım Stratejisi (Kesinleşti)

**Çalışma ortamı: Google Colab (T4 16GB ücretsiz veya Pro V100/A100).**

| Bileşen | Karar |
|---------|-------|
| Platform | Google Colab |
| GPU (free) | NVIDIA T4 16GB VRAM |
| GPU (Pro varsa) | V100 / A100 (oturum süresi 24 saat, idle daha toleranslı) |
| Model | Qwen 2.5 7B Instruct + bitsandbytes NF4 (≈5GB VRAM) |
| Inference | `transformers` + batched `generate` |
| Disk | Google Drive bağla (model cache + sonuçlar buraya yazılsın → oturum kopsa bile kayıp olmaz) |

**Önemli operasyonel notlar:**
- Colab Free oturumu 12 saatte kopar, idle ~90 dk. Her deneyin **checkpoint atması** gerekiyor — örn. her 100 cümlede `outputs.jsonl`'a flush.
- Drive mount: `from google.colab import drive; drive.mount('/content/drive')`
- Çalışma dizini Drive altında olsun: `/content/drive/MyDrive/RAG_Project/`
- Lokal dizin (`C:\Users\Mustafa\Desktop\RAG Project`) **sadece geliştirme/git senkronizasyonu** için; gerçek çalışma Colab'da.

---

## 4. Önerilen Proje Dizin Yapısı

```
RAG Project/
├── YOL_HARITASI.md              # bu doküman
├── README.md                    # nihai rapor (Türkçe + İngilizce başlıklar)
├── requirements.txt             # bağımlılıklar
├── config.yaml                  # model adı, dataset split, k, paths
├── data/
│   ├── raw/                     # WMT16'dan indirilen ham veriler (cache)
│   ├── processed/               # temizlenmiş train/test
│   └── samples.json             # rapora gidecek örnek input-output'lar
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # WMT16 yükleme + preprocessing
│   ├── model.py                 # LLM wrapper (vLLM veya transformers)
│   ├── prompts/
│   │   ├── zero_shot.py
│   │   ├── maps_strategy.py     # makale prompt'ları (decomposition + judge)
│   │   └── rag_few_shot.py
│   ├── rag/
│   │   ├── embedder.py          # multilingual-e5 wrapper
│   │   ├── indexer.py           # FAISS index oluşturma
│   │   └── retriever.py         # top-k similarity search
│   ├── evaluation/
│   │   ├── comet_scorer.py
│   │   └── runner.py            # tüm yaklaşımları batch çalıştırır
│   └── utils.py
├── experiments/
│   ├── 01_zero_shot.py
│   ├── 02_paper_strategy.py
│   ├── 03_rag_5shot.py
│   └── results/
│       ├── zero_shot_outputs.jsonl
│       ├── paper_outputs.jsonl
│       ├── rag_outputs.jsonl
│       └── comet_scores.json
├── notebooks/
│   ├── 01_dataset_eda.ipynb     # Part 1 görselleştirme
│   ├── 02_paper_summary.ipynb   # Part 3.A literatür özeti taslağı
│   └── 03_results_analysis.ipynb
├── diagrams/
│   └── rag_architecture.md      # Mermaid diyagramı
└── report/
    └── final_report.md          # tüm bölümlerin yazılı raporu (teslim)
```

---

## 5. Adım Adım Uygulama Planı

### Faz 0 — Ortam Hazırlığı (yarım gün)
- [ ] Python venv/conda env oluştur (`py310-rag`)
- [ ] CUDA + PyTorch eşleşmesi doğrula
- [ ] `requirements.txt` hazırla ve kur
- [ ] Hugging Face token (gerekirse model erişimi için)
- [ ] Donanım üzerinde küçük "hello-world" inference testi (model yükleniyor mu?)

### Faz 1 — Part 1: Dataset (yarım gün)
- [ ] `datasets.load_dataset("wmt16", "tr-en")` ile veriyi indir
- [ ] Train/Validation/Test split istatistiklerini çıkar (cümle sayısı, ortalama token, dil dağılımı)
- [ ] Preprocessing pipeline:
  - boş/duplicate cümleleri at
  - aşırı uzun (>200 token) ve aşırı kısa (<3 token) cümleleri filtrele
  - HTML/URL/encoding artıklarını temizle
  - opsiyonel: dil tespiti ile yanlış etiketli çiftleri ayıkla (`langdetect`)
- [ ] **Test seti** ayrı tutulur (yalnızca değerlendirmede kullanılır)
- [ ] Rapor için 5-10 örnek input-output çifti seç
- [ ] **Çıktı:** `data/processed/` + EDA notebook'u

### Faz 2 — Part 2: Model (yarım gün)
- [ ] Qwen 2.5 7B Instruct'ı yükle (4-bit AWQ veya GGUF)
- [ ] Donanım üzerinde inference benchmark (saniyede cümle, VRAM kullanımı)
- [ ] Model seçimi gerekçesini yaz: HW, MT uygunluğu, Türkçe desteği
- [ ] **Çıktı:** raporun Part 2 bölümü

### Faz 3 — Part 3: Prompt Engineering
**3.A — Literatür Özeti (yarım gün)**
- [ ] `Article.pdf` (varsayım: "Exploring Human-Like Translation Strategy with LLMs" — MAPS framework) tam okunur
- [ ] Motivasyon, önerilen strateji (knowledge mining: keywords/topics/demonstrations + selection via QE), bulgular özetlenir
- [ ] **Not:** Makale farklı çıkarsa stratejiyi makaledeki orijinal yaklaşıma birebir adapte edeceğiz

**3.B — Prompting Pattern Açıklamaları (yarım gün)**
- [ ] "Break Complex Tasks into Simpler Subtasks": tanım + neden işe yarar + makalede nasıl uygulanıyor
- [ ] "LLM as a Judge": tanım + değerlendirme/refine mekaniği + avantaj/dezavantaj + makaledeki kullanımı

**3.C — COMET Metriği (1-2 saat)**
- [ ] COMET çalışma prensibi (cross-lingual encoder + regression head, referans/kaynak/aday üçlüsü)
- [ ] BLEU ile karşılaştırma tablosu (n-gram vs. learned, insan korelasyonu, dil bağımsızlığı)

**3.D — Deneysel Değerlendirme (1-2 gün)**

MAPS pipeline'ı makalede şöyle tanımlanmış (bizim için sadeleştirilmiş):

```
INPUT: source cümlesi (örn. İngilizce)

Aşama 1 — KNOWLEDGE MINING (3 paralel LLM çağrısı, her biri 5-shot ile)
  1.a) Keywords: kaynak cümleden anahtar kelime çiftleri çıkar
       (örn. "machine translation" → "makine çevirisi")
  1.b) Topics: cümlenin konu/tonu nedir?
  1.c) Demonstrations: bu cümleye benzer 3 örnek çift üret

Aşama 2 — KNOWLEDGE INTEGRATION (4 LLM çağrısı)
  Baseline candidate:   bilgi olmadan zero-shot çeviri
  Candidate K:          keywords bağlamı ile çeviri
  Candidate T:          topics bağlamı ile çeviri
  Candidate D:          demonstrations bağlamı ile çeviri
  → 4 aday üretilir

Aşama 3 — KNOWLEDGE SELECTION (1 LLM çağrısı veya 1 QE skoru)
  Yöntem: LLM-SCQ (LLM-as-Judge, single choice question)
          → modele 4 adayı verip "hangisi en iyi?" diye sor
  Alternatif: COMET-QE (wmt22-cometkiwi-da, reference-free)
  → Final translation seçilir
```

- [ ] Zero-shot prompt (EN→TR ve TR→EN için tek esnek template; chat formatlı Qwen instruction)
- [ ] MAPS prompt'larını implemente et: 4 ayrı template (mining keywords/topics/demos + integration için baseline/K/T/D + selection için SCQ)
- [ ] MAPS dipnot 1'e göre **manuel 5-shot exemplar** her mining adımı için hazırla (toplam 15 örnek)
- [ ] **Selection methodu seçimi:**
  - **Birincil: LLM-SCQ** (extra model gerektirmez, hızlıdır) — kullanıcının ödev başlığındaki "LLM-as-Judge" beklentisiyle örtüşür
  - **Yedek: COMET-QE** (wmt22-cometkiwi-da, ~600MB) — daha güvenilir ama Colab indirme süresi
- [ ] Test subset (kararlaştırılan boyutta) üzerinde batched inference, her cümle için JSONL'a yaz (`{src, ref, baseline, cand_k, cand_t, cand_d, selected, method, direction}`)
- [ ] COMET skorlama: **`wmt22-comet-da`** referans-tabanlı (~2.3GB)
- [ ] **Çıktı:** `experiments/results/{zero_shot,maps}_outputs.jsonl` + `comet_scores.json` + Markdown tablo

### Faz 4 — Part 4: RAG (1-2 gün)
**4.A — Mimari Tasarım**
- [ ] Akış: query embed → FAISS top-k=5 → prompt'a 5-shot olarak yerleştir → LLM generate
- [ ] Mermaid diyagramı çiz:
  ```
  Test cümlesi → [Embedder] → vektör
                                 ↓
  WMT16 train pool → [Embedder] → [FAISS Index] → top-5 örnek
                                                       ↓
                            [Prompt Builder: 5-shot] → [LLM] → Çeviri
  ```
- [ ] İndeksleme, retrieval, prompt construction, generation süreçlerini yaz

**4.B — Implementasyon**
- [ ] **Corpus hazırlığı:** WMT16 train setinden sabit seed ile 50 000 çift örnekle (preprocessing pipeline uygulanmış olmalı)
- [ ] **EN ve TR için ayrı iki index oluştur:**
  - `index_en.faiss` → İngilizce cümlelerin embedding'i, retrieve TR→EN sırasında kullanılır (kaynak=İngilizce değil; kaynak Türkçe iken EN→TR'de İngilizceyi indexlemek mantıksız — düzelt: kaynak dil = indekslenecek dil)
  - **Doğrusu:** Her iki yön için kaynak dile karşılık gelen embedding'i kullan. EN→TR için index_en (İngilizce cümleler), TR→EN için index_tr (Türkçe cümleler).
- [ ] `intfloat/multilingual-e5-base` ile cümleleri embed et (E5 için passage prefix: `"passage: "`, query prefix: `"query: "`)
- [ ] FAISS **IndexFlatIP** (cosine için L2-normalize edilmiş vektörlerle) — 50K için brute-force yeterli, HNSW ileride gerekirse
- [ ] Test cümlesini embed et (query prefix ile), top-5 benzer (src, tgt) çiftini al
- [ ] **Prompt template (esnek, çift yönlü):**
  ```
  You are a professional translator. Translate the given sentence accurately.

  Here are some example translations:
  1. {src_lang}: {src1}
     {tgt_lang}: {tgt1}
  2. {src_lang}: {src2}
     {tgt_lang}: {tgt2}
  ...
  5. {src_lang}: {src5}
     {tgt_lang}: {tgt5}

  Now translate:
  {src_lang}: {input}
  {tgt_lang}:
  ```
- [ ] Batched inference (test subset üzerinde)
- [ ] COMET skorlama
- [ ] **Çıktı:** `experiments/results/rag_outputs.jsonl` + retrieval analizi (örn. tipik bir retrieve örneği rapora konulur)

### Faz 5 — Part 5: Karşılaştırma ve Rapor (1 gün)
- [ ] Üç yaklaşımı tek tabloda göster (COMET, çıkarım süresi, token tüketimi)
- [ ] İyileşme/regresyon analizi: hangi cümle tiplerinde RAG kazandı, hangilerinde kaybetti
- [ ] Computational overhead: embedding + retrieval ek maliyeti, çok-adımlı promptun token maliyeti
- [ ] RAG'in güçlü/zayıf yönleri (alan-spesifik terimler, kalıp ifadeler vs. tamamen yeni cümleler)
- [ ] Nihai raporu `report/final_report.md` altında derle

### Faz 6 — Teslim Kontrolü (yarım gün)
- [ ] Tüm scriptlerin tek komutla reproducible olduğunu doğrula
- [ ] `README.md`'a çalıştırma talimatları
- [ ] Sonuç dosyalarının teslimde olduğundan emin ol
- [ ] Raporun ödevdeki her başlığa karşılık geldiğini checklist ile doğrula

**Tahmini toplam süre:** 6-8 iş günü (donanıma bağlı)

---

## 6. Kararlaştırılmış Konfigürasyon

| # | Konu | Karar |
|---|------|-------|
| 1 | Platform | **Google Colab** (T4 free; Pro varsa V100/A100) |
| 2 | Model | **Qwen 2.5 7B Instruct** (bitsandbytes NF4 4-bit) |
| 3 | Çeviri yönü | **Çift yönlü (EN↔TR).** Tek bir esnek prompt, kaynak dile göre hedefi otomatik belirler |
| 4 | Test seti | **Subset.** Tam set yerine sabit-seed alt küme. Ön benchmark ile boyut netleşir (aşağıda 7. bölüm) |
| 5 | RAG corpus | **WMT16 train setinden subset** (test setinden ASLA — data leakage). Önerilen boyut: 50 000-100 000 çift |
| 6 | Inference engine | **`transformers`** (öncelik). Süre tahammül sınırını aşarsa vLLM'e geçiş hazır |
| 7 | Rapor dili | **İngilizce** |
| 8 | Makale | **MAPS** — He et al., TACL 2024 (Vol. 12, 229-246), DOI: 10.1162/tacl_a_00642 |

### Subset Boyutları (öneri — onayınıza)

| Veri | Önerilen boyut | Amaç |
|------|----------------|------|
| RAG index corpus (train'den) | **50 000** çift | Çeşitlilik yeterli, FAISS belleğe rahat sığar (~50K × 768 dim ≈ 150 MB) |
| Test subset (her yön için) | **500 cümle** (toplam 1000) | 3 yaklaşımın hepsi makul sürede biter, sabit seed=42 |
| Knowledge mining 5-shot exemplars (MAPS) | makaledeki gibi **manuel olarak 5'er adet** (keywords / topics / demos için ayrı ayrı) | MAPS makalesi de manuel hazırlamış (dipnot 1, s. 4) |

---

## 7. Ön Benchmark Adımı (Faz 2 sonunda zorunlu)

Tam subset boyutuna karar vermeden önce küçük bir kalibrasyon koşusu yapılacak:

- 50 cümlelik mini-test seçilir
- Üç yaklaşım da bu 50 cümle üzerinde çalıştırılır
- Ölçülecekler:
  - Cümle başına ortalama süre (s)
  - GPU bellek tüketimi (`nvidia-smi`)
  - Toplam token üretimi
- **Karar kuralı:**
  - Zero-shot ≤ 3 s/cümle ise → 1000 cümle/yön = 2000 toplam yapılabilir
  - MAPS ≤ 15 s/cümle ise → 500 cümle/yön, kabul edilebilir (~2 saat × 2 = 4 saat / yön)
  - Aşılırsa → vLLM'e migrate veya subset'i 250 cümleye indir
- **Çıktı:** `experiments/benchmark.md` — kalibrasyon sonuçları + nihai subset boyutu kararı

Bu adım hem deneyin gerçekçi planlanmasını sağlar hem de raporda "compute budget" gerekçelendirmesi olarak yer alır.

---

## 8. Riskler ve Hafifletme Stratejileri

| Risk | Olasılık | Etki | Hafifletme |
|------|----------|------|-----------|
| VRAM yetersizliği | Orta | Yüksek | 4-bit quantization, batch=1, GGUF/llama.cpp fallback |
| COMET model indirme problemi (~2GB, gated olabilir) | Düşük | Orta | `wmt22-comet-da` (açık) ile başla; `Unbabel/wmt22-cometkiwi-da` referanssız alternatif |
| Tam test setinde inference süresi çok uzun | Orta | Orta | Subset + reproducible seed; ödev gereği subset belirtilirse not düşülür |
| RAG retrieve edilen örneklerin kalitesizliği | Orta | Orta | Embedding modelinin EN ve TR yönünde test edilmesi; gerekirse `bge-m3` ile değiştirme |
| Çok-adımlı prompt'un (MAPS) toplam token maliyeti | Yüksek | Orta | Prompt'ları sıkıştır, max_new_tokens sınırla, batch ile paralelle |
| Colab oturum kopması | Yüksek | Yüksek | Drive'a checkpoint, JSONL incremental flush, resume mantığı |
| RAG retrieve edilen örnek dilinin yanlış yöne ait olması | Düşük | Orta | EN ve TR için ayrı index'ler — dikkatle test edilir |
| MAPS knowledge mining adımının çıktısının format dışı olması | Orta | Orta | Output parser'a regex/fallback ekle, hatalı parse'ları boş knowledge olarak işle |

---

## 9. Başarı Kriterleri

Ödev "tamamlandı" sayılabilmesi için:

- [ ] Üç yaklaşımın hepsi aynı test seti üzerinde çalışır ve COMET skoru üretir
- [ ] RAG sistemi reproducible (sabit seed, sabit index) ve diyagramı raporda
- [ ] Rapor, ödevin **her** alt başlığını (1.a-d, 2.a-c, 3.A-D, 4.A-B, 5) açık şekilde karşılar
- [ ] Kod modüler, README ile bir tek komutla çalıştırılabilir
- [ ] Sonuçlar tablo/grafik ile sunulmuş, tartışma bölümü kanıta dayalı

---

## Sıradaki Adım

Bu yol haritasını gözden geçirip:
1. **Bölüm 6**'daki sorulara cevap verirseniz,
2. Ekleme/çıkarma önerilerinizi belirtirseniz,

Faz 0'dan başlayarak implementasyona geçeceğim.
