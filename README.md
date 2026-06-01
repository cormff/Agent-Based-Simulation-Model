# Agent-Based Simulation Model (ABM)

İnsan ilişkilerindeki **kriz yönetimini**, Oyun Teorisi (işbirlikçi vs.
sıfır-toplamlı) ve Ayrık Matematik bağıntı prensipleri (**Yansıma, Simetri,
Geçişlilik**) üzerinden test eden, **Edge-to-Cloud Hibrit** bir Ajan Tabanlı
Simülasyon. Akademik makale verisi ve yeni nesil sosyal ağ eşleştirme
algoritmaları için bir **Kavram Kanıtı (PoC)** olarak tasarlanmıştır.

## İçindekiler

- [Mimari](#mimari-3-katman)
- [Dosyalar](#dosyalar)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç (önerilen iş akışı)](#hızlı-başlangıç-önerilen-iş-akışı)
- [1) Tek İlişki İzleme / Debug Modu (`--single`)](#1-tek-ilişki-izleme--debug-modu---single)
- [2) Batch / Veri Fabrikası Modu](#2-batch--veri-fabrikası-modu)
- [3) Veri Analizi ve Görselleştirme](#3-veri-analizi-ve-görselleştirme)
- [CLI Bayrak Referansı](#cli-bayrak-referansı)
- [Yapılandırma (config / .env)](#yapılandırma-config--env)
- [Oyun-Teorik Modeli](#oyun-teorik-modeli)
- [Çıkış Koşulları](#çıkış-koşulları)
- [Veritabanı Şeması](#veritabanı-şeması-networkx-uyumlu)
- [Hataya Dayanıklılık](#hataya-dayanıklılık)

## Mimari (3 Katman)

| Katman | Rol | Teknoloji |
| --- | --- | --- |
| **Orchestrator** | Akış, asenkron API çağrıları, DB yönetimi | `main_orchestrator.py` |
| **Game Master (Edge)** | Bağlamsal kriz/loophole üretimi (yerel, kuantize LLM) | Ollama @ `localhost:11434` |
| **Ajanlar (Cloud)** | Co-op / Zero-Sum karar mekanizması (JSON çıktı) | Gemini API |

## Dosyalar

- **`config.py`** — Merkezi yapılandırma, eşikler, arketipler, `.env` yüklemesi.
- **`db_manager.py`** — Asenkron SQLite şeması + CRUD + NetworkX `DiGraph` ve
  bağıntı (denklik) analizi.
- **`llm_clients.py`** — Asenkron Ollama & Gemini istemcileri, prompt şablonları,
  üstel geri çekilme (backoff) ve LLM tabanlı JSON onarımı.
- **`main_orchestrator.py`** — Simülasyon döngüsünü yürüten ana script
  (Teknik Borç dinamiği, chunk'lı batch çalıştırma ve `--single` debug modu dahil).
- **`data_analyzer.py`** *(V3)* — `simulation.db`'yi okuyup akademik grafikler
  (Kaplan-Meier sağkalım eğrisi + Simetri/Teknik-Borç ısı haritası) üretir.

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env          # GEMINI_API_KEY değerini doldurun

# Yerel Game Master (RTX 3060 8GB için kuantize model önerilir):
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve
```

**Bağımlılıklar:** `google-generativeai`, `aiohttp`, `aiosqlite`, `networkx`,
`python-dotenv` (çekirdek) + `pandas`, `matplotlib`, `seaborn`, `lifelines`
(analiz modülü için).

## Hızlı Başlangıç (önerilen iş akışı)

Akademik bir koşu için tavsiye edilen sıra:

```bash
# 1. Önce TEK bir ilişkiyi izleyip parametreleri/senaryoyu doğrula (ucuz, hızlı)
python main_orchestrator.py --single --step --db debug.db

# 2. Memnunsan TAM batch'i koştur (varsayılan 100 simülasyon -> simulation.db)
python main_orchestrator.py

# 3. Toplanan veriden akademik grafikleri üret (figures/ klasörüne)
python data_analyzer.py
```

> Mantık: Veri fabrikasını (yüzlerce API çağrısı) başlatmadan **önce** tek bir
> ilişkiyi canlı izleyerek eşikleri, arketipleri ve gidişatı ayarlarsınız.
> Debug koşusunu ayrı bir DB'ye (`--db debug.db`) yazarak asıl veri setinizi
> temiz tutarsınız.

---

## 1) Tek İlişki İzleme / Debug Modu (`--single`)

Veri üretmeye başlamadan önce **tek bir ilişkiyi** baştan sona canlı izleyip,
gidişata göre parametre ayarı yapmak için tasarlanmıştır. Her tur sonunda şunlar
okunabilir biçimde basılır:

- **Kriz**: Game Master'ın ürettiği olay metni, türü, şiddeti (1-10), hedefi.
- **Her iki ajanın hamlesi + gerekçesi** (LLM'in kendi açıklaması).
- **Güven değişimi**: eski → yeni puan ve delta (`+`/`-`).
- **Teknik Borç**: o ana dek biriken "bastırılmış gerilim".
- **Simetri İndeksi**: son 3 turun deterministik örtüşme ölçüsü.
- **İlk hamleci** (dinamik inisiyatif: tek tur A, çift tur B).

### Örnek kullanımlar

```bash
# Tek Co-op × Zero-Sum ilişkisini otomatik (duraklamadan) izle
python main_orchestrator.py --single

# ADIM-ADIM mod: her tur sonunda dur ve karar ver
#   Enter = sonraki tur · c = sona kadar koş · q = oturumu durdur
python main_orchestrator.py --single --step

# Farklı çift / tur sayısı / tohum + AYRI debug veritabanı
python main_orchestrator.py --single --pair co_op co_op --rounds 12 --seed 7 --db debug.db

# Co-op × Zero-Sum, 20 tur, ayrı DB
python main_orchestrator.py --single --pair co_op zero_sum --rounds 20 --db debug.db

# Travma enjeksiyonu vb. ayrıntıları görmek için DEBUG seviye loglama
python main_orchestrator.py --single --verbose
```

### Örnek çıktı (bir tur)

```
────────────────────────────────────────────────────────────────────────
  TUR 03  |  ilk hamle: Ajan A  |  sim_id=1  [DEBUG:CO_OPxZERO_SUM]
────────────────────────────────────────────────────────────────────────
  KRİZ [TRUST_TEST · şiddet 9/10 · hedef BOTH · TENSE]
    Ortak bir kaynağın paylaşımında beklenmedik bir hak talebi doğdu...

  Ajan A: CONCEDE  (güven 41.5 → 22.0 [-19.5],  borç 48.0,  etki 0.60)
    gerekçe: İlişkiyi korumak için yine geri adım atıyorum, ama bu sefer...

  Ajan B: EXPLOIT_LOOPHOLE  (güven 74.0 → 76.0 [+2.0],  borç 0.0,  etki 0.60)
    gerekçe: Karşı taraf taviz verdikçe avantajı büyütmek mantıklı.

  Simetri İndeksi: 0.292
────────────────────────────────────────────────────────────────────────
```

> **Önemli:** Gözlemci (observer), çıkış koşullarından **önce** çağrılır; bu sayede
> FATAL'ı (Buffer Overflow / TERMINATE / güven eşiği) **tetikleyen turu da** canlı
> görürsünüz — ilişkinin tam olarak nasıl koptuğunu izleyebilirsiniz. Bu mod batch
> akışını **hiç etkilemez** (`--single` olmadan davranış öncekiyle birebir aynıdır).

---

## 2) Batch / Veri Fabrikası Modu

```bash
python main_orchestrator.py
```

`BATCH_SIZE` adet `Co-op × Zero-Sum` + `BATCH_SIZE` adet `Co-op × Co-op` çifti
(varsayılan **50 + 50 = 100 simülasyon**) çalıştırılır. Özellikler:

- **Chunk'lı çalıştırma:** API kilitlenmelerini önlemek için koşular `CHUNK_SIZE`'lık
  (varsayılan 10) paketler halinde `asyncio.gather` ile koşturulur.
- **İki katmanlı hız sınırı:** `GEMINI_MAX_CONCURRENCY` semaphore'u, chunk'lamanın
  üstüne ek bir eşzamanlılık tavanı koyar.
- **Tekrarlanabilirlik:** Her simülasyon benzersiz ama sabit bir `random_seed` alır.
- **Çıktı:** Sonuçlar `simulation.db` içine yazılır; konsola istatistiksel özet
  (STABLE/FATAL dağılımı, eşleşme türü başına sağkalım yüzdesi, bağıntı özellikleri)
  raporlanır.

Boyutu küçük bir deneme için ortam değişkeniyle batch'i daraltabilirsiniz:

```bash
ABM_BATCH_SIZE=5 ABM_CHUNK_SIZE=5 python main_orchestrator.py   # 10 simülasyon
```

---

## 3) Veri Analizi ve Görselleştirme

Batch koşusu bittikten sonra grafikleri üretmek için:

```bash
python data_analyzer.py                                  # simulation.db -> figures/
python data_analyzer.py --db simulation.db --outdir figures
python data_analyzer.py --db debug.db --outdir debug_figs # debug koşusunu analiz et
```

Üretilen `.png` çıktıları:

- **`survival_curve_kaplan_meier.png`** — `lifelines` ile Kaplan-Meier sağkalım
  eğrisi: `Co-op × Co-op` ve `Co-op × Zero-Sum` eşleşmelerinin tur boyunca
  **STABLE kalma** olasılığı (kopuş = FATAL = olay). İki grup arasındaki fark
  **log-rank testi** (p-değeri) ile raporlanır.
- **`symmetry_vs_technical_debt_heatmap.png`** — `seaborn` ısı haritası: simetri
  indeksi düştükçe teknik borcun (bastırılmış gerilim) nasıl arttığını gösteren
  yoğunluk/korelasyon haritası (Pearson r ile).
- **`symmetry_vs_technical_debt_scatter.png`** — eşleşme türüne göre simetri-borç
  regresyon saçılımı (tamamlayıcı).

> `lifelines` kurulu değilse Kaplan-Meier grafiği zarifçe atlanır; diğer grafikler
> yine de üretilir. `matplotlib` başsız (Agg) backend kullanır, yani sunucu/CI
> ortamında ekran olmadan PNG üretebilir.

---

## CLI Bayrak Referansı

`python main_orchestrator.py [BAYRAKLAR]`

| Bayrak | Değer | Varsayılan | Açıklama |
| --- | --- | --- | --- |
| `--single` | — | (kapalı) | Tek ilişki izleme modunu açar (batch yerine). |
| `--pair A B` | `co_op` / `zero_sum` | `co_op zero_sum` | A ve B ajanlarının arketipleri. |
| `--rounds N` | tamsayı | `config.MAX_ROUNDS` (50) | Maksimum tur sayısı. |
| `--seed N` | tamsayı | `42` | Tekrarlanabilirlik için rastgele tohum. |
| `--step` | — | (kapalı) | Adım-adım: her tur sonunda interaktif duraklama. |
| `--db YOL` | dosya yolu | `config.DB_PATH` | Ayrı debug veritabanı (üretim verisini kirletmez). |
| `--verbose` | — | (kapalı) | DEBUG seviye loglama (travma enjeksiyonu vb.). |
| `-h, --help` | — | — | Yardım metnini ve örnekleri gösterir. |

> `--pair`, `--rounds`, `--seed`, `--step`, `--db` bayrakları yalnızca `--single`
> modunda anlamlıdır. `--single` olmadan tam batch çalışır.

`data_analyzer.py` bayrakları: `--db <yol>` (varsayılan `simulation.db`),
`--outdir <klasör>` (varsayılan `figures`).

---

## Yapılandırma (config / .env)

Tüm parametreler `config.py`'de tanımlıdır ve `.env` (veya ortam değişkenleri) ile
ezilebilir. Başlıca ayarlar:

| Ortam Değişkeni | Varsayılan | Anlamı |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Zorunlu.** Gemini API anahtarı. |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Bulut ajan modeli. |
| `GEMINI_MAX_CONCURRENCY` | `4` | Eşzamanlı Gemini isteği tavanı (semaphore). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Yerel Game Master adresi. |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Edge (yerel) LLM. |
| `ABM_DB_PATH` | `simulation.db` | SQLite veritabanı yolu. |
| `ABM_MAX_ROUNDS` | `50` | Tur üst sınırı (bu tura ulaşan çift STABLE). |
| `ABM_TRUST_THRESHOLD` | `25.0` | Bu eşiğin altına düşen güven → FATAL (kopuş). |
| `ABM_INITIAL_TRUST` | `70.0` | Başlangıç güven puanı (0-100). |
| `ABM_TECHNICAL_DEBT_LIMIT` | `30.0` | **V3:** Bu sınırı aşan teknik borç → Buffer Overflow. |
| `ABM_BATCH_SIZE` | `50` | **V3:** Her arketip eşleşmesinden simülasyon sayısı. |
| `ABM_CHUNK_SIZE` | `10` | **V3:** Aynı anda koşturulacak chunk büyüklüğü. |
| `ABM_RELATION_EDGE_THRESHOLD` | `0.5` | Bağıntı grafında kenar eşiği. |
| `ABM_MAX_RETRIES` | `5` | API yeniden deneme sayısı. |

---

## Oyun-Teorik Modeli

**Arketipler** (`config.py`):

- **Co-op (İşbirlikçi):** İlişkiyi pozitif-toplamlı görür; düşük loophole eğilimi
  (0.05–0.30), yüksek tolerans (0.60–0.95).
- **Zero-Sum (Sıfır-Toplamlı):** Rekabetçi; yüksek loophole eğilimi (0.55–0.95),
  düşük tolerans (0.15–0.50).

**Eylemler ve güven etkileri** (`ACTION_TRUST_IMPACT`):

| Eylem | Güven Etkisi | Teknik Borç tetikler mi? |
| --- | --- | --- |
| `COOPERATE` | +6.0 | — |
| `CONCEDE` | +4.0 | ✔ (güven yine de düşerse) |
| `NEGOTIATE` | +1.0 | — |
| `WITHDRAW` | -6.0 | ✔ (güven yine de düşerse) |
| `DEFECT` | -10.0 | — |
| `EXPLOIT_LOOPHOLE` | -14.0 | — |
| `TERMINATE` | -100.0 | — (doğrudan FATAL) |

**Dinamik inisiyatif:** İlk-hamle önyargısını ortadan kaldırmak için tek turlarda
A, çift turlarda B önce hareket eder.

**Deterministik Simetri İndeksi:** LLM'den istenmez; son 3 turun eylemleri
matematiksel olarak karşılaştırılarak hesaplanır (halüsinasyon engellenir). `1.0` =
tam örtüşme (ikisi de işbirlikçi ya da ikisi de toksik), `0.0` = tam asimetri.

**Geçmiş Travma Enjeksiyonu:** `severity ≥ 8` ve `≥ 10 tur önce` yaşanmış bir kriz,
Game Master'ın promptuna `<PAST_TRAUMA>` bloğu olarak eklenir; denklik bağıntısının
GEÇİŞLİLİK / AFFETME boyutunu sınar.

---

## Çıkış Koşulları

- **Güven eşiği:** Bir ajanın güven puanı `TRUST_THRESHOLD`'un (25) altına düşerse
  → **FATAL** (kopuş) — bu bir çökme değil, anlamlı deney verisidir.
- **TERMINATE:** Bir ajan `TERMINATE` kararı verirse → **FATAL**.
- **(V3) Teknik Borç Taşması (Buffer Overflow):** Bir ajan `CONCEDE`/`WITHDRAW` ile
  taviz verir ama güveni yine de düşerse (içine sindiremediği taviz), aradaki güven
  kaybı `technical_debt` hanesine birikir. Bu borç `TECHNICAL_DEBT_LIMIT`'i
  (varsayılan 30) aşarsa ajan biriken gerilimi boşaltır → **FATAL** (patlama).
- **STABLE:** `MAX_ROUNDS`'a (50) ulaşan çiftler **STABLE** işaretlenir ve denklik
  bağıntısı testine tabi tutulur.

---

## Veritabanı Şeması (NetworkX uyumlu)

`Round_Logs` tablosu doğrudan yönlü kenar (`source_agent → target_agent`,
`weight`) saklar; böylece bir `DiGraph` tek SELECT ile inşa edilir:

```python
graph = await db.build_digraph(sim_id)              # NetworkX DiGraph
edges = await db.get_influence_edges(sim_id)        # [(src, tgt, weight), ...]
```

Tablolar: `Simulations`, `Agents`, `Crisis_Events`, `Round_Logs`. V3 ile
`Agents` ve `Round_Logs` tablolarına `technical_debt (REAL)` sütunu eklenmiştir;
eski veritabanları açılışta otomatik (idempotent) `ALTER TABLE` ile göç ettirilir.

---

## Hataya Dayanıklılık

- Gemini `429 / 5xx / timeout` → **üstel geri çekilme + jitter** ile yeniden deneme.
- Bozuk JSON → deterministik çıkarım → **LLM tabanlı onarım** → güvenli varsayılan.
- `asyncio.Semaphore` ile **proaktif hız sınırlama**; WAL modlu SQLite ile
  eşzamanlı yazma güvenliği.

> **Not (bilimsel geçerlilik):** LLM yanıtları stokastiktir. Tekrarlanabilirlik
> için `SimulationConfig.random_seed` sabitlenebilir; istatistiksel anlamlılık
> için aynı yapılandırma çok sayıda koşturulup toplanmalıdır (bkz. batch modu).
> Geçişlilik bağıntısı yalnızca ≥3 düğümlü kurulumlarda anlamlıdır (2 ajanda trivial).
