# Agent-Based Simulation Model (ABM)

İnsan ilişkilerindeki **kriz yönetimini**, Oyun Teorisi (işbirlikçi vs.
sıfır-toplamlı) ve Ayrık Matematik bağıntı prensipleri (**Yansıma, Simetri,
Geçişlilik**) üzerinden test eden, **Edge-to-Cloud Hibrit** bir Ajan Tabanlı
Simülasyon. Akademik makale verisi ve yeni nesil sosyal ağ eşleştirme
algoritmaları için bir **Kavram Kanıtı (PoC)** olarak tasarlanmıştır.

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
  (Teknik Borç dinamiği + chunk'lı batch çalıştırma dahil).
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

## Çalıştırma

```bash
python main_orchestrator.py
```

**V3 (Akademik Veri Fabrikası):** `BATCH_SIZE` adet `Co-op × Zero-Sum` + `BATCH_SIZE`
adet `Co-op × Co-op` çifti (varsayılan 50+50 = **100 simülasyon**) çalıştırılır.
API kilitlenmelerini önlemek için koşular `CHUNK_SIZE`'lık (varsayılan 10) paketler
halinde `asyncio.gather` ile koşturulur; `GEMINI_MAX_CONCURRENCY` semaphore'u
ek bir hız sınırı katmanı sağlar. Sonuç `simulation.db` içine yazılır ve konsola
istatistiksel özet (STABLE/FATAL dağılımı, bağıntı özellikleri) raporlanır.

### Tek İlişki İzleme / Debug Modu (`--single`)

Veri fabrikasını başlatmadan önce **tek bir ilişkiyi** canlı izleyip parametre
ayarı yapmak için `--single` modunu kullanın. Her tur sonunda kriz metni, her iki
ajanın gerekçesi, güven değişimi (eski → yeni, delta), biriken teknik borç ve
simetri indeksi okunabilir biçimde basılır:

```bash
# Tek Co-op × Zero-Sum ilişkisini otomatik izle
python main_orchestrator.py --single

# Adım-adım: her tur sonunda dur (Enter=sonraki · c=sona kadar · q=durdur)
python main_orchestrator.py --single --step

# Farklı çift / tur / tohum + AYRI debug DB (üretim verisini kirletmez)
python main_orchestrator.py --single --pair co_op co_op --rounds 12 --seed 7 --db debug.db

# Travma enjeksiyonu gibi ayrıntılar için DEBUG loglama
python main_orchestrator.py --single --verbose
```

> İpucu: `--db debug.db` ile test koşularını `simulation.db`'den ayrı tutun;
> batch veri setiniz temiz kalır. Gözlemci (observer) çıkış koşullarından **önce**
> çağrıldığı için, FATAL'ı (Buffer Overflow / TERMINATE) tetikleyen turu da
> canlı görebilirsiniz. Batch modu (`--single` olmadan) hiç etkilenmez.

### Veri Analizi ve Görselleştirme (V3)

Batch koşusu bittikten sonra grafikleri üretmek için:

```bash
python data_analyzer.py                       # simulation.db -> figures/
python data_analyzer.py --db simulation.db --outdir figures
```

Üretilen `.png` çıktıları:

- **`survival_curve_kaplan_meier.png`** — `lifelines` ile Kaplan-Meier sağkalım
  eğrisi: `Co-op × Co-op` ve `Co-op × Zero-Sum` eşleşmelerinin tur boyunca
  **STABLE kalma** olasılığı (kopuş = FATAL = olay), log-rank p-değeriyle.
- **`symmetry_vs_technical_debt_heatmap.png`** — `seaborn` ısı haritası: simetri
  indeksi düştükçe teknik borcun (bastırılmış gerilim) nasıl arttığını gösteren
  yoğunluk/korelasyon haritası.
- **`symmetry_vs_technical_debt_scatter.png`** — eşleşme türüne göre simetri-borç
  regresyon saçılımı (tamamlayıcı).

## Veritabanı Şeması (NetworkX uyumlu)

`Round_Logs` tablosu doğrudan yönlü kenar (`source_agent → target_agent`,
`weight`) saklar; böylece bir `DiGraph` tek SELECT ile inşa edilir:

```python
graph = await db.build_digraph(sim_id)              # NetworkX DiGraph
edges = await db.get_influence_edges(sim_id)        # [(src, tgt, weight), ...]
```

## Çıkış Koşulları

- Bir ajanın güven puanı **eşiğin altına** düşerse veya **`TERMINATE`** kararı
  verirse → ilişki **FATAL** (kopuş) — bu bir çökme değil, anlamlı deney verisidir.
- **(V3) Teknik Borç Taşması (Buffer Overflow):** Bir ajan `CONCEDE`/`WITHDRAW` ile
  taviz verir ama güveni yine de düşerse (içine sindiremediği taviz), aradaki güven
  kaybı `technical_debt` hanesine birikir. Bu borç `TECHNICAL_DEBT_LIMIT`'i (varsayılan
  30) aşarsa ajan biriken gerilimi boşaltır → **FATAL** (patlama).
- **50 tura** ulaşan çiftler **STABLE** işaretlenir ve denklik bağıntısı testine
  tabi tutulur.

## Hataya Dayanıklılık

- Gemini `429 / 5xx / timeout` → **üstel geri çekilme + jitter** ile yeniden deneme.
- Bozuk JSON → deterministik çıkarım → **LLM tabanlı onarım** → güvenli varsayılan.
- `asyncio.Semaphore` ile **proaktif hız sınırlama**; WAL modlu SQLite ile
  eşzamanlı yazma güvenliği.

> **Not (bilimsel geçerlilik):** LLM yanıtları stokastiktir. Tekrarlanabilirlik
> için `SimulationConfig.random_seed` sabitlenebilir; istatistiksel anlamlılık
> için aynı yapılandırma çok sayıda koşturulup toplanmalıdır. Geçişlilik
> bağıntısı yalnızca ≥3 düğümlü kurulumlarda anlamlıdır (2 ajanda trivial).
