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
- **`main_orchestrator.py`** — Simülasyon döngüsünü yürüten ana script.

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

Varsayılan olarak iki ajan çifti (`Co-op × Zero-Sum` ve `Co-op × Co-op`)
`asyncio.gather` ile **paralel** koşar. Sonuç `simulation.db` içine yazılır ve
konsola bağıntı özellikleri (yansıma/simetri/geçişlilik) raporlanır.

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
