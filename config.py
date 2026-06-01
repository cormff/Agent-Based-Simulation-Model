"""
config.py
=========
Tüm simülasyon için merkezi yapılandırma katmanı.

Sabitler, eşik değerleri, model isimleri ve API anahtarları tek bir yerden
yönetilir; böylece akademik deneylerde parametre taraması (parameter sweep)
yapmak ve sonuçların tekrarlanabilirliğini (reproducibility) sağlamak kolaylaşır.

Çevresel değişkenler `.env` dosyasından (python-dotenv ile) okunur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    # .env dosyası varsa otomatik yükle (opsiyonel bağımlılık).
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv kurulu değilse sessizce geç.
    pass


# ---------------------------------------------------------------------------
# Veritabanı
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("ABM_DB_PATH", "simulation.db")


# ---------------------------------------------------------------------------
# Game Master (Edge / Local LLM - Ollama)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# RTX 3060 8GB VRAM'e sığacak kuantize bir model öneriliyor (Q4_K_M ~ 4.7GB).
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.85"))


# ---------------------------------------------------------------------------
# Ajanlar (Cloud / Gemini API)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.9"))
# Aynı anda açık tutulacak en fazla Gemini isteği (proaktif hız sınırlama).
GEMINI_MAX_CONCURRENCY: int = int(os.getenv("GEMINI_MAX_CONCURRENCY", "4"))


# ---------------------------------------------------------------------------
# Yeniden Deneme / Üstel Geri Çekilme (Exponential Backoff)
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("ABM_MAX_RETRIES", "5"))
BASE_RETRY_DELAY: float = float(os.getenv("ABM_BASE_RETRY_DELAY", "1.5"))  # saniye
MAX_RETRY_DELAY: float = float(os.getenv("ABM_MAX_RETRY_DELAY", "30.0"))
# Bozuk JSON'u onarmak için LLM'e en fazla kaç kez başvurulacağı
# (sonsuz döngüyü engeller).
MAX_JSON_REPAIR_ATTEMPTS: int = int(os.getenv("ABM_MAX_JSON_REPAIR_ATTEMPTS", "2"))


# ---------------------------------------------------------------------------
# Simülasyon Döngüsü Parametreleri
# ---------------------------------------------------------------------------
MAX_ROUNDS: int = int(os.getenv("ABM_MAX_ROUNDS", "50"))
# Güven puanı bu eşiğin altına düşerse ilişki kopar (Fatal / breakdown).
TRUST_THRESHOLD: float = float(os.getenv("ABM_TRUST_THRESHOLD", "25.0"))
# Başlangıç güven puanı (0-100 ölçeği).
INITIAL_TRUST: float = float(os.getenv("ABM_INITIAL_TRUST", "70.0"))
# Game Master'a bağlam olarak verilecek son log sayısı (context window kontrolü).
RECENT_LOG_LIMIT: int = int(os.getenv("ABM_RECENT_LOG_LIMIT", "6"))
# İlişki "denklik bağıntısı" olarak işaretlenirse ulaşılması gereken tur sayısı.
STABILITY_ROUND_TARGET: int = MAX_ROUNDS

# Yönlü graf (DiGraph) ilişki analizinde bir kenarın "var" sayılması için
# gereken minimum ortalama ağırlık eşiği (Ayrık Matematik bağıntı testi).
RELATION_EDGE_THRESHOLD: float = float(os.getenv("ABM_RELATION_EDGE_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Teknik Borç (Technical Debt) — V3
# ---------------------------------------------------------------------------
# Sosyolojik "bastırılmış duygu / çözülmemiş gerilim" metriği.
# Bir ajan CONCEDE/WITHDRAW ile taviz verir ama güveni yine de düşerse
# (içine sindiremediği taviz), kaybedilen güven kadar puan teknik borca eklenir.
# Bu borç aşağıdaki limiti (Tolerance Capacity tamponu) aşarsa ajan, biriken
# gerilimi boşaltarak TERMINATE kararı verir (Buffer Overflow / patlama).
TECHNICAL_DEBT_LIMIT: float = float(os.getenv("ABM_TECHNICAL_DEBT_LIMIT", "30.0"))


# ---------------------------------------------------------------------------
# İstatistiksel Yığın Çalıştırma (Batch Execution) — V3
# ---------------------------------------------------------------------------
# Her bir arketip eşleşmesinden (örn. "Co-op vs Zero-Sum", "Co-op vs Co-op")
# kaç adet bağımsız simülasyon koşturulacağı. İstatistiksel anlamlılık için
# aynı yapılandırma çok sayıda tekrarlanır (akademik veri fabrikası).
BATCH_SIZE: int = int(os.getenv("ABM_BATCH_SIZE", "50"))
# asyncio.gather'ı tek seferde değil "chunk"lar halinde çalıştırırken her bir
# paketteki eşzamanlı simülasyon (runner) sayısı. API kilitlenmelerini ve
# Ollama/Gemini üzerindeki ani yükü önler (Semaphore limitlerine ek katman).
CHUNK_SIZE: int = int(os.getenv("ABM_CHUNK_SIZE", "10"))


@dataclass
class AgentArchetype:
    """Bir ajan kişiliğinin (oyun-teorik duruşunun) şablonu."""

    name: str
    strategy: str  # "ZERO_SUM" | "CO_OP"
    # Gizli statülerin örnekleneceği aralıklar (min, max).
    loophole_rate_range: tuple[float, float] = (0.0, 1.0)
    tolerance_range: tuple[float, float] = (0.0, 1.0)


# İki temel oyun-teorik arketip: İşbirlikçi (Co-op) ve Sıfır Toplamlı (Zero-Sum).
ARCHETYPE_CO_OP = AgentArchetype(
    name="Co-op",
    strategy="CO_OP",
    loophole_rate_range=(0.05, 0.30),
    tolerance_range=(0.60, 0.95),
)
ARCHETYPE_ZERO_SUM = AgentArchetype(
    name="Zero-Sum",
    strategy="ZERO_SUM",
    loophole_rate_range=(0.55, 0.95),
    tolerance_range=(0.15, 0.50),
)


@dataclass
class SimulationConfig:
    """Tek bir simülasyon koşusunun (bir ajan çiftinin) yapılandırması."""

    agent_a_archetype: AgentArchetype = field(default_factory=lambda: ARCHETYPE_CO_OP)
    agent_b_archetype: AgentArchetype = field(default_factory=lambda: ARCHETYPE_ZERO_SUM)
    max_rounds: int = MAX_ROUNDS
    trust_threshold: float = TRUST_THRESHOLD
    initial_trust: float = INITIAL_TRUST
    random_seed: int | None = None  # Tekrarlanabilirlik için sabitlenebilir.
