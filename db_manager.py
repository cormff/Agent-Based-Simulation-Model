"""
db_manager.py
=============
Asenkron SQLite veri katmanı (aiosqlite tabanlı).

Sorumluluklar:
    * Şema kurulumu (Simulations, Agents, Crisis_Events, Round_Logs).
    * CRUD operasyonları (simülasyon/ajan/kriz/tur kayıtları).
    * NetworkX yönlü graf (DiGraph) analizine uygun veri çekimi
      (Source_Agent, Target_Agent, Weight üçlüsü).
    * Ayrık Matematik bağıntı özellikleri (Yansıma / Simetri / Geçişlilik)
      üzerinden "denklik bağıntısı" testleri.

Tasarım notları:
    * Tüm yazma işlemleri tek bir asyncio.Lock ile serileştirilir; böylece
      birden fazla simülasyon çifti aynı DB dosyasına paralel yazarken
      "database is locked" hatalarının önüne geçilir.
    * WAL (Write-Ahead Logging) modu, eşzamanlı okuma/yazma performansını
      artırmak için etkinleştirilir.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import aiosqlite

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx yoksa graf fonksiyonları devre dışı.
    nx = None  # type: ignore


# ---------------------------------------------------------------------------
# Şema tanımı
# ---------------------------------------------------------------------------
# NetworkX uyumu için Round_Logs tablosu doğrudan (source_agent -> target_agent)
# yönlü kenarları ve `weight` sütununu barındırır. Bu sayede bir DiGraph
# tek bir SELECT ile inşa edilebilir.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS Simulations (
    sim_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT,
    status          TEXT    NOT NULL DEFAULT 'RUNNING',   -- RUNNING|STABLE|FATAL|ERROR
    start_time      REAL    NOT NULL,
    end_time        REAL,
    total_rounds    INTEGER NOT NULL DEFAULT 0,
    end_reason      TEXT,
    random_seed     INTEGER
);

CREATE TABLE IF NOT EXISTS Agents (
    agent_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id                      INTEGER NOT NULL,
    name                        TEXT    NOT NULL,          -- 'A' | 'B'
    archetype                   TEXT    NOT NULL,          -- 'CO_OP' | 'ZERO_SUM'
    trust_score                 REAL    NOT NULL,          -- güncel (orkestratör otoritesi)
    -- Gizli statüler (ajan promptlarına enjekte edilir, dışarıdan görünmez):
    loophole_exploitation_rate  REAL    NOT NULL,
    tolerance_capacity          REAL    NOT NULL,
    FOREIGN KEY (sim_id) REFERENCES Simulations(sim_id)
);

CREATE TABLE IF NOT EXISTS Crisis_Events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id              INTEGER NOT NULL,
    round_number        INTEGER NOT NULL,
    event_text          TEXT    NOT NULL,
    event_type          TEXT,
    severity            INTEGER,
    targeted_agent      TEXT,                              -- 'A' | 'B' | 'BOTH'
    loophole_directive  TEXT,                              -- Game Master'ın gizli yönergesi
    stability_assessment TEXT,
    raw_response        TEXT,
    created_at          REAL    NOT NULL,
    FOREIGN KEY (sim_id) REFERENCES Simulations(sim_id)
);

-- Her tur, her ajan için bir satır. (source_agent -> target_agent) yönlü kenardır.
CREATE TABLE IF NOT EXISTS Round_Logs (
    log_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_id                INTEGER NOT NULL,
    round_number          INTEGER NOT NULL,
    event_id              INTEGER,
    source_agent          TEXT    NOT NULL,                -- etkiyi UYGULAYAN ajan
    target_agent          TEXT    NOT NULL,                -- etkiye MARUZ kalan ajan
    weight                REAL    NOT NULL DEFAULT 0.0,    -- yönlü kenar ağırlığı (etki)
    internal_trust_score  REAL,
    action                TEXT,
    reasoning             TEXT,
    symmetry_index        REAL,
    raw_response          TEXT,
    created_at            REAL    NOT NULL,
    FOREIGN KEY (sim_id)   REFERENCES Simulations(sim_id),
    FOREIGN KEY (event_id) REFERENCES Crisis_Events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_round_logs_sim    ON Round_Logs(sim_id, round_number);
CREATE INDEX IF NOT EXISTS idx_round_logs_edge   ON Round_Logs(sim_id, source_agent, target_agent);
CREATE INDEX IF NOT EXISTS idx_crisis_sim        ON Crisis_Events(sim_id, round_number);
CREATE INDEX IF NOT EXISTS idx_agents_sim        ON Agents(sim_id);
"""


class DatabaseManager:
    """aiosqlite üzerine kurulu asenkron veri erişim nesnesi (DAO)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        # Yazma işlemlerini serileştirmek için (SQLite tek yazar destekler).
        self._write_lock = asyncio.Lock()

    # -- Yaşam döngüsü ----------------------------------------------------
    async def connect(self) -> None:
        """Bağlantıyı açar, WAL modunu ve satır fabrikasını ayarlar."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.commit()

    async def initialize_schema(self) -> None:
        """Tüm tabloları ve indeksleri (yoksa) oluşturur."""
        assert self._conn is not None, "Önce connect() çağrılmalı."
        async with self._write_lock:
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "DatabaseManager":
        await self.connect()
        await self.initialize_schema()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- CREATE -----------------------------------------------------------
    async def create_simulation(
        self, label: str, random_seed: Optional[int] = None
    ) -> int:
        """Yeni bir simülasyon kaydı açar, sim_id döner."""
        assert self._conn is not None
        async with self._write_lock:
            cur = await self._conn.execute(
                "INSERT INTO Simulations (label, status, start_time, random_seed) "
                "VALUES (?, 'RUNNING', ?, ?)",
                (label, time.time(), random_seed),
            )
            await self._conn.commit()
            return int(cur.lastrowid)

    async def create_agent(
        self,
        sim_id: int,
        name: str,
        archetype: str,
        trust_score: float,
        loophole_exploitation_rate: float,
        tolerance_capacity: float,
    ) -> int:
        """Gizli statüleriyle birlikte bir ajan oluşturur, agent_id döner."""
        assert self._conn is not None
        async with self._write_lock:
            cur = await self._conn.execute(
                "INSERT INTO Agents (sim_id, name, archetype, trust_score, "
                "loophole_exploitation_rate, tolerance_capacity) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sim_id,
                    name,
                    archetype,
                    trust_score,
                    loophole_exploitation_rate,
                    tolerance_capacity,
                ),
            )
            await self._conn.commit()
            return int(cur.lastrowid)

    async def log_crisis_event(
        self,
        sim_id: int,
        round_number: int,
        event_text: str,
        event_type: Optional[str],
        severity: Optional[int],
        targeted_agent: Optional[str],
        loophole_directive: Optional[str],
        stability_assessment: Optional[str],
        raw_response: Optional[str],
    ) -> int:
        """Game Master tarafından üretilen kriz olayını kaydeder, event_id döner."""
        assert self._conn is not None
        async with self._write_lock:
            cur = await self._conn.execute(
                "INSERT INTO Crisis_Events (sim_id, round_number, event_text, "
                "event_type, severity, targeted_agent, loophole_directive, "
                "stability_assessment, raw_response, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sim_id,
                    round_number,
                    event_text,
                    event_type,
                    severity,
                    targeted_agent,
                    loophole_directive,
                    stability_assessment,
                    raw_response,
                    time.time(),
                ),
            )
            await self._conn.commit()
            return int(cur.lastrowid)

    async def log_round(
        self,
        sim_id: int,
        round_number: int,
        source_agent: str,
        target_agent: str,
        weight: float,
        internal_trust_score: Optional[float],
        action: Optional[str],
        reasoning: Optional[str],
        symmetry_index: Optional[float],
        event_id: Optional[int] = None,
        raw_response: Optional[str] = None,
    ) -> int:
        """
        Tek bir ajanın bir turdaki kararını yönlü kenar olarak kaydeder.

        (source_agent -> target_agent, weight) üçlüsü, ileride NetworkX
        DiGraph'ında doğrudan kenar olarak kullanılır.
        """
        assert self._conn is not None
        async with self._write_lock:
            cur = await self._conn.execute(
                "INSERT INTO Round_Logs (sim_id, round_number, event_id, "
                "source_agent, target_agent, weight, internal_trust_score, "
                "action, reasoning, symmetry_index, raw_response, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sim_id,
                    round_number,
                    event_id,
                    source_agent,
                    target_agent,
                    weight,
                    internal_trust_score,
                    action,
                    reasoning,
                    symmetry_index,
                    raw_response,
                    time.time(),
                ),
            )
            await self._conn.commit()
            return int(cur.lastrowid)

    # -- UPDATE -----------------------------------------------------------
    async def update_agent_trust(self, sim_id: int, name: str, trust_score: float) -> None:
        """Bir ajanın orkestratör otoritesindeki güncel güven puanını günceller."""
        assert self._conn is not None
        async with self._write_lock:
            await self._conn.execute(
                "UPDATE Agents SET trust_score = ? WHERE sim_id = ? AND name = ?",
                (trust_score, sim_id, name),
            )
            await self._conn.commit()

    async def finalize_simulation(
        self, sim_id: int, status: str, total_rounds: int, end_reason: str
    ) -> None:
        """Simülasyonu sonlandırır (STABLE / FATAL / ERROR)."""
        assert self._conn is not None
        async with self._write_lock:
            await self._conn.execute(
                "UPDATE Simulations SET status = ?, end_time = ?, "
                "total_rounds = ?, end_reason = ? WHERE sim_id = ?",
                (status, time.time(), total_rounds, end_reason, sim_id),
            )
            await self._conn.commit()

    # -- READ -------------------------------------------------------------
    async def get_simulation(self, sim_id: int) -> Optional[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM Simulations WHERE sim_id = ?", (sim_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_agents(self, sim_id: int) -> list[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM Agents WHERE sim_id = ? ORDER BY name", (sim_id,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_agent(self, sim_id: int, name: str) -> Optional[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM Agents WHERE sim_id = ? AND name = ?", (sim_id, name)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_recent_logs(
        self, sim_id: int, limit: int
    ) -> list[dict[str, Any]]:
        """
        Game Master'a bağlam olarak verilecek son N tur kaydını döner.

        Context window'u şişirmemek için yalnızca son `limit` satır çekilir;
        sonuç kronolojik (eskiden yeniye) sırada döndürülür.
        """
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT round_number, source_agent, target_agent, weight, "
            "internal_trust_score, action, reasoning, symmetry_index "
            "FROM Round_Logs WHERE sim_id = ? "
            "ORDER BY log_id DESC LIMIT ?",
            (sim_id, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_last_crisis(self, sim_id: int) -> Optional[dict[str, Any]]:
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT * FROM Crisis_Events WHERE sim_id = ? "
            "ORDER BY event_id DESC LIMIT 1",
            (sim_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_past_trauma(
        self,
        sim_id: int,
        current_round: int,
        min_severity: int = 8,
        min_rounds_ago: int = 10,
    ) -> Optional[dict[str, Any]]:
        """
        Game Master uzun-dönem hafızası için geçmiş travmayı sorgular.

        Kriter:
            - severity >= min_severity  (yüksek şiddetli kriz)
            - round_number <= current_round - min_rounds_ago  (en az 10 tur önce)
            - En yüksek şiddetli, ardından en eski (round_number ASC) seçilir.

        Dönüş None ise travma yok (ya da henüz yeterli tur geçmedi);
        orkestratör bu durumda GM'ye travma bağlamı göndermez.
        """
        assert self._conn is not None
        cutoff_round = current_round - min_rounds_ago
        if cutoff_round <= 0:
            return None
        cur = await self._conn.execute(
            "SELECT * FROM Crisis_Events "
            "WHERE sim_id = ? AND severity >= ? AND round_number <= ? "
            "ORDER BY severity DESC, round_number ASC LIMIT 1",
            (sim_id, min_severity, cutoff_round),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    # -- NetworkX / Ayrık Matematik analizi -------------------------------
    async def get_influence_edges(
        self, sim_id: int
    ) -> list[tuple[str, str, float]]:
        """
        Yönlü graf için (Source_Agent, Target_Agent, Weight) kenarlarını döner.

        Tüm turlar boyunca aynı (source, target) çiftinin ağırlıkları
        ORTALAMASI alınarak tek bir toplu kenara indirgenir. Bu, ilişkinin
        genel etki yönünü ve şiddetini temsil eder.
        """
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT source_agent, target_agent, AVG(weight) AS w "
            "FROM Round_Logs WHERE sim_id = ? "
            "GROUP BY source_agent, target_agent",
            (sim_id,),
        )
        rows = await cur.fetchall()
        return [(r["source_agent"], r["target_agent"], float(r["w"])) for r in rows]

    async def build_digraph(self, sim_id: int) -> "nx.DiGraph":
        """
        Simülasyonun toplu etki grafını bir NetworkX DiGraph olarak inşa eder.

        Düğümler = ajanlar, kenarlar = ortalama etki (weight).
        """
        if nx is None:
            raise RuntimeError("networkx kurulu değil: `pip install networkx`")
        edges = await self.get_influence_edges(sim_id)
        graph = nx.DiGraph()
        for source, target, weight in edges:
            graph.add_edge(source, target, weight=weight)
        return graph


# ---------------------------------------------------------------------------
# Saf fonksiyonlar: Ayrık Matematik bağıntı özellikleri
# ---------------------------------------------------------------------------
def build_relation_digraph(
    edges: list[tuple[str, str, float]],
    threshold: float,
    add_self_loops: bool = True,
) -> "nx.DiGraph":
    """
    Ağırlıklı etki kenarlarından ikili (binary) bir BAĞINTI grafı üretir.

    a R b  <=>  ortalama(weight(a -> b)) >= threshold

    `add_self_loops=True` ise her düğüme yansıma (reflexivity) testinin
    anlamlı olması için bir öz-döngü (self-loop) eklenir: bir ajanın kendisiyle
    olan ilişkisi (öz-güven) tanım gereği bağıntıya dahil edilir.
    """
    if nx is None:
        raise RuntimeError("networkx kurulu değil: `pip install networkx`")
    relation = nx.DiGraph()
    nodes: set[str] = set()
    for source, target, weight in edges:
        nodes.add(source)
        nodes.add(target)
        if weight >= threshold:
            relation.add_edge(source, target, weight=weight)
    relation.add_nodes_from(nodes)
    if add_self_loops:
        for n in nodes:
            relation.add_edge(n, n, weight=1.0)
    return relation


def analyze_relation_properties(relation: "nx.DiGraph") -> dict[str, bool]:
    """
    Bir bağıntının Yansıma / Simetri / Geçişlilik özelliklerini test eder ve
    bunların hepsi sağlanıyorsa "denklik bağıntısı" (equivalence relation)
    olduğunu raporlar.

    Not: Geçişlilik testi anlamlı olabilmesi için en az 3 düğüm gerektirir;
    2 ajanlık kurulumda geçişlilik vakitsiz/trivial olarak değerlendirilir.
    Fonksiyon N düğüm için genelleştirilmiştir (sosyal ağ eşleştirme PoC'si).
    """
    nodes = list(relation.nodes())
    edge_set = set(relation.edges())

    reflexive = all((n, n) in edge_set for n in nodes)
    symmetric = all((v, u) in edge_set for (u, v) in edge_set)
    transitive = all(
        (u, w) in edge_set
        for (u, v) in edge_set
        for (v2, w) in edge_set
        if v == v2
    )

    return {
        "reflexive": reflexive,
        "symmetric": symmetric,
        "transitive": transitive,
        "is_equivalence_relation": reflexive and symmetric and transitive,
        "node_count": len(nodes),
    }
