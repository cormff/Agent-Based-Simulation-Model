"""
main_orchestrator.py
====================
Simülasyon döngüsünü (Game Loop) yürüten ana orkestratör.

Akış (her tur):
    1. Son tur kayıtları DB'den çekilir.
    2. Game Master (yerel Ollama) bir kriz olayı üretir (event injection).
    3. Kriz, Ajan A'nın Gemini API'sine gönderilir; yanıt JSON olarak ayrıştırılır.
    4. Kriz + A'nın yanıtı, Ajan B'ye asenkron olarak iletilir.
    5. Internal_Trust_Score / Action / Reasoning / Symmetry_Index + yönlü kenar
       ağırlıkları (Weight) Round_Logs'a yazılır.
    6. Çıkış koşulu denetlenir:
         - Güven puanı eşiğin altına düşerse VEYA bir ajan "TERMINATE" derse
           -> ilişki kopar (FATAL). Bu, bir SİSTEM ÇÖKMESİ değil; beklenen ve
           anlamlı bir deney sonucudur (ayrı bir istisna sınıfıyla ayrıştırılır).
         - 50 tura ulaşılırsa -> STABLE (denklik bağıntısı adayı) işaretlenir ve
           Yansıma/Simetri/Geçişlilik analizi fiilen çalıştırılır.

Asenkron paralellik:
    Tek bir çiftte A -> B sıralı bir bağımlılıktır (B, A'nın hamlesini görür).
    Gerçek paralellik, BİRDEN FAZLA ajan çiftinin `asyncio.gather` ile aynı anda
    koşturulmasıyla elde edilir; bu sayede Gemini istekleri kilitlenmeden,
    semaphore ile sınırlandırılarak yönetilir.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

import config
from db_manager import (
    DatabaseManager,
    analyze_relation_properties,
    build_relation_digraph,
)
from llm_clients import (
    AgentClient,
    GameMasterClient,
    JSONValidator,
    LLMClientError,
    configure_gemini,
)

logger = logging.getLogger("orchestrator")


# ===========================================================================
# Özel istisnalar ve yardımcılar
# ===========================================================================
class FatalSimulationError(Exception):
    """
    İlişkinin KOPMASINI temsil eder (beklenen terminal durum).

    DİKKAT: Bu bir altyapı hatası DEĞİLDİR. Güven eşiği ihlali veya "TERMINATE"
    gibi tasarlanmış sonlanma durumlarını, gerçek çökmelerden (LLMClientError)
    ayırmak için kullanılır.
    """


# Karşı tarafın eylemine göre AJANIN güvenine yansıyan etki (oyun-teorik ödül).
ACTION_TRUST_IMPACT: dict[str, float] = {
    "COOPERATE": +6.0,
    "CONCEDE": +4.0,
    "NEGOTIATE": +1.0,
    "DEFECT": -10.0,
    "EXPLOIT_LOOPHOLE": -14.0,
    "WITHDRAW": -6.0,
    "TERMINATE": -100.0,
}
VALID_ACTIONS = set(ACTION_TRUST_IMPACT.keys())


def _as_float(value: Any, default: float) -> float:
    """LLM sayıyı string döndürse bile güvenli float'a çevirir."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_action(action: Any) -> str:
    """Eylemi normalleştirir; tanımsızsa güvenli varsayılana (NEGOTIATE) düşer."""
    if not isinstance(action, str):
        return "NEGOTIATE"
    a = action.strip().upper().replace(" ", "_")
    return a if a in VALID_ACTIONS else "NEGOTIATE"


# ===========================================================================
# Simülasyon Koşucusu (tek bir ajan çifti)
# ===========================================================================
class SimulationRunner:
    """Tek bir A-B çiftinin tüm yaşam döngüsünü yöneten birim."""

    def __init__(
        self,
        db: DatabaseManager,
        game_master: GameMasterClient,
        sim_config: config.SimulationConfig,
        semaphore: asyncio.Semaphore,
        validator: JSONValidator,
        label: str,
    ) -> None:
        self.db = db
        self.gm = game_master
        self.cfg = sim_config
        self.semaphore = semaphore
        self.validator = validator
        self.label = label

        self.sim_id: Optional[int] = None
        self.agents: dict[str, AgentClient] = {}
        self.trust: dict[str, float] = {}
        self.tolerance: dict[str, float] = {}

    # -- Kurulum / İlklendirme -------------------------------------------
    async def setup(self) -> None:
        """DB'de simülasyonu açar, gizli statülerle ajanları oluşturur."""
        rng = random.Random(self.cfg.random_seed)
        self.sim_id = await self.db.create_simulation(
            label=self.label, random_seed=self.cfg.random_seed
        )

        for name, arch in (
            ("A", self.cfg.agent_a_archetype),
            ("B", self.cfg.agent_b_archetype),
        ):
            # Gizli statüleri arketip aralıklarından örnekle.
            loophole = rng.uniform(*arch.loophole_rate_range)
            tolerance = rng.uniform(*arch.tolerance_range)
            trust = self.cfg.initial_trust

            await self.db.create_agent(
                sim_id=self.sim_id,
                name=name,
                archetype=arch.strategy,
                trust_score=trust,
                loophole_exploitation_rate=loophole,
                tolerance_capacity=tolerance,
            )
            self.trust[name] = trust
            self.tolerance[name] = tolerance
            self.agents[name] = AgentClient(
                name=name,
                strategy=arch.strategy,
                model_name=config.GEMINI_MODEL,
                temperature=config.GEMINI_TEMPERATURE,
                loophole_rate=loophole,
                tolerance=tolerance,
                semaphore=self.semaphore,
                validator=self.validator,
                max_retries=config.MAX_RETRIES,
                base_delay=config.BASE_RETRY_DELAY,
                max_delay=config.MAX_RETRY_DELAY,
            )

        logger.info(
            "[%s] sim_id=%s kuruldu | A(%s) vs B(%s)",
            self.label,
            self.sim_id,
            self.cfg.agent_a_archetype.strategy,
            self.cfg.agent_b_archetype.strategy,
        )

    # -- Güven dinamiği (orkestratör otoritesi) --------------------------
    def _update_trust(
        self, name: str, self_reported: float, partner_action: str, severity: float
    ) -> float:
        """
        Ajanın yeni güven puanını DETERMİNİSTİK olarak hesaplar.

        Tek doğruluk kaynağı (single source of truth) orkestratördür: ajanın
        öz-raporu ile geçmiş değer harmanlanır, ardından KARŞI tarafın eylemi
        (oyun-teorik ödül) tolerans kapasitesiyle yumuşatılarak eklenir.
        """
        prev = self.trust[name]
        base = 0.6 * prev + 0.4 * self_reported
        impact = ACTION_TRUST_IMPACT.get(partner_action, 0.0)
        if impact < 0:
            # Negatif etki şiddetle büyür, yüksek toleransla yumuşar.
            impact *= (severity / 5.0) * (1.0 - 0.5 * self.tolerance[name])
        new_trust = _clamp(base + impact, 0.0, 100.0)
        self.trust[name] = new_trust
        return new_trust

    # -- Tek tur ---------------------------------------------------------
    async def run_round(
        self, round_number: int, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Bir simülasyon turunu yürütür; özet sözlük döndürür."""
        assert self.sim_id is not None

        # (1) Bağlam: son loglar + ajan durumları (gizli statüler GM'e verilmez).
        recent_logs = await self.db.get_recent_logs(
            self.sim_id, config.RECENT_LOG_LIMIT
        )
        agent_summaries = [
            {"name": n, "archetype": self.agents[n].strategy, "trust": round(self.trust[n], 1)}
            for n in ("A", "B")
        ]

        # (2) Game Master krizi üretir.
        crisis = await self.gm.generate_crisis(
            round_number=round_number,
            agent_summaries=agent_summaries,
            recent_logs=recent_logs,
        )
        severity = _as_float(crisis.get("severity"), 5.0)

        # (3) Krizi kaydet.
        event_id = await self.db.log_crisis_event(
            sim_id=self.sim_id,
            round_number=round_number,
            event_text=str(crisis.get("event_text")),
            event_type=crisis.get("event_type"),
            severity=int(severity),
            targeted_agent=crisis.get("targeted_agent"),
            loophole_directive=crisis.get("loophole_directive"),
            stability_assessment=crisis.get("stability_assessment"),
            raw_response=crisis.get("raw_response"),
        )

        # (4) Ajan A önce karar verir.
        a_resp = await self.agents["A"].decide(
            round_number=round_number,
            crisis=crisis,
            current_trust=self.trust["A"],
            partner_move=None,
            history=history,
        )
        # (5) Ajan B, A'nın hamlesini görerek asenkron karar verir.
        b_resp = await self.agents["B"].decide(
            round_number=round_number,
            crisis=crisis,
            current_trust=self.trust["B"],
            partner_move={"agent": "A", **{k: a_resp.get(k) for k in ("Action", "Reasoning")}},
            history=history,
        )

        # Sayısal alanları güvenli biçimde normalize et.
        a_action = _normalize_action(a_resp.get("Action"))
        b_action = _normalize_action(b_resp.get("Action"))
        a_internal = _clamp(_as_float(a_resp.get("Internal_Trust_Score"), self.trust["A"]), 0, 100)
        b_internal = _clamp(_as_float(b_resp.get("Internal_Trust_Score"), self.trust["B"]), 0, 100)
        a_sym = _clamp(_as_float(a_resp.get("Symmetry_Index"), 0.5), 0.0, 1.0)
        b_sym = _clamp(_as_float(b_resp.get("Symmetry_Index"), 0.5), 0.0, 1.0)
        a_weight = _clamp(_as_float(a_resp.get("Influence_Weight"), 0.5), 0.0, 1.0)
        b_weight = _clamp(_as_float(b_resp.get("Influence_Weight"), 0.5), 0.0, 1.0)

        # (6) Yönlü kenarları yaz: (kaynak -> hedef, weight) = etki yönü.
        await self.db.log_round(
            sim_id=self.sim_id, round_number=round_number, source_agent="A",
            target_agent="B", weight=a_weight, internal_trust_score=a_internal,
            action=a_action, reasoning=str(a_resp.get("Reasoning")),
            symmetry_index=a_sym, event_id=event_id,
            raw_response=a_resp.get("raw_response"),
        )
        await self.db.log_round(
            sim_id=self.sim_id, round_number=round_number, source_agent="B",
            target_agent="A", weight=b_weight, internal_trust_score=b_internal,
            action=b_action, reasoning=str(b_resp.get("Reasoning")),
            symmetry_index=b_sym, event_id=event_id,
            raw_response=b_resp.get("raw_response"),
        )

        # Güven güncellemesi: A'nın güveni B'NİN eyleminden, B'ninki A'dan etkilenir.
        new_a = self._update_trust("A", a_internal, b_action, severity)
        new_b = self._update_trust("B", b_internal, a_action, severity)
        await self.db.update_agent_trust(self.sim_id, "A", new_a)
        await self.db.update_agent_trust(self.sim_id, "B", new_b)

        logger.info(
            "[%s] Tur %02d | A:%-16s(g=%.1f) B:%-16s(g=%.1f) | sev=%.0f sym=%.2f/%.2f",
            self.label, round_number, a_action, new_a, b_action, new_b,
            severity, a_sym, b_sym,
        )

        # (7) Çıkış koşulları (beklenen terminal durum -> FatalSimulationError).
        if a_action == "TERMINATE" or b_action == "TERMINATE":
            who = "A" if a_action == "TERMINATE" else "B"
            raise FatalSimulationError(
                f"Ajan {who} 'TERMINATE' kararı verdi (tur {round_number})."
            )
        if new_a < self.cfg.trust_threshold or new_b < self.cfg.trust_threshold:
            raise FatalSimulationError(
                f"Güven eşiği ({self.cfg.trust_threshold}) aşıldı "
                f"(A={new_a:.1f}, B={new_b:.1f}) tur {round_number}."
            )

        return {
            "round": round_number,
            "A": {"action": a_action, "trust": round(new_a, 1), "symmetry": a_sym},
            "B": {"action": b_action, "trust": round(new_b, 1), "symmetry": b_sym},
            "severity": severity,
        }

    # -- Tüm koşu --------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        """Çifti baştan sona koşturur, sonunda bağıntı analizini döndürür."""
        await self.setup()
        assert self.sim_id is not None

        history: list[dict[str, Any]] = []
        status = "STABLE"
        reason = f"{self.cfg.max_rounds} tura ulaşıldı (denklik bağıntısı adayı)."
        completed = 0

        try:
            for r in range(1, self.cfg.max_rounds + 1):
                summary = await self.run_round(r, history)
                history.append(summary)
                completed = r
        except FatalSimulationError as exc:
            # Beklenen sonlanma: ilişki koptu (anlamlı deney verisi).
            status, reason = "FATAL", str(exc)
            logger.warning("[%s] FATAL: %s", self.label, reason)
        except LLMClientError as exc:
            # Gerçek altyapı hatası: ayrı statü ile işaretle (çökmeyi gizleme).
            status, reason = "ERROR", f"Altyapı hatası: {exc}"
            logger.error("[%s] ERROR: %s", self.label, reason)

        await self.db.finalize_simulation(self.sim_id, status, completed, reason)

        # --- Ayrık Matematik: bağıntı özellikleri analizi -----------------
        relation_props: dict[str, Any] = {}
        edges: list[tuple[str, str, float]] = []
        try:
            edges = await self.db.get_influence_edges(self.sim_id)
            relation = build_relation_digraph(edges, config.RELATION_EDGE_THRESHOLD)
            relation_props = analyze_relation_properties(relation)
        except Exception as exc:  # noqa: BLE001 - analiz opsiyonel, koşuyu bozmasın.
            logger.warning("[%s] Bağıntı analizi atlandı: %s", self.label, exc)

        return {
            "label": self.label,
            "sim_id": self.sim_id,
            "status": status,
            "reason": reason,
            "rounds_completed": completed,
            "influence_edges": edges,
            "relation_properties": relation_props,
        }


# ===========================================================================
# Üst seviye orkestrasyon
# ===========================================================================
def _default_pair_configs() -> list[config.SimulationConfig]:
    """
    Çalıştırılacak ajan çiftleri. Akademik karşılaştırma için kontrol grupları:
    Co-op vs Zero-Sum (asıl), Co-op vs Co-op ve Zero-Sum vs Zero-Sum (baz çizgi).
    """
    return [
        config.SimulationConfig(
            agent_a_archetype=config.ARCHETYPE_CO_OP,
            agent_b_archetype=config.ARCHETYPE_ZERO_SUM,
            random_seed=42,
        ),
        config.SimulationConfig(
            agent_a_archetype=config.ARCHETYPE_CO_OP,
            agent_b_archetype=config.ARCHETYPE_CO_OP,
            random_seed=7,
        ),
    ]


def _print_report(results: list[Any]) -> None:
    """Konsola özet rapor basar."""
    print("\n" + "=" * 72)
    print("SİMÜLASYON RAPORU")
    print("=" * 72)
    for res in results:
        if isinstance(res, BaseException):
            print(f"  [!] Beklenmeyen hata: {res!r}")
            continue
        print(f"\n  • {res['label']} (sim_id={res['sim_id']})")
        print(f"      Durum        : {res['status']}")
        print(f"      Tamamlanan   : {res['rounds_completed']} tur")
        print(f"      Gerekçe      : {res['reason']}")
        props = res.get("relation_properties") or {}
        if props:
            print(
                "      Bağıntı      : "
                f"yansıma={props.get('reflexive')}, "
                f"simetri={props.get('symmetric')}, "
                f"geçişlilik={props.get('transitive')} "
                f"=> denklik={props.get('is_equivalence_relation')}"
            )
        if res.get("influence_edges"):
            edge_str = ", ".join(
                f"{s}->{t}:{w:.2f}" for s, t, w in res["influence_edges"]
            )
            print(f"      Etki Grafı   : {edge_str}")
    print("\n" + "=" * 72 + "\n")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Gemini SDK yapılandırması (anahtar yoksa anlamlı hata).
    try:
        configure_gemini(config.GEMINI_API_KEY)
    except RuntimeError as exc:
        logger.error("Gemini yapılandırılamadı: %s", exc)
        return

    # Paylaşımlı kaynaklar: tek DB, tek Game Master, tek semaphore + validator.
    semaphore = asyncio.Semaphore(config.GEMINI_MAX_CONCURRENCY)

    async with DatabaseManager(config.DB_PATH) as db:
        game_master = GameMasterClient(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=config.OLLAMA_TEMPERATURE,
            timeout=config.OLLAMA_TIMEOUT,
            max_retries=config.MAX_RETRIES,
            base_delay=config.BASE_RETRY_DELAY,
            max_delay=config.MAX_RETRY_DELAY,
        )
        # Validator, onarım için yerel Game Master'ı kullanır (ucuz LLM repair).
        validator = JSONValidator(
            repair_client=game_master,
            max_repair_attempts=config.MAX_JSON_REPAIR_ATTEMPTS,
        )
        game_master.validator = validator

        try:
            await game_master.start()
            runners = [
                SimulationRunner(
                    db=db,
                    game_master=game_master,
                    sim_config=cfg,
                    semaphore=semaphore,
                    validator=validator,
                    label=f"PAIR-{i+1}:{cfg.agent_a_archetype.strategy}x{cfg.agent_b_archetype.strategy}",
                )
                for i, cfg in enumerate(_default_pair_configs())
            ]
            # asyncio.gather -> çiftler PARALEL koşar; biri çökse diğeri sürer.
            results = await asyncio.gather(
                *(r.run() for r in runners), return_exceptions=True
            )
            _print_report(results)
        finally:
            await game_master.close()


if __name__ == "__main__":
    asyncio.run(main())
