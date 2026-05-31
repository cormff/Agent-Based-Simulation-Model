"""
main_orchestrator.py
====================
Simülasyon döngüsünü yürüten ana orkestratör (v2).

v2 Mimarisi:
    1. Dinamik İnisiyatif: tek turlarda A, çift turlarda B önce hareket eder.
       Bu sayede ilk-hamle önyargısı (first-mover bias) ortadan kalkar.

    2. Deterministik Simetri İndeksi: LLM'den `Symmetry_Index` artık
       istenmez. `compute_symmetry_index()` son 3 turun eylemlerini
       karşılaştırarak matematiksel olarak hesaplar.

    3. Geçmiş Travma Enjeksiyonu: Her turdan önce DB'den "10+ tur öncesi,
       severity >= 8" kriteriyle en yüksek şiddetli kriz sorgulanır ve
       Game Master'ın promptuna <PAST_TRAUMA> bloğu olarak eklenir.
       Bu, denklik bağıntısının GEÇİŞLİLİK / AFFETME boyutunu sınar.
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
# Özel istisnalar
# ===========================================================================
class FatalSimulationError(Exception):
    """
    İlişkinin KOPMASINI temsil eder (beklenen terminal durum).

    Güven eşiği ihlali veya "TERMINATE" hamlesi bu sınıfla raporlanır.
    Gerçek altyapı hataları (LLMClientError) ile karıştırılmamalıdır.
    """


# ===========================================================================
# Oyun-teorik sabitler
# ===========================================================================
ACTION_TRUST_IMPACT: dict[str, float] = {
    "COOPERATE":        +6.0,
    "CONCEDE":          +4.0,
    "NEGOTIATE":        +1.0,
    "DEFECT":          -10.0,
    "EXPLOIT_LOOPHOLE":-14.0,
    "WITHDRAW":         -6.0,
    "TERMINATE":      -100.0,
}
VALID_ACTIONS = set(ACTION_TRUST_IMPACT.keys())

# Deterministik Simetri İndeksi için işbirliği puanları [-1, 1].
# Pozitif = ilişkiyi destekler, negatif = ilişkiyi yıpratır.
_ACTION_COOP_SCORE: dict[str, float] = {
    "COOPERATE":        +1.00,
    "CONCEDE":          +0.50,
    "NEGOTIATE":        +0.25,
    "WITHDRAW":         -0.50,
    "DEFECT":           -0.75,
    "EXPLOIT_LOOPHOLE": -1.00,
    "TERMINATE":        -1.00,
}


# ===========================================================================
# Saf yardımcı fonksiyonlar
# ===========================================================================
def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_action(action: Any) -> str:
    if not isinstance(action, str):
        return "NEGOTIATE"
    a = action.strip().upper().replace(" ", "_")
    return a if a in VALID_ACTIONS else "NEGOTIATE"


def compute_symmetry_index(
    history: list[dict[str, Any]],
    current_a_action: str,
    current_b_action: str,
    window: int = 3,
) -> float:
    """
    Deterministik Simetri İndeksi: son `window` turda (mevcut tur dahil)
    her iki ajanın eylemlerinin ne kadar örtüştüğünü ölçer.

    Formül (her tur için):
        score(agent) = _ACTION_COOP_SCORE[action]   ∈ [-1, 1]
        round_sym    = 1.0 - |score_A - score_B| / 2.0   ∈ [0, 1]
    Simetri İndeksi = ortalama(round_sym, son N tur)

    Yorumlama:
        1.0 → tam örtüşme (ikisi de işbirlikçi VEYA ikisi de toksik)
        0.0 → tam asimetri (biri işbirliği yaparken diğeri sömürüyor)

    Bu formül "matched-but-toxic" (her ikisi de defect) durumunu 1.0'a yakın
    tutar; böylece oyun-teorik "her ikisi de rasyonel" Nash denge noktaları
    simetrik olarak işaretlenir ve ayrıca izlenebilir.
    """
    # Mevcut turu history'ye ekle, son `window` turu al.
    all_rounds = list(history) + [
        {"A": {"action": current_a_action}, "B": {"action": current_b_action}}
    ]
    recent = all_rounds[-window:]

    total = 0.0
    for entry in recent:
        a_act = entry.get("A", {}).get("action", "NEGOTIATE")
        b_act = entry.get("B", {}).get("action", "NEGOTIATE")
        sa = _ACTION_COOP_SCORE.get(a_act, 0.0)
        sb = _ACTION_COOP_SCORE.get(b_act, 0.0)
        # Maksimum olası fark = 2.0 (+1 ile -1 arasında).
        total += 1.0 - abs(sa - sb) / 2.0

    return round(total / len(recent), 4)


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
        rng = random.Random(self.cfg.random_seed)
        self.sim_id = await self.db.create_simulation(
            label=self.label, random_seed=self.cfg.random_seed
        )
        for name, arch in (
            ("A", self.cfg.agent_a_archetype),
            ("B", self.cfg.agent_b_archetype),
        ):
            loophole  = rng.uniform(*arch.loophole_rate_range)
            tolerance = rng.uniform(*arch.tolerance_range)
            trust     = self.cfg.initial_trust

            await self.db.create_agent(
                sim_id=self.sim_id,
                name=name,
                archetype=arch.strategy,
                trust_score=trust,
                loophole_exploitation_rate=loophole,
                tolerance_capacity=tolerance,
            )
            self.trust[name]     = trust
            self.tolerance[name] = tolerance
            self.agents[name]    = AgentClient(
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
            self.label, self.sim_id,
            self.cfg.agent_a_archetype.strategy,
            self.cfg.agent_b_archetype.strategy,
        )

    # -- Güven dinamiği (orkestratör otoritesi) --------------------------
    def _update_trust(
        self, name: str, self_reported: float, partner_action: str, severity: float
    ) -> float:
        """
        Ajanın güven puanını deterministik olarak günceller.
        Tek kaynak orkestratördür; LLM öz-raporu geçmiş değerle harmanlanır,
        ardından karşı tarafın eylemi (tolerance ile yumuşatılmış) eklenir.
        """
        prev   = self.trust[name]
        base   = 0.6 * prev + 0.4 * self_reported
        impact = ACTION_TRUST_IMPACT.get(partner_action, 0.0)
        if impact < 0:
            impact *= (severity / 5.0) * (1.0 - 0.5 * self.tolerance[name])
        new_trust = _clamp(base + impact, 0.0, 100.0)
        self.trust[name] = new_trust
        return new_trust

    # -- Tek tur ---------------------------------------------------------
    async def run_round(
        self, round_number: int, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Bir simülasyon turunu yürütür.

        Adımlar:
            1. Son tur bağlamı + geçmiş travma DB'den çekilir.
            2. Game Master krizi üretir (travma bağlamıyla).
            3. Dinamik inisiyatif: ilk ajan partner_move olmadan hareket eder.
            4. İkinci ajan ilkinin hamlesini görerek reaktif karar verir.
            5. Deterministik Simetri İndeksi hesaplanır (rolling window=3).
            6. Yönlü kenarlar ve güven güncellemeleri kaydedilir.
            7. Çıkış koşulları denetlenir.
        """
        assert self.sim_id is not None

        # (1) Bağlam hazırlığı.
        recent_logs    = await self.db.get_recent_logs(self.sim_id, config.RECENT_LOG_LIMIT)
        agent_summaries = [
            {"name": n, "archetype": self.agents[n].strategy, "trust": round(self.trust[n], 1)}
            for n in ("A", "B")
        ]

        # (2) Geçmiş travma: severity >= 8 ve en az 10 tur öncesi.
        past_trauma = await self.db.get_past_trauma(
            self.sim_id,
            current_round=round_number,
            min_severity=8,
            min_rounds_ago=10,
        )
        if past_trauma:
            logger.debug(
                "[%s] Tur %02d — travma enjeksiyonu: tur=%s sev=%s",
                self.label, round_number,
                past_trauma["round_number"], past_trauma["severity"],
            )

        # (3) Game Master kriz üretir.
        crisis = await self.gm.generate_crisis(
            round_number=round_number,
            agent_summaries=agent_summaries,
            recent_logs=recent_logs,
            past_trauma=past_trauma,
        )
        severity = _as_float(crisis.get("severity"), 5.0)

        # (4) Krizi kaydet.
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

        # (5) Dinamik inisiyatif: tek tur → A önce, çift tur → B önce.
        first_name  = "A" if round_number % 2 != 0 else "B"
        second_name = "B" if first_name == "A" else "A"

        # (6) İlk ajan hareket eder (partner_move yok).
        first_resp = await self.agents[first_name].decide(
            round_number=round_number,
            crisis=crisis,
            current_trust=self.trust[first_name],
            partner_move=None,
            history=history,
        )

        # (7) İkinci ajan ilk ajanın kararını görerek reaktif hareket eder.
        second_resp = await self.agents[second_name].decide(
            round_number=round_number,
            crisis=crisis,
            current_trust=self.trust[second_name],
            partner_move={
                "agent":     first_name,
                "Action":    first_resp.get("Action"),
                "Reasoning": first_resp.get("Reasoning"),
            },
            history=history,
        )

        # A/B ismine normalize et (sıra bağımsız analiz için).
        resp = {
            "A": first_resp  if first_name == "A" else second_resp,
            "B": first_resp  if first_name == "B" else second_resp,
        }

        a_action   = _normalize_action(resp["A"].get("Action"))
        b_action   = _normalize_action(resp["B"].get("Action"))
        a_internal = _clamp(_as_float(resp["A"].get("Internal_Trust_Score"), self.trust["A"]), 0, 100)
        b_internal = _clamp(_as_float(resp["B"].get("Internal_Trust_Score"), self.trust["B"]), 0, 100)
        a_weight   = _clamp(_as_float(resp["A"].get("Influence_Weight"), 0.5), 0.0, 1.0)
        b_weight   = _clamp(_as_float(resp["B"].get("Influence_Weight"), 0.5), 0.0, 1.0)

        # (8) Deterministik Simetri İndeksi (LLM halüsinasyonunu ortadan kaldırır).
        symmetry = compute_symmetry_index(history, a_action, b_action)

        # (9) Yönlü kenarları yaz. Her iki kayıt aynı deterministik simetri değerini alır.
        await self.db.log_round(
            sim_id=self.sim_id,         round_number=round_number,
            source_agent="A",           target_agent="B",
            weight=a_weight,            internal_trust_score=a_internal,
            action=a_action,            reasoning=str(resp["A"].get("Reasoning")),
            symmetry_index=symmetry,    event_id=event_id,
            raw_response=resp["A"].get("raw_response"),
        )
        await self.db.log_round(
            sim_id=self.sim_id,         round_number=round_number,
            source_agent="B",           target_agent="A",
            weight=b_weight,            internal_trust_score=b_internal,
            action=b_action,            reasoning=str(resp["B"].get("Reasoning")),
            symmetry_index=symmetry,    event_id=event_id,
            raw_response=resp["B"].get("raw_response"),
        )

        # (10) Güven güncellemesi: A'nın güveni B'nin eyleminden etkilenir, vice versa.
        new_a = self._update_trust("A", a_internal, b_action, severity)
        new_b = self._update_trust("B", b_internal, a_action, severity)
        await self.db.update_agent_trust(self.sim_id, "A", new_a)
        await self.db.update_agent_trust(self.sim_id, "B", new_b)

        logger.info(
            "[%s] Tur %02d | init=%s | A:%-16s(g=%.1f) B:%-16s(g=%.1f)"
            " | sev=%.0f sym=%.3f trauma=%s",
            self.label, round_number, first_name,
            a_action, new_a, b_action, new_b,
            severity, symmetry,
            f"tur{past_trauma['round_number']}" if past_trauma else "—",
        )

        # (11) Çıkış koşulları (beklenen terminal durum → FatalSimulationError).
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
            "round":         round_number,
            "initiative":    first_name,        # bu tur ilk hamleci
            "A":             {"action": a_action, "trust": round(new_a, 1)},
            "B":             {"action": b_action, "trust": round(new_b, 1)},
            "symmetry_index": symmetry,
            "severity":      severity,
        }

    # -- Tüm koşu --------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        await self.setup()
        assert self.sim_id is not None

        history:   list[dict[str, Any]] = []
        status   = "STABLE"
        reason   = f"{self.cfg.max_rounds} tura ulaşıldı (denklik bağıntısı adayı)."
        completed = 0

        try:
            for r in range(1, self.cfg.max_rounds + 1):
                summary = await self.run_round(r, history)
                history.append(summary)
                completed = r
        except FatalSimulationError as exc:
            status, reason = "FATAL", str(exc)
            logger.warning("[%s] FATAL: %s", self.label, reason)
        except LLMClientError as exc:
            status, reason = "ERROR", f"Altyapı hatası: {exc}"
            logger.error("[%s] ERROR: %s", self.label, reason)

        await self.db.finalize_simulation(self.sim_id, status, completed, reason)

        # Ayrık Matematik: bağıntı özellikleri analizi.
        relation_props: dict[str, Any] = {}
        edges: list[tuple[str, str, float]] = []
        try:
            edges     = await self.db.get_influence_edges(self.sim_id)
            relation  = build_relation_digraph(edges, config.RELATION_EDGE_THRESHOLD)
            relation_props = analyze_relation_properties(relation)
        except Exception as exc:  # noqa: BLE001 — analiz opsiyonel
            logger.warning("[%s] Bağıntı analizi atlandı: %s", self.label, exc)

        return {
            "label":              self.label,
            "sim_id":             self.sim_id,
            "status":             status,
            "reason":             reason,
            "rounds_completed":   completed,
            "influence_edges":    edges,
            "relation_properties": relation_props,
        }


# ===========================================================================
# Üst seviye orkestrasyon
# ===========================================================================
def _default_pair_configs() -> list[config.SimulationConfig]:
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
            print("      Etki Grafı   :", ", ".join(
                f"{s}->{t}:{w:.2f}" for s, t, w in res["influence_edges"]
            ))
    print("\n" + "=" * 72 + "\n")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        configure_gemini(config.GEMINI_API_KEY)
    except RuntimeError as exc:
        logger.error("Gemini yapılandırılamadı: %s", exc)
        return

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
            results = await asyncio.gather(*(r.run() for r in runners), return_exceptions=True)
            _print_report(results)
        finally:
            await game_master.close()


if __name__ == "__main__":
    asyncio.run(main())
