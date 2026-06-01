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

import argparse
import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Optional

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
        observer: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self.db = db
        self.gm = game_master
        self.cfg = sim_config
        self.semaphore = semaphore
        self.validator = validator
        self.label = label
        # Opsiyonel gözlemci: her tur sonunda tüm iç durumu (kriz, gerekçeler,
        # güven deltaları, borç, simetri) alır. Yalnızca debug/single modunda
        # set edilir; batch koşusunda None kalır → davranış değişmez.
        self.observer = observer

        self.sim_id: Optional[int] = None
        self.agents: dict[str, AgentClient] = {}
        self.trust: dict[str, float] = {}
        self.tolerance: dict[str, float] = {}
        self.technical_debt: dict[str, float] = {}

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
            self.trust[name]          = trust
            self.tolerance[name]      = tolerance
            self.technical_debt[name] = 0.0
            self.agents[name]         = AgentClient(
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

        # (9) Güven güncellemesi (log_round öncesine alındı; teknik borç için eski değer gerekli).
        old_a_trust = self.trust["A"]
        old_b_trust = self.trust["B"]
        new_a = self._update_trust("A", a_internal, b_action, severity)
        new_b = self._update_trust("B", b_internal, a_action, severity)

        # (10) Teknik Borç birikimi (V3 — bastırılmış duygu / çözülmemiş gerilim).
        # Ajan CONCEDE veya WITHDRAW yaptı VE güveni yine de düştüyse, tavize
        # rağmen içine sindiremediği gerilim miktarı teknik borca eklenir.
        for _name, _action, _old_t, _new_t in (
            ("A", a_action, old_a_trust, new_a),
            ("B", b_action, old_b_trust, new_b),
        ):
            if _action in ("CONCEDE", "WITHDRAW") and _new_t < _old_t:
                self.technical_debt[_name] = round(
                    self.technical_debt[_name] + (_old_t - _new_t), 4
                )

        # (11) Yönlü kenarları yaz (teknik borç, tur sonu birikimli değeriyle birlikte).
        await self.db.log_round(
            sim_id=self.sim_id,         round_number=round_number,
            source_agent="A",           target_agent="B",
            weight=a_weight,            internal_trust_score=a_internal,
            action=a_action,            reasoning=str(resp["A"].get("Reasoning")),
            symmetry_index=symmetry,    event_id=event_id,
            raw_response=resp["A"].get("raw_response"),
            technical_debt=self.technical_debt["A"],
        )
        await self.db.log_round(
            sim_id=self.sim_id,         round_number=round_number,
            source_agent="B",           target_agent="A",
            weight=b_weight,            internal_trust_score=b_internal,
            action=b_action,            reasoning=str(resp["B"].get("Reasoning")),
            symmetry_index=symmetry,    event_id=event_id,
            raw_response=resp["B"].get("raw_response"),
            technical_debt=self.technical_debt["B"],
        )

        # (12) DB güven ve teknik borç güncellemesi.
        await self.db.update_agent_trust(self.sim_id, "A", new_a)
        await self.db.update_agent_trust(self.sim_id, "B", new_b)
        await self.db.update_agent_technical_debt(self.sim_id, "A", self.technical_debt["A"])
        await self.db.update_agent_technical_debt(self.sim_id, "B", self.technical_debt["B"])

        logger.info(
            "[%s] Tur %02d | init=%s | A:%-16s(g=%.1f d=%.1f) B:%-16s(g=%.1f d=%.1f)"
            " | sev=%.0f sym=%.3f trauma=%s",
            self.label, round_number, first_name,
            a_action, new_a, self.technical_debt["A"],
            b_action, new_b, self.technical_debt["B"],
            severity, symmetry,
            f"tur{past_trauma['round_number']}" if past_trauma else "—",
        )

        # (12.5) Gözlemci (debug/single mod). Çıkış koşullarından ÖNCE çağrılır;
        # böylece FATAL'ı tetikleyen tur da (overflow/terminate dahil) izlenebilir.
        if self.observer is not None:
            await self.observer({
                "label":          self.label,
                "sim_id":         self.sim_id,
                "round":          round_number,
                "initiative":     first_name,
                "crisis": {
                    "event_text":  crisis.get("event_text"),
                    "event_type":  crisis.get("event_type"),
                    "severity":    severity,
                    "targeted":    crisis.get("targeted_agent"),
                    "stability":   crisis.get("stability_assessment"),
                },
                "past_trauma_round": past_trauma["round_number"] if past_trauma else None,
                "A": {
                    "action":      a_action,
                    "reasoning":   str(resp["A"].get("Reasoning")),
                    "trust_old":   round(old_a_trust, 2),
                    "trust_new":   round(new_a, 2),
                    "trust_delta": round(new_a - old_a_trust, 2),
                    "debt":        round(self.technical_debt["A"], 2),
                    "weight":      a_weight,
                },
                "B": {
                    "action":      b_action,
                    "reasoning":   str(resp["B"].get("Reasoning")),
                    "trust_old":   round(old_b_trust, 2),
                    "trust_new":   round(new_b, 2),
                    "trust_delta": round(new_b - old_b_trust, 2),
                    "debt":        round(self.technical_debt["B"], 2),
                    "weight":      b_weight,
                },
                "symmetry_index": symmetry,
            })

        # (13) Çıkış koşulları.
        # Buffer Overflow: birikmiş teknik borç TECHNICAL_DEBT_LIMIT'i aştı.
        overflow = [n for n in ("A", "B") if self.technical_debt[n] > config.TECHNICAL_DEBT_LIMIT]
        if overflow:
            raise FatalSimulationError(
                f"Teknik Borç Taşması (Buffer Overflow) — Ajan(lar) {', '.join(overflow)}: "
                f"A_borç={self.technical_debt['A']:.1f}, B_borç={self.technical_debt['B']:.1f} "
                f"(limit={config.TECHNICAL_DEBT_LIMIT}, tur {round_number})."
            )
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
            "round":          round_number,
            "initiative":     first_name,
            "A":              {"action": a_action, "trust": round(new_a, 1), "debt": round(self.technical_debt["A"], 2)},
            "B":              {"action": b_action, "trust": round(new_b, 1), "debt": round(self.technical_debt["B"], 2)},
            "symmetry_index": symmetry,
            "severity":       severity,
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
            "label":               self.label,
            "sim_id":              self.sim_id,
            "status":              status,
            "reason":              reason,
            "rounds_completed":    completed,
            "influence_edges":     edges,
            "relation_properties": relation_props,
            "pair_type":           (
                f"{self.cfg.agent_a_archetype.strategy}"
                f"x{self.cfg.agent_b_archetype.strategy}"
            ),
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


def _batch_pair_configs() -> list[config.SimulationConfig]:
    """
    BATCH_SIZE adet Co-op×Zero-Sum + BATCH_SIZE adet Co-op×Co-op config üretir.

    Her simülasyon benzersiz bir random_seed alır (tekrarlanabilirlik + varyasyon).
    Akademik veri fabrikası modunda istatistiksel anlamlılık için kullanılır.
    """
    configs: list[config.SimulationConfig] = []
    for i in range(config.BATCH_SIZE):
        configs.append(config.SimulationConfig(
            agent_a_archetype=config.ARCHETYPE_CO_OP,
            agent_b_archetype=config.ARCHETYPE_ZERO_SUM,
            random_seed=1000 + i,
        ))
    for i in range(config.BATCH_SIZE):
        configs.append(config.SimulationConfig(
            agent_a_archetype=config.ARCHETYPE_CO_OP,
            agent_b_archetype=config.ARCHETYPE_CO_OP,
            random_seed=2000 + i,
        ))
    return configs


async def _run_batch_chunked(
    runners: list[SimulationRunner],
    chunk_size: int,
) -> list[Any]:
    """
    SimulationRunner listesini `chunk_size`'lık paketler halinde çalıştırır.

    Semaphore Gemini API eşzamanlılığını sınırlarken, chunk'lama asyncio görev
    kuyruğunu yönetilebilir tutar ve Ollama üzerindeki ani yükü dağıtır.
    """
    all_results: list[Any] = []
    total      = len(runners)
    num_chunks = (total + chunk_size - 1) // chunk_size
    for i in range(0, total, chunk_size):
        chunk     = runners[i : i + chunk_size]
        chunk_idx = i // chunk_size + 1
        logger.info(
            "=== Batch Chunk %d/%d başlatılıyor (%d simülasyon) ===",
            chunk_idx, num_chunks, len(chunk),
        )
        chunk_results = await asyncio.gather(
            *(r.run() for r in chunk), return_exceptions=True
        )
        all_results.extend(chunk_results)
        ok = sum(
            1 for r in chunk_results
            if isinstance(r, dict) and r.get("status") in ("STABLE", "FATAL")
        )
        logger.info(
            "=== Chunk %d/%d tamamlandı: %d/%d başarılı ===",
            chunk_idx, num_chunks, ok, len(chunk),
        )
    return all_results


def _print_report(results: list[Any]) -> None:
    from collections import Counter

    print("\n" + "=" * 72)
    print("SİMÜLASYON RAPORU")
    print("=" * 72)

    successful = [r for r in results if isinstance(r, dict)]
    errors     = [r for r in results if isinstance(r, BaseException)]
    verbose    = len(successful) <= 10

    for res in results:
        if isinstance(res, BaseException):
            print(f"  [!] Beklenmeyen hata: {res!r}")
            continue
        if verbose:
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
        else:
            icon = "✓" if res["status"] == "STABLE" else "✗"
            print(
                f"  {icon} {res['label']:20s} | {res['status']:6s}"
                f" | {res['rounds_completed']:3d} tur | {res['reason'][:55]}"
            )

    if len(successful) > 1:
        print("\n" + "-" * 72)
        print("İSTATİSTİKSEL ÖZET")
        print("-" * 72)
        status_counts = Counter(r["status"] for r in successful)
        rounds_vals   = [r["rounds_completed"] for r in successful]
        print(f"  Toplam           : {len(results)} sim | Hata: {len(errors)}")
        print(f"  Durum dağılımı   : {dict(status_counts)}")
        print(
            f"  Tur istatistiği  : ort={sum(rounds_vals)/len(rounds_vals):.1f}"
            f"  min={min(rounds_vals)}  max={max(rounds_vals)}"
        )
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in successful:
            key = r.get("pair_type", "?x?")
            groups.setdefault(key, []).append(r)
        for pt, grp in sorted(groups.items()):
            stable = sum(1 for r in grp if r["status"] == "STABLE")
            pct    = 100 * stable // len(grp) if grp else 0
            print(
                f"  {pt:30s}: {len(grp):3d} sim"
                f" | STABLE={stable} ({pct}%) | FATAL={len(grp) - stable}"
            )

    print("\n" + "=" * 72 + "\n")


# ===========================================================================
# Debug / Tek İlişki İzleme (single-run modu)
# ===========================================================================
_ARCHETYPE_BY_KEY: dict[str, config.AgentArchetype] = {
    "CO_OP":    config.ARCHETYPE_CO_OP,
    "COOP":     config.ARCHETYPE_CO_OP,
    "C":        config.ARCHETYPE_CO_OP,
    "ZERO_SUM": config.ARCHETYPE_ZERO_SUM,
    "ZEROSUM":  config.ARCHETYPE_ZERO_SUM,
    "Z":        config.ARCHETYPE_ZERO_SUM,
}


def _resolve_archetype(key: str) -> config.AgentArchetype:
    """'co_op' / 'zero_sum' / 'c' / 'z' gibi serbest girdileri arketipe çevirir."""
    norm = key.strip().upper().replace("-", "_").replace(" ", "_")
    if norm not in _ARCHETYPE_BY_KEY:
        valid = "co_op | zero_sum"
        raise ValueError(f"Bilinmeyen arketip {key!r}. Geçerli: {valid}")
    return _ARCHETYPE_BY_KEY[norm]


def _make_observer(step: bool) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """
    Tek-ilişki modunda her tur sonu çağrılan, insan-okur bir konsol gözlemcisi.

    `step=True` ise her turdan sonra duraklar (interaktif adım-adım izleme):
        [Enter] sonraki tur · 'c' sona kadar devam · 'q' durdur.
    `step=False` ise yalnızca zengin bir tur özeti basar (otomatik akış).
    """
    state = {"continue_to_end": False}

    def _fmt_delta(d: float) -> str:
        return f"+{d:.1f}" if d >= 0 else f"{d:.1f}"

    async def observer(frame: dict[str, Any]) -> None:
        cr = frame["crisis"]
        print("\n" + "─" * 72)
        print(
            f"  TUR {frame['round']:02d}  |  ilk hamle: Ajan {frame['initiative']}"
            f"  |  sim_id={frame['sim_id']}  [{frame['label']}]"
        )
        if frame.get("past_trauma_round"):
            print(f"  ⚠ Geçmiş travma enjekte edildi (kaynak tur {frame['past_trauma_round']})")
        print("─" * 72)
        print(
            f"  KRİZ [{cr['event_type']} · şiddet {cr['severity']:.0f}/10 · "
            f"hedef {cr['targeted']} · {cr['stability']}]"
        )
        print(f"    {cr['event_text']}")
        for nm in ("A", "B"):
            ag = frame[nm]
            print(
                f"\n  Ajan {nm}: {ag['action']}"
                f"  (güven {ag['trust_old']:.1f} → {ag['trust_new']:.1f}"
                f" [{_fmt_delta(ag['trust_delta'])}],"
                f"  borç {ag['debt']:.1f},  etki {ag['weight']:.2f})"
            )
            print(f"    gerekçe: {ag['reasoning']}")
        print(f"\n  Simetri İndeksi: {frame['symmetry_index']:.3f}")
        print("─" * 72)

        if not step or state["continue_to_end"]:
            return
        # İnteraktif duraklama — event loop'u bloklamamak için thread'e devret.
        choice = (await asyncio.to_thread(
            input, "  [Enter]=sonraki tur  ·  c=sona kadar  ·  q=durdur > "
        )).strip().lower()
        if choice == "q":
            raise FatalSimulationError("Kullanıcı debug oturumunu durdurdu (q).")
        if choice == "c":
            state["continue_to_end"] = True

    return observer


async def run_single(
    pair: tuple[config.AgentArchetype, config.AgentArchetype],
    rounds: Optional[int] = None,
    seed: Optional[int] = 42,
    step: bool = False,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Tek bir ilişkiyi (A-B çifti) izlenebilir biçimde çalıştırır.

    Veri fabrikasını (batch) başlatmadan önce bir senaryoyu canlı izlemek,
    parametre ayarı yapmak ve gidişata göre müdahale etmek için tasarlanmıştır.

    Args:
        pair:     (A_arketip, B_arketip) — örn. (CO_OP, ZERO_SUM).
        rounds:   Maks tur (None → config.MAX_ROUNDS).
        seed:     Tekrarlanabilirlik için sabit tohum.
        step:     True → her tur sonunda interaktif duraklama.
        db_path:  Ayrı bir debug DB yolu (None → config.DB_PATH).
                  Test koşusunun üretim verisini kirletmemesi için önerilir.

    Batch akışına dokunmaz; mevcut SimulationRunner'ı `observer` ile besler.
    """
    a_arch, b_arch = pair
    sim_cfg = config.SimulationConfig(
        agent_a_archetype=a_arch,
        agent_b_archetype=b_arch,
        max_rounds=rounds if rounds is not None else config.MAX_ROUNDS,
        random_seed=seed,
    )
    label = f"DEBUG:{a_arch.strategy}x{b_arch.strategy}"
    target_db = db_path or config.DB_PATH

    print("\n" + "=" * 72)
    print("TEK İLİŞKİ İZLEME MODU (single-run debug)")
    print("=" * 72)
    print(f"  Çift        : {a_arch.name} (A) vs {b_arch.name} (B)")
    print(f"  Maks tur    : {sim_cfg.max_rounds}")
    print(f"  Tohum (seed): {seed}")
    print(f"  Adım modu   : {'AÇIK (interaktif)' if step else 'kapalı (otomatik)'}")
    print(f"  Veritabanı  : {target_db}")
    print(f"  Eşikler     : güven<{sim_cfg.trust_threshold}  borç>{config.TECHNICAL_DEBT_LIMIT}")
    print("=" * 72)

    configure_gemini(config.GEMINI_API_KEY)
    semaphore = asyncio.Semaphore(config.GEMINI_MAX_CONCURRENCY)

    async with DatabaseManager(target_db) as db:
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
            runner = SimulationRunner(
                db=db,
                game_master=game_master,
                sim_config=sim_cfg,
                semaphore=semaphore,
                validator=validator,
                label=label,
                observer=_make_observer(step),
            )
            result = await runner.run()
            _print_report([result])
            return result
        finally:
            await game_master.close()


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
            pair_cfgs = _batch_pair_configs()
            logger.info(
                "Batch başlatılıyor: %d simülasyon (%d Co-op×Zero-Sum + %d Co-op×Co-op)"
                " | chunk_size=%d | semaphore=%d",
                len(pair_cfgs), config.BATCH_SIZE, config.BATCH_SIZE,
                config.CHUNK_SIZE, config.GEMINI_MAX_CONCURRENCY,
            )
            runners = []
            for i, cfg in enumerate(pair_cfgs):
                label = (
                    f"CZ-{i + 1:03d}"
                    if i < config.BATCH_SIZE
                    else f"CC-{i - config.BATCH_SIZE + 1:03d}"
                )
                runners.append(SimulationRunner(
                    db=db,
                    game_master=game_master,
                    sim_config=cfg,
                    semaphore=semaphore,
                    validator=validator,
                    label=label,
                ))
            results = await _run_batch_chunked(runners, config.CHUNK_SIZE)
            _print_report(results)
        finally:
            await game_master.close()


# ===========================================================================
# CLI giriş noktası
# ===========================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ABM orkestratörü. Varsayılan: tam batch (veri fabrikası). "
            "--single ile tek bir ilişkiyi izleyip ayar yapabilirsiniz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python main_orchestrator.py                      # tam batch (100 sim)\n"
            "  python main_orchestrator.py --single             # tek Co-op×Zero-Sum izle\n"
            "  python main_orchestrator.py --single --step      # adım-adım (Enter ile ilerle)\n"
            "  python main_orchestrator.py --single --pair co_op co_op --rounds 12\n"
            "  python main_orchestrator.py --single --db debug.db --seed 7\n"
        ),
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Tek bir ilişkiyi izleme modunda çalıştır (batch yerine).",
    )
    parser.add_argument(
        "--pair", nargs=2, metavar=("A", "B"), default=["co_op", "zero_sum"],
        help="Ajan arketipleri: co_op | zero_sum (varsayılan: co_op zero_sum).",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help=f"Maks tur sayısı (varsayılan: config.MAX_ROUNDS={config.MAX_ROUNDS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Tekrarlanabilirlik için rastgele tohum (varsayılan: 42).",
    )
    parser.add_argument(
        "--step", action="store_true",
        help="Adım-adım mod: her turdan sonra interaktif duraklama.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Ayrı debug veritabanı yolu (üretim verisini kirletmemek için).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="DEBUG seviye loglama (travma enjeksiyonu vb. ayrıntılar).",
    )
    return parser


def _run_cli() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.single:
        asyncio.run(main())
        return

    try:
        pair = (_resolve_archetype(args.pair[0]), _resolve_archetype(args.pair[1]))
    except ValueError as exc:
        raise SystemExit(f"[HATA] {exc}")

    try:
        asyncio.run(run_single(
            pair=pair,
            rounds=args.rounds,
            seed=args.seed,
            step=args.step,
            db_path=args.db,
        ))
    except RuntimeError as exc:
        # configure_gemini başarısızsa (API anahtarı yok) anlamlı bir mesaj ver.
        raise SystemExit(f"[HATA] {exc}")
    except KeyboardInterrupt:
        print("\n[İPTAL] Debug oturumu kullanıcı tarafından kesildi.")


if __name__ == "__main__":
    _run_cli()
