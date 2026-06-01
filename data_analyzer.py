"""
data_analyzer.py
================
V3 — Akademik Veri Analiz ve Görselleştirme Modülü.

`simulation.db` içindeki batch koşu verisini okuyup iki temel akademik grafik üretir:

    a) Kaplan-Meier Sağkalım Eğrisi (Survival Curve):
       Co-op×Co-op ile Co-op×Zero-Sum eşleşmelerinin tur sayısı (Rounds)
       boyunca "STABLE kalma" (hayatta kalma) olasılığını karşılaştırır.
       Olay (event / kopuş) = simülasyonun FATAL bitmesi.
       Sansürleme (censoring) = MAX_ROUNDS'a STABLE ulaşan koşular.

    b) Simetri vs. Teknik Borç Isı Haritası (Heatmap):
       Ajanların simetri indeksi düştükçe teknik borçlarının nasıl arttığını
       (negatif korelasyon) gösteren 2B yoğunluk / korelasyon haritası.

Kullanım:
    python data_analyzer.py
    python data_analyzer.py --db simulation.db --outdir figures

Bağımlılıklar: pandas, matplotlib, seaborn, lifelines (bkz. requirements.txt).
`lifelines` kurulu değilse Kaplan-Meier grafiği zarifçe atlanır; diğer grafikler
yine de üretilir.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Optional

import pandas as pd

import matplotlib

# Başsız (headless) ortamlarda (sunucu / CI) ekran olmadan PNG üretebilmek için.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    _HAS_LIFELINES = True
except ImportError:  # pragma: no cover - lifelines opsiyonel
    KaplanMeierFitter = None  # type: ignore
    logrank_test = None  # type: ignore
    _HAS_LIFELINES = False


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = os.getenv("ABM_DB_PATH", "simulation.db")
DEFAULT_OUTDIR = "figures"

# Eşleşme türü etiketleri (label ön ekleri ve pair_type'tan türetilir).
PAIR_CO_OP_VS_CO_OP = "CO_OPxCO_OP"
PAIR_CO_OP_VS_ZERO_SUM = "CO_OPxZERO_SUM"

sns.set_theme(style="whitegrid", context="talk")


# ===========================================================================
# 1) Veri Erişim Katmanı
# ===========================================================================
def connect(db_path: str) -> sqlite3.Connection:
    """simulation.db dosyasına salt-okunur amaçlı bir bağlantı açar."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Veritabanı bulunamadı: {db_path!r}. Önce simülasyonu çalıştırın "
            f"(python main_orchestrator.py)."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _classify_pair_type(row: pd.Series) -> str:
    """
    Bir simülasyon satırını eşleşme türüne göre sınıflandırır.

    Önce ajan arketiplerine (Agents tablosundan) bakar; yoksa `label` ön ekine
    (CC-* = Co-op×Co-op, CZ-* = Co-op×Zero-Sum) düşer.
    """
    a_arch = row.get("a_archetype")
    b_arch = row.get("b_archetype")
    if isinstance(a_arch, str) and isinstance(b_arch, str):
        archset = {a_arch.upper(), b_arch.upper()}
        if archset == {"CO_OP"}:
            return PAIR_CO_OP_VS_CO_OP
        if archset == {"CO_OP", "ZERO_SUM"}:
            return PAIR_CO_OP_VS_ZERO_SUM
        return f"{a_arch}x{b_arch}"

    label = str(row.get("label", "") or "")
    if label.startswith("CC"):
        return PAIR_CO_OP_VS_CO_OP
    if label.startswith("CZ"):
        return PAIR_CO_OP_VS_ZERO_SUM
    return "UNKNOWN"


def load_simulations(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Her simülasyon için bir satır döndüren, sağkalım analizine hazır DataFrame.

    Sütunlar:
        sim_id, label, status, total_rounds, end_reason,
        a_archetype, b_archetype, pair_type,
        duration  (sağkalım süresi = total_rounds),
        event     (1 = kopuş/FATAL gözlemlendi, 0 = sansürlü/STABLE).
    """
    sims = pd.read_sql_query(
        "SELECT sim_id, label, status, total_rounds, end_reason FROM Simulations",
        conn,
    )
    if sims.empty:
        return sims

    # Ajan arketiplerini A/B olarak geniş (wide) formata çevir.
    agents = pd.read_sql_query(
        "SELECT sim_id, name, archetype FROM Agents", conn
    )
    if not agents.empty:
        wide = agents.pivot_table(
            index="sim_id", columns="name", values="archetype", aggfunc="first"
        )
        wide = wide.rename(
            columns={"A": "a_archetype", "B": "b_archetype"}
        ).reset_index()
        sims = sims.merge(wide, on="sim_id", how="left")
    else:
        sims["a_archetype"] = None
        sims["b_archetype"] = None

    sims["pair_type"] = sims.apply(_classify_pair_type, axis=1)
    # Sağkalım süresi = tamamlanan tur sayısı.
    sims["duration"] = sims["total_rounds"].clip(lower=0)
    # Olay = ilişkinin KOPMASI (FATAL). STABLE = sansürlü (henüz kopmadı).
    # ERROR durumları analiz dışı bırakılır (altyapı hatası, gerçek olay değil).
    sims["event"] = (sims["status"] == "FATAL").astype(int)
    return sims


def load_round_logs(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Round_Logs + Simulations birleşimini (her tur, her ajan satırı) döndürür.

    symmetry_index ve technical_debt korelasyonu için kullanılır.
    """
    df = pd.read_sql_query(
        """
        SELECT
            rl.sim_id,
            rl.round_number,
            rl.source_agent,
            rl.action,
            rl.symmetry_index,
            rl.technical_debt,
            rl.internal_trust_score,
            rl.weight,
            s.status,
            s.label
        FROM Round_Logs rl
        JOIN Simulations s ON s.sim_id = rl.sim_id
        """,
        conn,
    )
    if not df.empty:
        df["pair_type"] = df.apply(_classify_pair_type, axis=1)
    return df


# ===========================================================================
# 2) Grafik A: Kaplan-Meier Sağkalım Eğrisi
# ===========================================================================
def plot_survival_curve(
    sims: pd.DataFrame, outdir: str
) -> Optional[str]:
    """
    Co-op×Co-op vs Co-op×Zero-Sum eşleşmelerinin Kaplan-Meier sağkalım
    eğrisini çizer ve PNG olarak kaydeder.

    Hayatta kalma = ilişkinin STABLE kalması; olay (event) = FATAL kopuş.
    İki grup arasındaki fark log-rank testiyle (p-değeri) raporlanır.

    `lifelines` kurulu değilse None döner (zarif atlama).
    """
    if not _HAS_LIFELINES:
        print(
            "[UYARI] lifelines kurulu değil; Kaplan-Meier grafiği atlanıyor. "
            "Kurmak için: pip install lifelines"
        )
        return None

    # Yalnızca anlamlı iki eşleşme türünü ve geçerli (STABLE/FATAL) koşuları al.
    valid = sims[sims["status"].isin(["STABLE", "FATAL"])]
    groups = {
        "Co-op vs Co-op": valid[valid["pair_type"] == PAIR_CO_OP_VS_CO_OP],
        "Co-op vs Zero-Sum": valid[valid["pair_type"] == PAIR_CO_OP_VS_ZERO_SUM],
    }
    groups = {k: v for k, v in groups.items() if not v.empty}
    if not groups:
        print("[UYARI] Sağkalım analizi için yeterli veri yok; grafik atlanıyor.")
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    kmf = KaplanMeierFitter()
    palette = {"Co-op vs Co-op": "#2a9d8f", "Co-op vs Zero-Sum": "#e76f51"}

    for name, grp in groups.items():
        kmf.fit(
            durations=grp["duration"],
            event_observed=grp["event"],
            label=f"{name} (n={len(grp)})",
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=palette.get(name),
            linewidth=2.5,
        )

    # İki grup varsa log-rank testi ile istatistiksel anlamlılık ekle.
    if len(groups) == 2:
        (g1, d1), (g2, d2) = list(groups.items())
        try:
            res = logrank_test(
                d1["duration"], d2["duration"],
                event_observed_A=d1["event"], event_observed_B=d2["event"],
            )
            p = res.p_value
            sig = "anlamlı" if p < 0.05 else "anlamsız"
            ax.text(
                0.02, 0.05,
                f"Log-rank p = {p:.4g} ({sig})",
                transform=ax.transAxes,
                fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[UYARI] Log-rank testi hesaplanamadı: {exc}")

    ax.set_title(
        "Kaplan-Meier Sağkalım Eğrisi\nİlişki Stabilitesi (Co-op vs Co-op vs. Co-op vs Zero-Sum)",
        fontsize=15,
    )
    ax.set_xlabel("Tur (Round)")
    ax.set_ylabel("Hayatta Kalma (STABLE) Olasılığı")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()

    path = os.path.join(outdir, "survival_curve_kaplan_meier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Kaplan-Meier sağkalım eğrisi kaydedildi: {path}")
    return path


# ===========================================================================
# 3) Grafik B: Simetri vs. Teknik Borç Isı Haritası
# ===========================================================================
def plot_symmetry_debt_heatmap(
    logs: pd.DataFrame, outdir: str
) -> Optional[str]:
    """
    Simetri indeksi (düşüş) ile teknik borç (artış) arasındaki ilişkiyi gösteren
    2B yoğunluk ısı haritası (binned heatmap) çizer ve PNG olarak kaydeder.

    Hücre rengi = o (simetri-aralığı, borç-aralığı) gözesindeki gözlem sayısı.
    Negatif korelasyon beklenir: simetri düştükçe teknik borç birikir.
    Pearson korelasyon katsayısı grafiğe eklenir.
    """
    df = logs.dropna(subset=["symmetry_index", "technical_debt"]).copy()
    if df.empty:
        print(
            "[UYARI] symmetry_index / technical_debt verisi yok; "
            "ısı haritası atlanıyor."
        )
        return None

    # Sürekli değerleri ayrık kovalara (bin) ayır → ısı haritası matrisi.
    sym_bins = pd.cut(df["symmetry_index"], bins=[i / 10 for i in range(0, 11)])
    max_debt = max(float(df["technical_debt"].max()), 1.0)
    debt_edges = [round(i * max_debt / 8, 1) for i in range(9)]
    debt_bins = pd.cut(df["technical_debt"], bins=debt_edges, include_lowest=True)

    pivot = (
        df.assign(_sym=sym_bins, _debt=debt_bins)
        .pivot_table(index="_debt", columns="_sym", values="sim_id",
                     aggfunc="count", observed=False)
        .fillna(0)
        .sort_index(ascending=False)  # yüksek borç üstte
    )

    corr = df["symmetry_index"].corr(df["technical_debt"])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        cmap="rocket_r",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Gözlem Sayısı (tur-ajan)"},
        ax=ax,
    )
    ax.set_title(
        "Simetri İndeksi vs. Teknik Borç Yoğunluk Isı Haritası\n"
        f"(Pearson r = {corr:.3f} — simetri düştükçe borç artar)",
        fontsize=15,
    )
    ax.set_xlabel("Simetri İndeksi Aralığı (düşük → yüksek)")
    ax.set_ylabel("Teknik Borç Aralığı (yüksek → düşük)")
    fig.tight_layout()

    path = os.path.join(outdir, "symmetry_vs_technical_debt_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Simetri-Teknik Borç ısı haritası kaydedildi: {path}")
    return path


def plot_symmetry_debt_correlation(
    logs: pd.DataFrame, outdir: str
) -> Optional[str]:
    """
    Tamamlayıcı grafik: simetri-borç ilişkisinin eşleşme türüne göre dağılım +
    regresyon eğrisi (seaborn lmplot). Isı haritasını niceliksel olarak destekler.
    """
    df = logs.dropna(subset=["symmetry_index", "technical_debt"]).copy()
    df = df[df["technical_debt"] > 0]  # yalnızca borç biriken (anlamlı) gözlemler
    if df.empty:
        print(
            "[BİLGİ] Pozitif teknik borç gözlemi yok; "
            "korelasyon saçılım grafiği atlanıyor (henüz taviz birikmemiş)."
        )
        return None

    grid = sns.lmplot(
        data=df,
        x="symmetry_index",
        y="technical_debt",
        hue="pair_type",
        height=7,
        aspect=1.4,
        scatter_kws={"alpha": 0.4, "s": 35},
        line_kws={"linewidth": 2.5},
    )
    grid.set_axis_labels("Simetri İndeksi", "Teknik Borç (birikmiş gerilim)")
    grid.figure.suptitle(
        "Simetri İndeksi ↓  ⇒  Teknik Borç ↑  (eşleşme türüne göre)",
        y=1.03, fontsize=14,
    )

    path = os.path.join(outdir, "symmetry_vs_technical_debt_scatter.png")
    grid.savefig(path, dpi=150)
    plt.close(grid.figure)
    print(f"[OK] Simetri-Teknik Borç saçılım/regresyon grafiği kaydedildi: {path}")
    return path


# ===========================================================================
# 4) Konsol Özeti
# ===========================================================================
def print_summary(sims: pd.DataFrame, logs: pd.DataFrame) -> None:
    """Üretilen grafiklere eşlik eden kısa istatistiksel konsol özeti."""
    print("\n" + "=" * 64)
    print("VERİ ANALİZİ ÖZETİ")
    print("=" * 64)
    print(f"  Toplam simülasyon : {len(sims)}")
    if not sims.empty:
        print(f"  Durum dağılımı    : {sims['status'].value_counts().to_dict()}")
        for pt, grp in sims.groupby("pair_type"):
            stable = int((grp["status"] == "STABLE").sum())
            fatal = int((grp["status"] == "FATAL").sum())
            mean_dur = grp["duration"].mean()
            print(
                f"  {pt:22s}: n={len(grp):3d} | STABLE={stable} FATAL={fatal}"
                f" | ort.tur={mean_dur:.1f}"
            )
    if not logs.empty:
        valid = logs.dropna(subset=["symmetry_index", "technical_debt"])
        if not valid.empty:
            corr = valid["symmetry_index"].corr(valid["technical_debt"])
            print(f"\n  Tur kaydı (log)   : {len(logs)} satır")
            print(f"  Simetri-Borç korel: r = {corr:.3f}")
            print(f"  Maks teknik borç  : {logs['technical_debt'].max():.1f}")
    print("=" * 64 + "\n")


# ===========================================================================
# 5) Giriş Noktası
# ===========================================================================
def run_analysis(db_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    conn = connect(db_path)
    try:
        sims = load_simulations(conn)
        logs = load_round_logs(conn)
    finally:
        conn.close()

    if sims.empty:
        print("[UYARI] Simulations tablosu boş; analiz yapılacak veri yok.")
        return

    print_summary(sims, logs)
    plot_survival_curve(sims, outdir)
    plot_symmetry_debt_heatmap(logs, outdir)
    plot_symmetry_debt_correlation(logs, outdir)
    print(f"Tüm grafikler '{outdir}/' klasörüne yazıldı.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABM simülasyon verisini analiz eder ve akademik grafikler üretir."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH,
        help=f"SQLite veritabanı yolu (varsayılan: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help=f"Grafiklerin kaydedileceği klasör (varsayılan: {DEFAULT_OUTDIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_analysis(args.db, args.outdir)
