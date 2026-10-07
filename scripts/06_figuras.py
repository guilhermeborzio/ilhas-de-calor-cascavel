"""Gráficos do artigo (figuras 3a, 4, 5 e 6) e rasters usados nos mapas feitos no QGIS
(figuras 2, 3b e 7).

    python 06_figuras.py
"""
import json
import re
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
E2 = OUT / "etapa2_series"
E4 = OUT / "etapa4_censo"
FIG = OUT / "figuras"
RAS = FIG / "rasters_qgis"

LARG = 6.3          # polegadas ≈ 16 cm (largura útil A4 com margens ABNT)
DPI = 300
COR = {"Urbano consolidado (pré-1985)": "#7f0000", "Vegetação natural → urbano": "#e66101",
       "Agropecuária → urbano": "#d9a400", "Vegetação natural estável": "#1a7837"}
TRAJ = {1: "Urbano consolidado (pré-1985)", 2: "Vegetação natural → urbano",
        3: "Agropecuária → urbano", 5: "Vegetação natural estável"}
PARQUES_BACI = ["Parque Vitória / Hilário Zardo", "Parque Tarquínio Joslin dos Santos", "Ecopark Morumbi", "Ecopark Oeste"]
CURTO = {"Parque Ecológico Paulo Gorski": "Paulo Gorski", "Parque Municipal Danilo José Galafassi": "Zoológico",
         "Parque Vitória / Hilário Zardo": "Vitória", "Parque Tarquínio Joslin dos Santos": "Tarquínio",
         "Ecopark Morumbi": "Ecopark Morumbi", "Ecopark Oeste": "Ecopark Oeste",
         "Ecopark Santa Felicidade": "Ecopark Santa Felicidade", "Ecopark Norte Floresta": "Ecopark Floresta",
         "Parque Riviera": "Riviera", "Parque do Guarujá": "Guarujá"}

GRUPOS = {1: [3, 4, 5, 6, 49], 2: [9], 3: [11, 12, 32, 50], 4: [15],
          5: [18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48], 6: [21], 7: [24],
          8: [23, 25, 29, 30], 9: [31, 33]}
LUT = np.zeros(256, dtype=np.uint8)
for g, cls in GRUPOS.items():
    LUT[cls] = g


def estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


def virgula(*axs):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    for ax in axs:
        ax.yaxis.set_major_formatter(f)


def rotulo(ax, letra, x=-0.12):
    ax.text(x, 1.04, letra, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def ler_stack(nome, escala):
    import rasterio
    with rasterio.open(E2 / nome) as s:
        a = s.read().astype(np.float32)
        perfil = s.profile
    a[a == -32768] = np.nan
    return a / escala, perfil


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    estilo()
    FIG.mkdir(parents=True, exist_ok=True)
    RAS.mkdir(parents=True, exist_ok=True)
    print("Séries:", E2.name)

    # ---------- dados comuns ----------
    q = pd.read_csv(E2 / "qualidade_e_referencia_anual.csv")
    anos = q["verao"].values
    usado = q["usado"].values.astype(bool) & np.isfinite(q["lst_ref_rural"].values)
    lst, perfil = ler_stack("lst_veroes_x100.tif", 100)
    anom = lst - q["lst_ref_rural"].values[:, None, None].astype(np.float32)
    anom[~usado] = np.nan
    with rasterio.open(E2 / "trajetorias_mapbiomas.tif") as s:
        traj = s.read(1)
    with rasterio.open(E2 / "mapbiomas_anual.tif") as s:
        mb_ult = s.read(s.count)

    # ---------- rasters para o QGIS ----------
    k_rec = np.flatnonzero(usado)[-5:]
    anom_rec = np.nanmean(anom[k_rec], 0)
    p1 = perfil.copy()
    p1.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(RAS / f"anomalia_recente_{anos[k_rec][0]}_{anos[k_rec][-1]}.tif", "w", **p1) as d:
        d.write(anom_rec.astype(np.float32), 1)
    p2 = perfil.copy()
    p2.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(RAS / "mapbiomas_ultimo_ano_agrupado.tif", "w", **p2) as d:
        d.write(LUT[mb_ult], 1)

    # ---------- Fig. 3a — séries por trajetória ----------
    fig, ax = plt.subplots(figsize=(LARG, 2.8))
    for era, (a, b, c) in {"Landsat 5/7": (1984.5, 2013.5, "#f0f0f0"), "Landsat 8/9": (2013.5, 2026.5, "#e0e0e0")}.items():
        ax.axvspan(a, b, color=c, zorder=0)
        ax.text((a + b) / 2, 0.98, era, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#555")
    for cod, nome in TRAJ.items():
        m = traj == cod
        y = np.array([np.nanmean(anom[t][m]) if usado[t] else np.nan for t in range(len(anos))])
        ok = np.isfinite(y)
        ax.plot(anos[ok], y[ok], "-o", ms=2.5, lw=1.2, color=COR[nome], label=nome)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(1984.5, 2026.5)
    ax.set_xlabel("Verão (dez–mar)")
    ax.set_ylabel("Anomalia térmica de superfície (°C)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    virgula(ax)
    fig.savefig(FIG / "fig3a_series_trajetorias.png", dpi=DPI)
    plt.close(fig)

    # ---------- Fig. 4 — estudo de evento ----------
    ev = pd.read_csv(E2 / "estudo_evento_urbanizacao.csv")
    fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.6))
    for ax, var, yl, letra in [(axs[0], "anomalia", "Δ anomalia vs. controle (°C)", "a"),
                               (axs[1], "ndvi", "Δ NDVI vs. controle", "b")]:
        ax.axvspan(3, 10, color="#f2f2f2", zorder=0)
        for tr, d in ev[ev.variavel == var].groupby("trajetoria"):
            d = d.sort_values("k")
            c = COR.get(tr, "grey")
            ax.fill_between(d.k, d.media - d.ic95, d.media + d.ic95, color=c, alpha=0.2, lw=0)
            ax.plot(d.k, d.media, "-o", ms=2.5, lw=1.2, color=c, label=tr)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Anos desde a conversão para área urbanizada")
        ax.set_ylabel(yl)
        rotulo(ax, letra)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for ax in axs:
        ax.text(6.5, 1.0, "janela do efeito", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color="#555")
    virgula(*axs)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig4_estudo_evento.png", dpi=DPI)
    plt.close(fig)

    # ---------- Fig. 5 — parques ----------
    import geopandas as gpd
    pqg = gpd.read_file(OUT / "parques_cascavel.geojson")
    pqg["base"] = pqg["nome"].str.replace(r" \(.*\)$", "", regex=True)
    ano_pq = dict(zip(pqg["base"], pd.to_numeric(pqg["ano"], errors="coerce")))
    ser = pd.read_csv(E2 / "parques_series.csv")
    per = pd.read_csv(E2 / "parques_perfil_resfriamento.csv")
    fig = plt.figure(figsize=(LARG, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.28)
    for i, p in enumerate(PARQUES_BACI):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        d = ser[ser.parque == p].pivot(index="verao", columns="zona", values="anomalia")
        if d.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{CURTO.get(p, p)}: sem dados", ha="center", transform=ax.transAxes)
            continue
        D = (d["parque"] - d["controle 500-1000 m"]).dropna()
        ac = ano_pq.get(p)
        ax.plot(D.index, D.values, "-o", ms=2, lw=1, color="#2166ac")
        if ac is not None and np.isfinite(ac):
            ax.axvline(ac, color="#1a7837", ls="--", lw=1)
            antes, depois = D[D.index < ac], D[D.index > ac + 1]
            if len(antes):
                ax.hlines(antes.mean(), antes.index.min(), antes.index.max(), color="k", lw=1.6)
            if len(depois):
                ax.hlines(depois.mean(), depois.index.min(), depois.index.max(), color="#b2182b", lw=1.6)
                ef = depois.mean() - antes.mean()
                ax.set_title(f"{CURTO.get(p, p)} ({int(ac)})  ·  efeito {ef:+.1f} °C".replace(".", ",").replace("-", "\u2212"),
                             color="#b2182b" if ef > 0 else "#2166ac", fontweight="bold")
        ax.axhline(0, color="k", lw=0.4)
        if not ax.get_title():
            ax.set_title(f"{CURTO.get(p, p)} ({int(ac) if ac is not None and np.isfinite(ac) else '?'})")
        virgula(ax)
        ax.set_ylabel("parque − controle (°C)")
        if i == 0:
            rotulo(ax, "a")
    ax = fig.add_subplot(gs[2, :])
    for p, d in per.sort_values("dist_m").groupby("parque"):
        if CURTO.get(p) in ("Guarujá", "Riviera", "Ecopark Floresta", None):
            continue
        ax.plot(d.dist_m, d.anomalia, "-o", ms=2, lw=1.1, label=CURTO.get(p, p))
    ax.set_xlabel("Distância da borda do parque (m)   [0 = interior]")
    ax.set_ylabel("Anomalia 2022–2026 (°C)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    virgula(ax)
    rotulo(ax, "b", x=-0.06)
    fig.savefig(FIG / "fig5_parques.png", dpi=DPI)
    plt.close(fig)

    # ---------- Fig. 6 — exposição e justiça térmica ----------
    exp = pd.read_csv(E4 / "exposicao_populacao_por_classe.csv")
    jus = pd.read_csv(E4 / "justica_termica_quintis_renda.csv")
    jus.columns = ["quintil"] + list(jus.columns[1:])
    fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    x = np.arange(len(exp))
    for i, (col, lab, c) in enumerate([("pop_%", "População total", "#4d4d4d"),
                                       ("pop60_%", "60 anos ou mais", "#b2182b"),
                                       ("pop09_%", "0 a 9 anos", "#ef8a62")]):
        ax.bar(x + (i - 1) * 0.27, exp[col], 0.27, label=lab, color=c)
    ax.set_xticks(x, exp.classe_anomalia, rotation=0)
    ax.set_xlabel("Anomalia térmica de superfície")
    ax.set_ylabel("% do grupo")
    ax.legend(frameon=False)
    rotulo(ax, "a")
    ax = axs[1]
    xq = np.arange(len(jus))
    ax.bar(xq, jus["dist_parque_pond_m"] / 1000, color="#7fbf7b", label="Distância média ao parque (km)")
    ax.set_ylabel("Distância média ao parque (km)")
    ax.set_xticks(xq, ["Q1\nmenor", "Q2", "Q3", "Q4", "Q5\nmaior"])
    ax.set_xlabel("Quintil de renda do responsável")
    ax2 = ax.twinx()
    ax2.plot(xq, jus["anom_pond"], "-o", color="#b2182b", ms=3, lw=1.2, label="Anomalia (°C)")
    ax2.set_ylabel("Anomalia média (°C)", color="#b2182b")
    ax2.tick_params(axis="y", colors="#b2182b")
    ax2.spines["right"].set_visible(True)
    rotulo(ax, "b")
    virgula(axs[0], ax, ax2)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_exposicao_justica.png", dpi=DPI)
    plt.close(fig)

    json.dump({"series": E2.name, "veroes_recentes": anos[k_rec].tolist(),
               "figuras": sorted(p.name for p in FIG.glob("*.png")),
               "rasters_qgis": sorted(p.name for p in RAS.glob("*.tif"))},
              open(FIG / "figuras_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("ok:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
