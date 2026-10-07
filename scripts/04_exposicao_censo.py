"""Exposição da população ao calor, acesso a parques e praças e Índice de Prioridade
para Arborização (IPA) por setor censitário do Censo 2022.

A população de cada setor é distribuída aos pixels de área urbanizada do MapBiomas
(mapeamento dasimétrico). O IPA é a média dos postos percentuais de exposição
(anomalia), sensibilidade (0 a 9 e 60 anos ou mais), capacidade adaptativa (renda)
e déficit verde (NDVI e distância ao parque).

    python 04_exposicao_censo.py
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
E2 = OUT / "etapa2_series"
E4 = OUT / "etapa4_censo"
CENSO = OUT / "censo2022" / "cascavel_setores_censo2022.gpkg"
PARQUES = OUT / "parques_cascavel.geojson"

XMIN, YMIN = -53.548843580949914, -25.011253466569364
XMAX, YMAX = -53.375828057228496, -24.903994621645495
RES = 0.000269494585236
DX_M = RES * 111320 * np.cos(np.radians(24.96))
DY_M = RES * 110574

N_VEROES_RECENTES = 5
PESO_NAO_URBANO = 0.05
POP_MIN_SETOR = 150
ANO_CORTE_PARQUE = 2025   # parques criados depois do último verão entram só no cenário de 2026
PRACAS = OUT / "geocascavel" / "pracas.geojson"
CLASSES_ANOM = [-np.inf, 4, 6, 8, 10, 12, np.inf]
ROTULOS_ANOM = ["< 4 °C", "4–6 °C", "6–8 °C", "8–10 °C", "10–12 °C", "> 12 °C"]


def sem_nan(o):
    """Converte NaN/inf (Python ou numpy) em None para gerar JSON válido (null)."""
    if isinstance(o, dict):
        return {k: sem_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sem_nan(v) for v in o]
    if isinstance(o, np.ndarray):
        return sem_nan(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def ler(nome, escala=None):
    import rasterio
    with rasterio.open(E2 / nome) as s:
        a = s.read().astype(np.float32)
        tr = s.transform
    if escala:
        a[a == -32768] = np.nan
        a /= escala
    return a, tr


def posto(s):
    return s.rank(pct=True)


def main():
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.features import rasterize
    from scipy import stats
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import box

    E4.mkdir(parents=True, exist_ok=True)
    print("Séries usadas:", E2.name)

    # ---------- rasters ----------
    lst, tr = ler("lst_veroes_x100.tif", 100)
    nd, _ = ler("ndvi_veroes_x10000.tif", 10000)
    mb, _ = ler("mapbiomas_anual.tif")
    q = pd.read_csv(E2 / "qualidade_e_referencia_anual.csv")
    ok = q["usado"].values.astype(bool) & np.isfinite(q["lst_ref_rural"].values)
    k_rec = np.flatnonzero(ok)[-N_VEROES_RECENTES:]
    k_ini = np.flatnonzero(ok)[:10]
    anom = lst - q["lst_ref_rural"].values[:, None, None].astype(np.float32)
    anom_rec = np.nanmean(anom[k_rec], 0)
    anom_ini = np.nanmean(anom[k_ini], 0)
    ndvi_rec = np.nanmean(nd[k_rec], 0)
    urbano = mb[-1] == 24
    H, W = anom_rec.shape
    veroes_rec = q["verao"].values[k_rec].tolist()

    # ---------- setores ----------
    st = gpd.read_file(CENSO, layer="setores_cascavel").to_crs(4674)
    aoi = box(XMIN, YMIN, XMAX, YMAX)
    st = st[st.intersects(aoi)].copy().reset_index(drop=True)
    st_m = st.to_crs(31982)
    aoi_m = gpd.GeoSeries([aoi], crs=4674).to_crs(31982).iloc[0]
    st["frac_dentro"] = (st_m.intersection(aoi_m).area / st_m.area).values
    st["sid"] = np.arange(1, len(st) + 1)
    sid = rasterize(zip(st.geometry, st.sid), out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    faltam = set(st.sid) - set(np.unique(sid))
    if faltam:  # setores menores que um pixel: rasteriza tocando bordas
        sub = st[st.sid.isin(faltam)]
        s2 = rasterize(zip(sub.geometry, sub.sid), out_shape=(H, W), transform=tr, fill=0, dtype="int32", all_touched=True)
        sid = np.where((sid == 0) & (s2 > 0), s2, sid)

    # ---------- dasimetria ----------
    peso = np.where(urbano, 1.0, PESO_NAO_URBANO) * (sid > 0)
    soma_peso = np.bincount(sid.ravel(), weights=peso.ravel(), minlength=len(st) + 1)
    popcols = {"pop_total": "pop", "pop_60mais": "pop60", "pop_0a9": "pop09"}
    px = {}
    for col, nome in popcols.items():
        v = np.zeros(len(st) + 1)
        v[st.sid] = (st[col].fillna(0) * st["frac_dentro"]).values
        with np.errstate(invalid="ignore", divide="ignore"):
            px[nome] = np.where(sid > 0, v[sid] * peso / soma_peso[sid], 0.0)
        px[nome] = np.nan_to_num(px[nome])

    # ---------- parques e distância ----------
    pq = gpd.read_file(PARQUES)
    pq["geometry"] = pq.geometry.make_valid()
    if "na_area_estudo" in pq.columns:
        pq = pq[pq["na_area_estudo"].astype(bool)].copy()
    ano_pq = pd.to_numeric(pq.get("ano"), errors="coerce")
    pq_exist = pq[ano_pq.isna() | (ano_pq <= ANO_CORTE_PARQUE)]   # ano incerto: assume existente
    def dist_de(geoms):
        m = rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=tr, fill=0, dtype="uint8") > 0
        return distance_transform_edt(~m, sampling=(DY_M, DX_M))
    dist_pq = dist_de(pq_exist.to_crs(4674).geometry)            # parques existentes no período
    dist_pq_atual = dist_de(pq.to_crs(4674).geometry)            # cenário 2026 (inclui Ecopark Floresta)
    pr = gpd.read_file(PRACAS).to_crs(4674) if PRACAS.exists() else None
    dist_pr = dist_de(pr.geometry.buffer(0.00005)) if pr is not None else np.full((H, W), np.inf)
    dist_verde = np.minimum(dist_pq, dist_pr)

    # ---------- 3. exposição ----------
    valido = np.isfinite(anom_rec)
    cls = np.digitize(anom_rec, CLASSES_ANOM[1:-1])
    exp = []
    for i, r in enumerate(ROTULOS_ANOM):
        m = valido & (cls == i)
        exp.append({"classe_anomalia": r, **{k: float(px[k][m].sum()) for k in px}})
    exp = pd.DataFrame(exp)
    for k in px:
        exp[f"{k}_%"] = 100 * exp[k] / exp[k].sum()
    exp.to_csv(E4 / "exposicao_populacao_por_classe.csv", index=False)

    # ---------- 4. acesso a parques ----------
    acesso = {}
    for rot, dd in [("parques (período)", dist_pq), ("parques + Ecopark Floresta (2026)", dist_pq_atual),
                    ("praças", dist_pr), ("parque ou praça", dist_verde)]:
        for d in (300, 500):
            m = dd <= d
            acesso[f"{rot} <= {d} m"] = {k: {"pessoas": float(px[k][m].sum()), "pct": float(100 * px[k][m].sum() / px[k].sum())} for k in px}
    pd.DataFrame([{"cenario": k, **{f"{g}_{c}": v[g][c] for g in v for c in v[g]}} for k, v in acesso.items()]).to_csv(
        E4 / "acesso_areas_verdes.csv", index=False)

    # ---------- indicadores por setor ----------
    def media_setor(arr, w=None):
        m = (sid > 0) & np.isfinite(arr)
        ww = np.ones_like(arr) if w is None else w
        num = np.bincount(sid[m], weights=(arr * ww)[m], minlength=len(st) + 1)
        den = np.bincount(sid[m], weights=ww[m], minlength=len(st) + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num / den)[st.sid]

    st["anom_recente"] = media_setor(anom_rec, px["pop"] + 1e-9)
    st["anom_inicio"] = media_setor(anom_ini, px["pop"] + 1e-9)
    st["delta_anom"] = st["anom_recente"] - st["anom_inicio"]
    st["ndvi_recente"] = media_setor(ndvi_rec, px["pop"] + 1e-9)
    st["dist_parque_m"] = media_setor(dist_pq.astype(np.float32), px["pop"] + 1e-9)
    st["dist_praca_m"] = media_setor(np.minimum(dist_pr, 1e5).astype(np.float32), px["pop"] + 1e-9)
    st["dist_parque_2026_m"] = media_setor(dist_pq_atual.astype(np.float32), px["pop"] + 1e-9)
    st["pct_urbano"] = 100 * media_setor(urbano.astype(np.float32))
    st["pct_60mais"] = 100 * st["pop_60mais"] / st["pop_total"]
    st["pct_0a9"] = 100 * st["pop_0a9"] / st["pop_total"]

    alvo = (st["pop_total"] >= POP_MIN_SETOR) & (st["frac_dentro"] > 0.5) & st["anom_recente"].notna()
    a = st[alvo].copy()
    a["E"] = posto(a["anom_recente"])
    a["S"] = posto((a["pct_60mais"].fillna(a["pct_60mais"].median()) + a["pct_0a9"].fillna(a["pct_0a9"].median())))
    a["CA"] = 1 - posto(a["renda_media_resp"].fillna(a["renda_media_resp"].median()))
    a["DV"] = (posto(1 - a["ndvi_recente"]) + posto(a["dist_parque_m"])) / 2
    a["IVT"] = a[["E", "S", "CA"]].mean(1)
    a["IPA"] = a[["E", "S", "CA", "DV"]].mean(1)
    a["IPA_quintil"] = pd.qcut(a["IPA"], 5, labels=["muito baixa", "baixa", "média", "alta", "muito alta"])
    st = st.merge(a[["CD_SETOR", "E", "S", "CA", "DV", "IVT", "IPA", "IPA_quintil"]], on="CD_SETOR", how="left")
    st["IPA_quintil"] = st["IPA_quintil"].astype(str).replace("nan", "")
    st.drop(columns="sid").to_file(E4 / "setores_indicadores_calor.gpkg", layer="setores", driver="GPKG")
    cols = ["CD_SETOR", "NM_BAIRRO", "pop_total", "pct_60mais", "pct_0a9", "renda_media_resp", "anom_recente",
            "delta_anom", "ndvi_recente", "dist_parque_m", "dist_parque_2026_m", "dist_praca_m", "IVT", "IPA", "IPA_quintil"]
    st[cols].to_csv(E4 / "setores_indicadores_calor.csv", index=False)
    top = a.sort_values("IPA", ascending=False).head(30)[cols[:-1]]
    top.to_csv(E4 / "top30_setores_prioridade_arborizacao.csv", index=False)
    bairros = (a.groupby("NM_BAIRRO").apply(lambda d: pd.Series({
        "setores": len(d), "pop": d.pop_total.sum(),
        "anom_media_pond": np.average(d.anom_recente, weights=d.pop_total),
        "IPA_medio": np.average(d.IPA, weights=d.pop_total),
        "setores_muito_alta": int((d.IPA_quintil == "muito alta").sum())}), include_groups=False)
        .sort_values("IPA_medio", ascending=False))
    bairros.to_csv(E4 / "bairros_prioridade.csv")

    # ---------- justiça térmica ----------
    rq = pd.qcut(a["renda_media_resp"], 5, labels=["Q1 (menor renda)", "Q2", "Q3", "Q4", "Q5 (maior renda)"])
    just = a.groupby(rq, observed=True).apply(lambda d: pd.Series({
        "pop": d.pop_total.sum(), "renda_media": d.renda_media_resp.mean(),
        "anom_pond": np.average(d.anom_recente, weights=d.pop_total),
        "ndvi_pond": np.average(d.ndvi_recente, weights=d.pop_total),
        "dist_parque_pond_m": np.average(d.dist_parque_m, weights=d.pop_total)}), include_groups=False)
    just.to_csv(E4 / "justica_termica_quintis_renda.csv")
    cor = {}
    for v in ["renda_media_resp", "pct_60mais", "pct_0a9", "ndvi_recente", "dist_parque_m"]:
        r, p = stats.spearmanr(a["anom_recente"], a[v], nan_policy="omit")
        cor[f"anom_recente × {v}"] = {"rho": float(r), "p": float(p)}
    r, p = stats.spearmanr(a["renda_media_resp"], a["ndvi_recente"], nan_policy="omit")
    cor["renda × ndvi_recente"] = {"rho": float(r), "p": float(p)}

    resumo = {"series": E2.name, "veroes_recentes": veroes_rec,
              "setores_na_area": int(len(st)), "setores_no_indice": int(len(a)),
              "pop_na_area_dasimetrica": float(px["pop"].sum()),
              "pop_em_anomalia_>=8C_pct": float(100 * px["pop"][valido & (anom_rec >= 8)].sum() / px["pop"].sum()),
              "pop60_em_anomalia_>=8C_pct": float(100 * px["pop60"][valido & (anom_rec >= 8)].sum() / px["pop60"].sum()),
              "acesso_parques": acesso, "correlacoes_spearman_setor": cor,
              "justica_termica": just.round(3).to_dict("index")}
    json.dump(sem_nan(resumo), open(E4 / "resumo_etapa4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, allow_nan=False)

    # ---------- figuras ----------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    st.plot(column="anom_recente", cmap="RdYlBu_r", legend=True, ax=axs[0], vmin=2, vmax=13,
            legend_kwds={"label": "Anomalia térmica (°C)", "shrink": 0.7}, missing_kwds={"color": "lightgrey"})
    axs[0].set_title(f"Anomalia média por setor — verões {veroes_rec[0]}–{veroes_rec[-1]}")
    cores = {"muito baixa": "#1a9641", "baixa": "#a6d96a", "média": "#ffffbf", "alta": "#fdae61", "muito alta": "#d7191c"}
    for k, c in cores.items():
        s = st[st.IPA_quintil == k]
        if len(s):
            s.plot(ax=axs[1], color=c, label=k, edgecolor="none")
    st[st.IPA_quintil == ""].plot(ax=axs[1], color="lightgrey", edgecolor="none")
    pq.to_crs(4674).boundary.plot(ax=axs[1], color="darkgreen", lw=1.2)
    axs[1].legend(title="Prioridade (IPA)", fontsize=8, loc="lower left")
    axs[1].set_title("Índice de Prioridade para Arborização")
    for ax in axs:
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(E4 / "fig6_anomalia_e_prioridade_setores.png", dpi=220)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(exp))
    for i, (k, rot) in enumerate([("pop_%", "População total"), ("pop60_%", "60 anos ou mais"), ("pop09_%", "0 a 9 anos")]):
        ax.bar(x + (i - 1) * 0.27, exp[k], 0.27, label=rot)
    ax.set_xticks(x, exp.classe_anomalia)
    ax.set_ylabel("% da população")
    ax.set_xlabel("Anomalia térmica de superfície (°C acima da referência rural)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(E4 / "fig7_exposicao_populacao.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(a.renda_media_resp, a.anom_recente, c=a.ndvi_recente, cmap="Greens", s=np.sqrt(a.pop_total) * 1.5, edgecolor="k", lw=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Renda média do responsável (R$, escala log)")
    ax.set_ylabel("Anomalia térmica (°C)")
    plt.colorbar(sc, label="NDVI")
    ax.set_title(f"ρ = {cor['anom_recente × renda_media_resp']['rho']:.2f}")
    fig.tight_layout()
    fig.savefig(E4 / "fig8_renda_x_anomalia.png", dpi=200)

    print(json.dumps(sem_nan(resumo), ensure_ascii=False, indent=2, allow_nan=False))
    print(exp.round(1).to_string(index=False))
    print(bairros.head(10).round(2).to_string())


if __name__ == "__main__":
    main()
