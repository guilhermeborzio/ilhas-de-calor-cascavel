"""Cobertura construída (edificações de 1995, 2007 e cadastro atual) e arborização viária
em relação à anomalia térmica, numa grade de 300 m.

As edificações são rasterizadas a 2 m para não contar sobreposições em dobro. Os intervalos
de confiança vêm de bootstrap em blocos de 1,5 km.

    python 05_edificacoes_arvores.py
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
GEO = OUT / "geocascavel"
E2 = OUT / "etapa2_series"
E5 = OUT / "etapa5_edificacoes_arvores"

XMIN, YMIN = -53.548843580949914, -25.011253466569364
XMAX, YMAX = -53.375828057228496, -24.903994621645495
RES_FINO = 2.0        # m — rasterização das edificações
CEL = 300.0           # m — célula de análise
BLOCO = 5             # células por lado no bootstrap em blocos (5 × 300 m = 1,5 km)
N_BOOT = 1000
JANELAS = {"1995": (1993, 1997), "2007": (2005, 2009), "recente": (2022, 2026)}
MIN_PIX_CEL = 40      # nº mínimo de pixels Landsat válidos (30 m) por célula
URB_CONSOLIDADO = 0.8
FRAC_MIN_COB_ATUAL = 0.8   # BCF atual mínimo, relativo a 2007, para usar a célula em 2007→recente


# ---------------------------------------------------------------------
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


def grade_utm():
    from pyproj import Transformer
    t = Transformer.from_crs(4674, 31982, always_xy=True)
    xs, ys = t.transform([XMIN, XMAX, XMIN, XMAX], [YMIN, YMIN, YMAX, YMAX])
    x0, y0 = np.floor(min(xs) / CEL) * CEL, np.floor(min(ys) / CEL) * CEL
    x1, y1 = np.ceil(max(xs) / CEL) * CEL, np.ceil(max(ys) / CEL) * CEL
    return x0, y0, x1, y1


def rasterizar(gdf, x0, y0, x1, y1, valor=None):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    W, H = int(round((x1 - x0) / RES_FINO)), int(round((y1 - y0) / RES_FINO))
    tr = from_origin(x0, y1, RES_FINO, RES_FINO)
    if len(gdf) == 0:
        return np.zeros((H, W), np.uint8)
    shapes = zip(gdf.geometry, gdf[valor] if valor else np.ones(len(gdf), np.uint8))
    return rasterize(shapes, out_shape=(H, W), transform=tr, fill=0, dtype="uint8", merge_alg=__import__("rasterio").enums.MergeAlg.replace)


def agregar(fino, f=int(CEL / RES_FINO), func="mean"):
    H, W = fino.shape
    a = fino[:H - H % f, :W - W % f].reshape(H // f, f, W // f, f)
    return a.mean(axis=(1, 3)) if func == "mean" else a.max(axis=(1, 3))


def casco(gdf):
    # Casco convexo do conjunto = casco convexo da união, sem depender de união topológica
    # (falha com geometrias inválidas do cadastro: "side location conflict").
    from shapely.geometry import GeometryCollection
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    return GeometryCollection(geoms).convex_hull


def boot_blocos(df, func, blocos, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocos)
    idx = {b: np.flatnonzero(blocos == b) for b in ub}
    est = []
    for _ in range(n):
        sel = np.concatenate([idx[b] for b in rng.choice(ub, ub.size, replace=True)])
        try:
            est.append(func(df.iloc[sel]))
        except Exception:
            pass
    est = np.array(est, dtype=float)
    return np.nanpercentile(est, [2.5, 97.5], axis=0)


def ols_pad(d, y, xs):
    """Coeficientes padronizados (z-scores) por OLS."""
    Z = d[[y] + xs].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    X = np.column_stack([np.ones(len(Z))] + [Z[x].values for x in xs])
    b, *_ = np.linalg.lstsq(X, Z[y].values, rcond=None)
    r2 = 1 - ((Z[y].values - X @ b) ** 2).sum() / (Z[y].values ** 2).sum()
    return np.r_[b[1:], r2]


def sen(x, y):
    from scipy import stats
    return stats.theilslopes(y, x)[0]


# ---------------------------------------------------------------------
def main():
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from pyproj import Transformer
    from scipy import stats

    E5.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = grade_utm()
    nx, ny = int((x1 - x0) / CEL), int((y1 - y0) / CEL)
    print(f"Grade: {nx} × {ny} células de {CEL:.0f} m | séries: {E2.name}")
    resumo = {"series": E2.name, "celula_m": CEL}

    # ---------- 1. Edificações ----------
    def ler(nome, **kw):
        g = gpd.read_file(GEO / nome, **kw)
        return g.to_crs(31982) if g.crs and g.crs.to_epsg() != 31982 else g

    e95 = ler("edificacoes_1995.geojson")
    e95 = e95[e95["Layer"].astype(str).str.startswith("EDIF")]
    e07 = ler("edificacoes_2007.geojson")
    camada = e07["Layer"].astype(str)
    e07 = e07[camada.str.match(r"^Pavimento_\d+$")].copy()
    e07["pav"] = e07["Layer"].str.extract(r"(\d+)$")[0].astype(int) + 1
    eat = ler("edificacoes_atual_geometria.geojson")
    if "nivel" in eat.columns and (eat["nivel"] == 0).any():
        eat = eat[eat["nivel"].fillna(0) == 0]
    resumo["n_edificacoes"] = {"1995": int(len(e95)), "2007": int(len(e07)), "atual": int(len(eat))}

    bcf = {}
    for ep, g in [("1995", e95), ("2007", e07), ("atual", eat)]:
        fino = rasterizar(g, x0, y0, x1, y1)
        bcf[ep] = agregar(fino.astype(np.float32))[::-1][:ny, :nx]  # linha 0 = sul
        resumo.setdefault("area_construida_km2", {})[ep] = float(fino.sum() * RES_FINO ** 2 / 1e6)
        del fino
    pav = rasterizar(e07.sort_values("pav"), x0, y0, x1, y1, valor="pav")
    far07 = agregar(pav.astype(np.float32))[::-1][:ny, :nx]          # média de pavimentos × cobertura ≈ FAR
    del pav

    cob = {ep: casco(g) for ep, g in [("1995", e95), ("2007", e07), ("atual", eat)]}

    # ---------- 2. Árvores ----------
    arv = ler("arvores_inventario.geojson")
    arv = arv[arv.geometry.notna()]
    ix = ((arv.geometry.x - x0) // CEL).astype(int).values
    iy = ((arv.geometry.y - y0) // CEL).astype(int).values
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    def contar(m):
        c = np.zeros((ny, nx))
        np.add.at(c, (iy[ok & m], ix[ok & m]), 1)
        return c
    sit = arv["situacao"].astype(str).values
    arv_plant = contar(sit == "PLANTADA")
    arv_cort = contar(sit == "CORTADA")
    area_ha = CEL ** 2 / 1e4
    resumo["arvores"] = {"plantadas": int((sit == "PLANTADA").sum()), "cortadas": int((sit == "CORTADA").sum())}

    # ---------- 3. Térmico e MapBiomas por célula ----------
    with rasterio.open(E2 / "lst_veroes_x100.tif") as s:
        lst = s.read().astype(np.float32)
        trl = s.transform
    lst[lst == -32768] = np.nan
    lst /= 100
    with rasterio.open(E2 / "ndvi_veroes_x10000.tif") as s:
        nd = s.read().astype(np.float32)
    nd[nd == -32768] = np.nan
    nd /= 10000
    with rasterio.open(E2 / "mapbiomas_anual.tif") as s:
        mb = s.read()
    meta = json.load(open(E2 / "metadados.json", encoding="utf-8"))
    mb_anos = meta["mapbiomas_anos"]
    q = pd.read_csv(E2 / "qualidade_e_referencia_anual.csv")
    anos = q["verao"].values
    anom = lst - q["lst_ref_rural"].values[:, None, None].astype(np.float32)
    anom[~q["usado"].values.astype(bool)] = np.nan
    nd[~q["usado"].values.astype(bool)] = np.nan

    Hp, Wp = lst.shape[1:]
    cc, rr = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
    lon, lat = trl * (cc, rr)
    ux, uy = Transformer.from_crs(4674, 31982, always_xy=True).transform(lon, lat)
    cx = ((ux - x0) // CEL).astype(int)
    cy = ((uy - y0) // CEL).astype(int)
    dentro = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    cid = np.where(dentro, cy * nx + cx, -1)

    def por_celula(arr2d):
        m = (cid >= 0) & np.isfinite(arr2d)
        s_ = np.bincount(cid[m], weights=arr2d[m], minlength=nx * ny)
        n_ = np.bincount(cid[m], minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = s_ / n_
        v[n_ < MIN_PIX_CEL] = np.nan
        return v.reshape(ny, nx)

    an_cel, nd_cel = {}, {}
    for k, (a, b) in JANELAS.items():
        sel = (anos >= a) & (anos <= b)
        an_cel[k] = por_celula(np.nanmean(anom[sel], 0))
        nd_cel[k] = por_celula(np.nanmean(nd[sel], 0))

    def urb_frac(ano):
        k = mb_anos.index(min(max(ano, mb_anos[0]), mb_anos[-1]))
        return por_celula((mb[k] == 24).astype(np.float32))
    urb = {"1985": urb_frac(1985), "1995": urb_frac(1995), "2007": urb_frac(2007), "atual": urb_frac(mb_anos[-1])}

    # ---------- 4. Tabela da grade ----------
    from shapely.geometry import box
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    geoms = [box(x0 + j * CEL, y0 + i * CEL, x0 + (j + 1) * CEL, y0 + (i + 1) * CEL) for i, j in zip(ii.ravel(), jj.ravel())]
    g = gpd.GeoDataFrame({
        "bcf_1995": bcf["1995"].ravel(), "bcf_2007": bcf["2007"].ravel(), "bcf_atual": bcf["atual"].ravel(),
        "far_2007": far07.ravel(),
        "arv_ha": (arv_plant / area_ha).ravel(), "arv_cortadas": arv_cort.ravel(),
        "frac_cortadas": np.where(arv_plant + arv_cort > 0, arv_cort / np.maximum(arv_plant + arv_cort, 1), np.nan).ravel(),
        **{f"anom_{k}": v.ravel() for k, v in an_cel.items()},
        **{f"ndvi_{k}": v.ravel() for k, v in nd_cel.items()},
        **{f"urb_{k}": v.ravel() for k, v in urb.items()},
        "bloco": ((ii // BLOCO) * 1000 + jj // BLOCO).ravel(),
    }, geometry=geoms, crs=31982)
    for ep, poly in cob.items():
        g[f"cob_{ep}"] = g.geometry.centroid.within(poly)
    g["d_bcf_95_07"] = g.bcf_2007 - g.bcf_1995
    g["d_bcf_07_at"] = g.bcf_atual - g.bcf_2007
    g["d_anom_95_07"] = g.anom_2007 - g.anom_1995
    g["d_anom_07_rec"] = g.anom_recente - g.anom_2007
    g["d_ndvi_95_07"] = g.ndvi_2007 - g.ndvi_1995
    g["d_ndvi_07_rec"] = g.ndvi_recente - g.ndvi_2007
    g.to_file(E5 / "grade_300m_edificacoes_arvores.gpkg", layer="grade", driver="GPKG")

    res = []

    def registrar(nome, d, y, x, extra=None):
        d = d[[y, x, "bloco"] + (extra or [])].dropna()
        if len(d) < 30:
            res.append({"analise": nome, "n_celulas": len(d)})
            return
        rho, p = stats.spearmanr(d[x], d[y])
        sl = sen(d[x].values, d[y].values)
        lo, hi = boot_blocos(d, lambda s: sen(s[x].values, s[y].values), d["bloco"].values)
        r = {"analise": nome, "n_celulas": len(d), "spearman_rho": rho, "p": p,
             "sen_C_por_10pp" if x.startswith(("bcf", "d_bcf")) else "sen_C_por_unidade": sl * (0.1 if x.startswith(("bcf", "d_bcf")) else 1),
             "ic95_boot_blocos": f"[{lo * (0.1 if x.startswith(('bcf', 'd_bcf')) else 1):.3f}; {hi * (0.1 if x.startswith(('bcf', 'd_bcf')) else 1):.3f}]"}
        if extra:
            b = ols_pad(d, y, [x] + extra)
            r.update({f"beta_pad_{v}": bb for v, bb in zip([x] + extra, b[:-1])})
            r["r2"] = b[-1]
        res.append(r)

    urbano_rec = g.urb_atual >= 0.5
    # (a) transversal recente
    d = g[urbano_rec & g.cob_atual]
    registrar("recente: anomalia × BCF atual", d, "anom_recente", "bcf_atual", ["arv_ha", "ndvi_recente"])
    registrar("recente: anomalia × árvores/ha (controle BCF, NDVI)", d, "anom_recente", "arv_ha", ["bcf_atual", "ndvi_recente"])
    # (b) primeira diferença
    m1 = g.cob_1995 & g.cob_2007 & (g.urb_2007 >= 0.5)
    registrar("1995→2007: Δanomalia × ΔBCF (urbano)", g[m1], "d_anom_95_07", "d_bcf_95_07", ["d_ndvi_95_07"])
    m1c = m1 & (g.urb_1995 >= URB_CONSOLIDADO) & (g.urb_2007 >= URB_CONSOLIDADO)
    registrar("1995→2007: Δanomalia × ΔBCF (urbano consolidado)", g[m1c], "d_anom_95_07", "d_bcf_95_07", ["d_ndvi_95_07"])
    # Lacuna do cadastro atual: a área construída "atual" é menor que a de 2007 apesar de ~2,5× mais
    # edificações, o que indica omissões/diferença de digitalização. Para não confundir lacuna com
    # demolição, a análise 2007→recente usa só células onde BCF atual ≥ FRAC_MIN_COB_ATUAL × BCF 2007.
    m2_base = g.cob_2007 & g.cob_atual & urbano_rec
    ok_cad = g.bcf_atual >= FRAC_MIN_COB_ATUAL * g.bcf_2007
    m2 = m2_base & ok_cad
    resumo["lacuna_cadastro_atual"] = {
        "area_construida_2007_km2": resumo["area_construida_km2"]["2007"],
        "area_construida_atual_km2": resumo["area_construida_km2"]["atual"],
        "razao_atual_2007": resumo["area_construida_km2"]["atual"] / resumo["area_construida_km2"]["2007"],
        "criterio_2007_recente": f"bcf_atual >= {FRAC_MIN_COB_ATUAL} * bcf_2007",
        "celulas_urbanas_cobertas": int(m2_base.sum()),
        "celulas_excluidas": int((m2_base & ~ok_cad).sum()),
        "celulas_mantidas": int(m2.sum()),
        "nota": "Análises 2007→recente restritas às células com cadastro atual cobrindo ≥80% da área construída de 2007; "
                "a restrição trunca ΔBCF negativo, então o resultado deve ser lido como associação em células sem lacuna aparente.",
    }
    registrar("2007→recente: Δanomalia × ΔBCF (urbano, cadastro ≥80% de 2007)", g[m2], "d_anom_07_rec", "d_bcf_07_at", ["d_ndvi_07_rec"])
    m2c = m2 & (g.urb_2007 >= URB_CONSOLIDADO) & (g.urb_atual >= URB_CONSOLIDADO)
    registrar("2007→recente: Δanomalia × ΔBCF (urbano consolidado, cadastro ≥80% de 2007)", g[m2c], "d_anom_07_rec", "d_bcf_07_at", ["d_ndvi_07_rec"])
    # (c) verticalização 2007
    registrar("~2007: anomalia × FAR 2007 (urbano)", g[g.cob_2007 & (g.urb_2007 >= 0.5)], "anom_2007", "far_2007", ["bcf_2007", "ndvi_2007"])
    # (d) árvores cortadas (exploratório: data do corte desconhecida)
    registrar("exploratório: Δanomalia 2007→recente × fração de árvores cortadas (cadastro ≥80% de 2007)", g[m2 & (g.arv_ha > 0)],
              "d_anom_07_rec", "frac_cortadas", ["d_bcf_07_at"])
    tab = pd.DataFrame(res)
    tab.to_csv(E5 / "resultados_regressoes.csv", index=False)

    # resumo por quintil de BCF e de árvores (recente, urbano)
    d = g[urbano_rec & g.cob_atual].dropna(subset=["anom_recente"])
    d["q_bcf"] = pd.qcut(d.bcf_atual, 5, labels=False, duplicates="drop")
    d["q_arv"] = pd.qcut(d.arv_ha.rank(method="first"), 5, labels=False)
    qb = d.groupby("q_bcf").agg(bcf_medio=("bcf_atual", "mean"), anom=("anom_recente", "mean"), n=("anom_recente", "size"))
    qa = d.groupby("q_arv").agg(arv_ha=("arv_ha", "mean"), bcf=("bcf_atual", "mean"), anom=("anom_recente", "mean"), n=("anom_recente", "size"))
    qb.to_csv(E5 / "quintis_bcf.csv")
    qa.to_csv(E5 / "quintis_arvores.csv")
    resumo["quintis_bcf"] = qb.round(3).to_dict("index")
    resumo["quintis_arvores"] = qa.round(3).to_dict("index")
    resumo["regressoes"] = tab.round(4).to_dict("records")
    m = g.cob_1995 & g.cob_2007 & g.cob_atual & (g.urb_1985 >= URB_CONSOLIDADO)
    resumo["urbano_1985_bcf_medio"] = {ep: float(g.loc[m, f"bcf_{ep}"].mean()) for ep in ("1995", "2007", "atual")}
    resumo["urbano_1985_anom_media"] = {k: float(g.loc[m, f"anom_{k}"].mean()) for k in JANELAS}
    json.dump(sem_nan(resumo), open(E5 / "resumo_etapa5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, allow_nan=False)

    # ---------- 5. Figuras ----------
    ext = [x0, x1, y0, y1]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    for a, (arr, tit, cm, vmin, vmax) in zip(axs, [
            (bcf["1995"], "Cobertura construída 1995", "magma_r", 0, 0.6),
            (bcf["2007"], "Cobertura construída 2007", "magma_r", 0, 0.6),
            (bcf["atual"], "Cobertura construída (cadastro atual)", "magma_r", 0, 0.6),
            (arv_plant / area_ha, "Árvores viárias por ha (inventário)", "Greens", 0, 30)]):
        im = a.imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax)
        a.set_title(tit, fontsize=10)
        a.set_axis_off()
        plt.colorbar(im, ax=a, shrink=0.7)
    fig.tight_layout()
    fig.savefig(E5 / "fig9_edificacoes_arvores_mapas.png", dpi=200)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for a, (msk, xcol, ycol, tit) in zip(axs, [
            (m1, "d_bcf_95_07", "d_anom_95_07", "1995 → 2007"),
            (m2, "d_bcf_07_at", "d_anom_07_rec", "2007 → recente (cadastro ≥80% de 2007)"),
            (urbano_rec & g.cob_atual, "arv_ha", "anom_recente", "Recente: árvores × anomalia")]):
        dd = g[msk][[xcol, ycol, "urb_1985"]].dropna()
        sc = a.scatter(dd[xcol] * (100 if "bcf" in xcol else 1), dd[ycol], c=dd["urb_1985"], cmap="viridis", s=8, alpha=0.6)
        a.axhline(0 if "d_" in ycol else np.nan, c="k", lw=0.5)
        a.set_xlabel("Δ cobertura construída (p.p.)" if "bcf" in xcol else "Árvores por ha")
        a.set_ylabel("Δ anomalia (°C)" if "d_" in ycol else "Anomalia (°C)")
        a.set_title(tit)
        plt.colorbar(sc, ax=a, label="fração urbana em 1985")
    fig.tight_layout()
    fig.savefig(E5 / "fig10_densificacao_arvores_vs_calor.png", dpi=200)

    print(json.dumps({k: resumo[k] for k in ("n_edificacoes", "area_construida_km2", "arvores",
                                             "urbano_1985_bcf_medio", "urbano_1985_anom_media", "lacuna_cadastro_atual")}, ensure_ascii=False, indent=2))
    print(tab.round(3).to_string(index=False))
    print(qb.round(3).to_string())
    print(qa.round(3).to_string())


if __name__ == "__main__":
    main()
