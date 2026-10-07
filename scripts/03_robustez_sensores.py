"""Testes de robustez da série de anomalia térmica.

A. Viés entre sensores nos verões de sobreposição (L5 x L7, L7 x L8, L8 x L9),
   com ganho estimado por regressão do eixo maior reduzido.
B. Horário de passagem de cada cena, por sensor e por verão.
C. Tendência do urbano consolidado em cinco cenários (S0 a S4), com Theil-Sen,
   Mann-Kendall, regressão com variáveis de era de sensor e regressão segmentada.

    python 03_robustez_sensores.py [--sem-download]

Depende das saídas de 02_series_1985_2026.py.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# CONFIGURAÇÃO
OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
E2 = OUT / "etapa2_series"
E3 = OUT / "etapa3_robustez"
EE_PROJECT = os.environ.get("EE_PROJECT")  # projeto do Google Cloud com Earth Engine habilitado

XMIN, YMIN = -53.548843580949914, -25.011253466569364
XMAX, YMAX = -53.375828057228496, -24.903994621645495
RES = 0.000269494585236
CRS = "EPSG:4674"
CRS_TRANSFORM = [RES, 0, XMIN, 0, -RES, YMAX]
DX_M = RES * 111320 * np.cos(np.radians(24.96))
DY_M = RES * 110574

MAX_NUVEM_CENA = 70
COBERTURA_MIN_PAR = 0.3     # fração mínima de pixels válidos em AMBOS os sensores no verão
DIST_REF_URBANO_M = 1000
MIN_CENAS_S1 = 3
VEROES_SUSPEITOS = [1992]

PARES = {
    "L5_L7": ("L5", "L7", list(range(2000, 2012))),
    "L7_L8": ("L7", "L8", list(range(2014, 2023))),
    "L8_L9": ("L8", "L9", list(range(2022, 2027))),
}
COLECOES = {"L5": "LANDSAT/LT05/C02/T1_L2", "L7": "LANDSAT/LE07/C02/T1_L2",
            "L8": "LANDSAT/LC08/C02/T1_L2", "L9": "LANDSAT/LC09/C02/T1_L2"}
SAT_ID = {"LANDSAT_5": "L5", "LANDSAT_7": "L7", "LANDSAT_8": "L8", "LANDSAT_9": "L9"}

# mesmos grupos MapBiomas das etapas 1 e 2
GRUPOS = {1: [3, 4, 5, 6, 49], 2: [9], 3: [11, 12, 32, 50], 4: [15],
          5: [18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48], 6: [21], 7: [24],
          8: [23, 25, 29, 30], 9: [31, 33]}
LUT = np.zeros(256, dtype=np.uint8)
for g, cls in GRUPOS.items():
    LUT[cls] = g


# A/B. GEE — composições por sensor nos períodos de sobreposição + inventário
def baixar_gee():
    import ee
    import requests

    ee.Initialize(project=EE_PROJECT)
    regiao = ee.Geometry.Rectangle([XMIN + RES / 2, YMIN + RES / 2, XMAX - RES / 2, YMAX - RES / 2], CRS, False)
    meses = ee.Filter.Or(ee.Filter.calendarRange(12, 12, "month"), ee.Filter.calendarRange(1, 3, "month"))

    def prep(sensor):
        tm = sensor in ("L5", "L7")
        red, nir, st = ("SR_B3", "SR_B4", "ST_B6") if tm else ("SR_B4", "SR_B5", "ST_B10")

        def f(img):
            qa = img.select("QA_PIXEL")
            limpo = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                     .And(qa.bitwiseAnd(1 << 4).eq(0)).And(qa.bitwiseAnd(1 << 5).eq(0)))
            rn = img.select([red, nir], ["red", "nir"]).multiply(0.0000275).add(-0.2)
            if tm:  # TM/ETM+ -> OLI (Roy et al., 2016)
                rn = rn.multiply(ee.Image.constant([0.9047, 0.8462])).add(
                    ee.Image.constant([0.0061, 0.0412])).rename(["red", "nir"])
            ndvi = rn.normalizedDifference(["nir", "red"]).rename("NDVI")
            lst = img.select(st).multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
            ok = limpo.And(rn.select("red").gt(0)).And(rn.select("nir").gt(0)).And(lst.gt(5)).And(lst.lt(70))
            return ndvi.addBands(lst).updateMask(ok).copyProperties(img, ["system:time_start"])
        return f

    def colecao(sensor, cru=False):
        c = (ee.ImageCollection(COLECOES[sensor]).filterBounds(regiao).filter(meses)
             .filter(ee.Filter.lt("CLOUD_COVER", MAX_NUVEM_CENA)))
        return c if cru else c.map(prep(sensor))

    # Inventário com horário de passagem (todas as cenas de verão, 1985–2026)
    linhas = []
    for s in COLECOES:
        inv = colecao(s, cru=True).reduceColumns(
            ee.Reducer.toList(3), ["system:time_start", "SCENE_CENTER_TIME", "CLOUD_COVER"]).get("list").getInfo()
        for t, hora, nuvem in inv:
            linhas.append({"sensor": s, "t": t, "hora_utc": hora, "nuvem_cena": nuvem})
    inv = pd.DataFrame(linhas)
    inv["data"] = pd.to_datetime(inv["t"], unit="ms")
    inv["verao"] = np.where(inv["data"].dt.month == 12, inv["data"].dt.year + 1, inv["data"].dt.year)
    hh = inv["hora_utc"].str.slice(0, 8).str.split(":", expand=True).astype(float)
    inv["hora_local_dec"] = hh[0] + hh[1] / 60 + hh[2] / 3600 - 3  # Cascavel: UTC-3
    inv.drop(columns="t").sort_values("data").to_csv(E3 / "inventario_cenas_por_sensor.csv", index=False)

    vazio = ee.Image.constant([0, 0, 0]).rename(["NDVI", "LST", "N"]).toFloat().updateMask(ee.Image(0))

    def baixar(img, nome):
        p = E3 / nome
        if p.exists():
            print("já existe:", nome)
            return
        url = img.getDownloadURL({"region": regiao, "crs": CRS, "crs_transform": CRS_TRANSFORM, "format": "GEO_TIFF"})
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        p.write_bytes(r.content)
        print("baixado:", nome, f"{len(r.content)/1e6:.1f} MB")

    meta = {}
    for par, (s1, s2, anos) in PARES.items():
        nd_b, lst_b, n_b, ordem = [], [], [], []
        for y in anos:
            for s in (s1, s2):
                col = colecao(s).filterDate(f"{y-1}-12-01", f"{y}-04-01")
                comp = ee.Image(ee.Algorithms.If(
                    col.size().gt(0),
                    col.median().addBands(col.select("LST").count().rename("N")).select(["NDVI", "LST", "N"]).toFloat(),
                    vazio))
                tag = f"{s}_{y}"
                nd_b.append(comp.select("NDVI").multiply(10000).round().unmask(-32768).toInt16().rename(f"NDVI_{tag}"))
                lst_b.append(comp.select("LST").multiply(100).round().unmask(-32768).toInt16().rename(f"LST_{tag}"))
                n_b.append(comp.select("N").unmask(0).toUint8().rename(f"N_{tag}"))
                ordem.append([s, y])
        baixar(ee.Image.cat(lst_b), f"{par}_lst_x100.tif")
        baixar(ee.Image.cat(nd_b), f"{par}_ndvi_x10000.tif")
        baixar(ee.Image.cat(n_b), f"{par}_nobs.tif")
        meta[par] = ordem
    json.dump(meta, open(E3 / "metadados_pares.json", "w", encoding="utf-8"), indent=1)


# Utilidades
def ler_tif(p, escala=None):
    import rasterio
    with rasterio.open(p) as s:
        a = s.read().astype(np.float32)
    if escala:
        a[a == -32768] = np.nan
        a /= escala
    return a


def mascaras_referencia():
    from scipy.ndimage import distance_transform_edt
    mb = ler_tif(E2 / "mapbiomas_anual.tif").astype(np.uint8)
    G = LUT[mb]
    longe = distance_transform_edt(~(G[-1] == 7), sampling=(DY_M, DX_M)) > DIST_REF_URBANO_M
    ref_nat = (G == 1).all(0) & longe
    ref_agro = np.isin(G, [4, 5, 6]).all(0) & longe
    traj = ler_tif(E2 / "trajetorias_mapbiomas.tif")[0]
    return ref_nat, ref_agro, traj


def sen_mk(x, y):
    from scipy import stats
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    sl, _, lo, hi = stats.theilslopes(y, x)
    tau, p = stats.kendalltau(x, y)
    return sl * 10, lo * 10, hi * 10, p, len(x)


def rma(x, y):
    """Regressão de eixo maior reduzido (RMA): simétrica, sem atenuação por ruído em x."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = np.corrcoef(x, y)[0, 1]
    g = np.sign(r) * np.std(y) / np.std(x)
    return g, np.mean(y) - g * np.mean(x)


def ols(X, y):
    """OLS com erro-padrão clássico. Retorna coeficientes, EP, p."""
    from scipy import stats
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    gl = len(y) - X.shape[1]
    s2 = res @ res / gl
    ep = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    p = 2 * (1 - stats.t.cdf(np.abs(b / ep), gl))
    return b, ep, p


def segmentada(x, y, quebras=range(1995, 2021)):
    ok = np.isfinite(y)
    x, y = x[ok].astype(float), y[ok]
    melhor = None
    for q in quebras:
        X = np.column_stack([np.ones_like(x), x - q, np.clip(x - q, 0, None)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ b) ** 2).sum()
        if melhor is None or sse < melhor[0]:
            melhor = (sse, q, b)
    sse, q, b = melhor
    return {"ano_quebra": q, "inclinacao_antes_C_dec": b[1] * 10, "inclinacao_depois_C_dec": (b[1] + b[2]) * 10}


# ANÁLISE
def analisar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_nat, ref_agro, traj = mascaras_referencia()
    meta = json.load(open(E3 / "metadados_pares.json", encoding="utf-8"))
    rng = np.random.default_rng(42)
    resumo = {"pixels_ref_natural": int(ref_nat.sum()), "pixels_ref_agro": int(ref_agro.sum())}

    # ---------- A. Viés entre sensores ----------
    por_ano, correcoes, amostras = [], {}, {}
    for par, ordem in meta.items():
        s1, s2 = PARES[par][0], PARES[par][1]
        L = ler_tif(E3 / f"{par}_lst_x100.tif", 100)
        Nd = ler_tif(E3 / f"{par}_ndvi_x10000.tif", 10000)
        idx = {(s, y): k for k, (s, y) in enumerate(ordem)}
        xa, ya, xn, yn = [], [], [], []
        for y in PARES[par][2]:
            a1, a2 = L[idx[(s1, y)]], L[idx[(s2, y)]]
            ok = np.isfinite(a1) & np.isfinite(a2)
            if ok.mean() < COBERTURA_MIN_PAR or np.isfinite(a1[ref_nat]).sum() < 50 or np.isfinite(a2[ref_nat]).sum() < 50:
                continue
            an1 = a1 - np.nanmedian(a1[ref_nat])
            an2 = a2 - np.nanmedian(a2[ref_nat])
            n1, n2 = Nd[idx[(s1, y)]], Nd[idx[(s2, y)]]
            u = ok & (traj == 1)
            g, o = rma(an1[ok], an2[ok])
            por_ano.append({"par": par, "verao": y, "pixels_pareados": int(ok.sum()),
                            "dif_LST_abs_media": float(np.nanmean((a2 - a1)[ok])),
                            "dif_anom_media": float(np.nanmean((an2 - an1)[ok])),
                            "dif_anom_urbano_consolidado": float(np.nanmean((an2 - an1)[u])),
                            "anom_urb_" + s1: float(np.nanmean(an1[u])), "anom_urb_" + s2: float(np.nanmean(an2[u])),
                            "ganho_anom": float(g), "offset_anom": float(o),
                            "r_anom": float(np.corrcoef(an1[ok], an2[ok])[0, 1]),
                            "rmsd_anom": float(np.sqrt(np.nanmean((an2 - an1)[ok] ** 2))),
                            "dif_ndvi_media": float(np.nanmean((n2 - n1)[ok & np.isfinite(n1) & np.isfinite(n2)])),
                            "dif_ndvi_urbano_consolidado": float(np.nanmean((n2 - n1)[u & np.isfinite(n1) & np.isfinite(n2)]))})
            k = np.flatnonzero(ok.ravel())
            k = rng.choice(k, min(15000, k.size), replace=False)
            xa.append(an1.ravel()[k]); ya.append(an2.ravel()[k])
            okn = (ok & np.isfinite(n1) & np.isfinite(n2)).ravel()
            kn = rng.choice(np.flatnonzero(okn), min(15000, okn.sum()), replace=False)
            xn.append(n1.ravel()[kn]); yn.append(n2.ravel()[kn])
        if xa:
            xa, ya = np.concatenate(xa), np.concatenate(ya)
            xn, yn = np.concatenate(xn), np.concatenate(yn)
            ga, oa = rma(xa, ya)
            gn, on = rma(xn, yn)
            correcoes[par] = {"de": s1, "para": s2, "ganho_anom": float(ga), "offset_anom": float(oa),
                              "ganho_ndvi": float(gn), "offset_ndvi": float(on),
                              "veroes_validos": int(sum(1 for r in por_ano if r["par"] == par))}
            amostras[par] = (xa, ya)
    tab_a = pd.DataFrame(por_ano)
    tab_a.to_csv(E3 / "A_vies_sensores_por_verao.csv", index=False)
    resumo["vies_medio_por_par"] = (tab_a.groupby("par")[["dif_anom_media", "dif_anom_urbano_consolidado", "ganho_anom",
                                                         "r_anom", "rmsd_anom", "dif_ndvi_media",
                                                         "dif_ndvi_urbano_consolidado"]].mean().round(3).to_dict("index"))
    resumo["correcoes_pooled"] = correcoes

    # Correções encadeadas para a escala do L8: y_L8 = g * x + o
    def compor(c1, c2):  # aplica c1 e depois c2
        return {"g": c2["g"] * c1["g"], "o": c2["g"] * c1["o"] + c2["o"]}
    ident = {"g": 1.0, "o": 0.0}
    c57 = {"g": correcoes.get("L5_L7", {}).get("ganho_anom", 1.0), "o": correcoes.get("L5_L7", {}).get("offset_anom", 0.0)}
    c78 = {"g": correcoes.get("L7_L8", {}).get("ganho_anom", 1.0), "o": correcoes.get("L7_L8", {}).get("offset_anom", 0.0)}
    c89 = {"g": correcoes.get("L8_L9", {}).get("ganho_anom", 1.0), "o": correcoes.get("L8_L9", {}).get("offset_anom", 0.0)}
    para_L8 = {"L5": compor(c57, c78), "L7": c78, "L8": ident,
               "L9": {"g": 1 / c89["g"], "o": -c89["o"] / c89["g"]}}
    resumo["correcao_anomalia_para_escala_L8"] = para_L8

    # ---------- B. Horário de passagem ----------
    inv = pd.read_csv(E3 / "inventario_cenas_por_sensor.csv")
    hora = inv.groupby(["sensor", "verao"])["hora_local_dec"].mean().unstack(0)
    hora.to_csv(E3 / "B_horario_passagem_local.csv")
    resumo["horario_local_medio_por_sensor"] = inv.groupby("sensor")["hora_local_dec"].agg(["mean", "min", "max"]).round(2).to_dict("index")

    # ---------- C. Sensibilidade da tendência ----------
    meta2 = json.load(open(E2 / "metadados.json", encoding="utf-8"))
    anos = np.array(meta2["anos_verao"])
    L2 = ler_tif(E2 / "lst_veroes_x100.tif", 100)
    q2 = pd.read_csv(E2 / "qualidade_e_referencia_anual.csv")
    usado = q2.set_index("verao").loc[anos, "usado"].values.astype(bool)
    L2[~usado] = np.nan
    inv2 = pd.read_csv(E2 / "inventario_cenas.csv")
    inv2["s"] = inv2["sat"].map(SAT_ID)
    cont = inv2.groupby(["verao", "s"]).size().unstack(fill_value=0).reindex(anos, fill_value=0)
    n_cenas = cont.sum(axis=1).values

    # correção por verão = média das correções dos sensores ponderada pelo nº de cenas
    g_ano, o_ano = np.ones(len(anos)), np.zeros(len(anos))
    for k, y in enumerate(anos):
        if n_cenas[k] == 0:
            continue
        w = {s: cont.loc[y, s] / n_cenas[k] for s in cont.columns}
        g_ano[k] = sum(w[s] * para_L8[s]["g"] for s in w)
        o_ano[k] = sum(w[s] * para_L8[s]["o"] for s in w)

    def serie(ref, corrigir=False, filtrar=False):
        out = np.full(len(anos), np.nan)
        for k in range(len(anos)):
            if filtrar and (n_cenas[k] < MIN_CENAS_S1 or anos[k] in VEROES_SUSPEITOS):
                continue
            a = L2[k]
            if np.isfinite(a[ref]).sum() < 50:
                continue
            an = a - np.nanmedian(a[ref])
            if corrigir:
                an = g_ano[k] * an + o_ano[k]
            out[k] = np.nanmean(an[traj == 1])
        return out

    cenarios = {"S0 original": serie(ref_nat),
                "S1 sem verões <3 cenas e 1992": serie(ref_nat, filtrar=True),
                "S2 correção entre sensores": serie(ref_nat, corrigir=True),
                "S3 referência agropecuária": serie(ref_agro),
                "S4 = S1+S2+S3": serie(ref_agro, corrigir=True, filtrar=True)}
    era = np.where(anos <= 1999, 0, np.where(anos <= 2013, 1, 2))
    linhas = []
    for nome, y in cenarios.items():
        r = {"cenario": nome, "veroes": int(np.isfinite(y).sum())}
        for rot, m in [("1985-2026", anos >= 0), ("1985-2011", anos <= 2011), ("2014-2026", anos >= 2014)]:
            sl, lo, hi, p, n = sen_mk(anos[m].astype(float), y[m])
            r[f"sen_{rot}_C_dec"], r[f"ic95_{rot}"], r[f"p_mk_{rot}"] = sl, f"[{lo:.2f}; {hi:.2f}]", p
        X = np.column_stack([np.ones(len(anos)), anos - 2000, era == 1, era == 2]).astype(float)
        b, ep, p = ols(X, y)
        r["ols_ano_com_dummy_era_C_dec"], r["ols_p_ano"] = b[1] * 10, p[1]
        r["ols_degrau_era_2000_2013_C"], r["ols_degrau_era_2014_2026_C"] = b[2], b[3]
        r.update(segmentada(anos, y))
        r["media_1985_1999"], r["media_2014_2026"] = np.nanmean(y[anos <= 1999]), np.nanmean(y[anos >= 2014])
        linhas.append(r)
    tab_c = pd.DataFrame(linhas)
    tab_c.to_csv(E3 / "C_sensibilidade_tendencia_urbano_consolidado.csv", index=False)
    pd.DataFrame({"verao": anos, "n_cenas": n_cenas, "ganho_corr": g_ano, "offset_corr": o_ano,
                  **{k: v for k, v in cenarios.items()}}).to_csv(E3 / "C_series_por_cenario.csv", index=False)

    # ---------- Figuras ----------
    if amostras:
        fig, axs = plt.subplots(1, len(amostras), figsize=(5 * len(amostras), 4.6))
        for a, (par, (x, y)) in zip(np.atleast_1d(axs), amostras.items()):
            a.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
            lim = [np.nanpercentile(np.r_[x, y], 0.5), np.nanpercentile(np.r_[x, y], 99.5)]
            a.plot(lim, lim, "w--", lw=1)
            c = correcoes[par]
            a.plot(lim, [c["ganho_anom"] * v + c["offset_anom"] for v in lim], "r-", lw=1.2,
                   label=f"y = {c['ganho_anom']:.2f}x {c['offset_anom']:+.2f}")
            a.set_xlabel(f"anomalia {c['de']} (°C)")
            a.set_ylabel(f"anomalia {c['para']} (°C)")
            a.set_title(par)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(E3 / "figR1_vies_sensores.png", dpi=200)

    fig, a = plt.subplots(figsize=(10, 4.8))
    for nome, y in cenarios.items():
        a.plot(anos, y, "o-", ms=3, lw=1, label=nome)
    for x0 in (1999.5, 2013.5):
        a.axvline(x0, c="gray", ls=":", lw=1)
    a.set_ylabel("Anomalia média — urbano consolidado (°C)")
    a.set_xlabel("Verão")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(E3 / "figR2_cenarios_tendencia.png", dpi=200)

    fig, a = plt.subplots(figsize=(10, 3.8))
    for s in hora.columns:
        a.plot(hora.index, hora[s], "o-", ms=3, label=s)
    a.set_ylabel("Horário local médio da passagem (h)")
    a.set_xlabel("Verão")
    a.legend()
    fig.tight_layout()
    fig.savefig(E3 / "figR3_horario_passagem.png", dpi=200)

    json.dump(resumo, open(E3 / "resumo_robustez.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps(resumo, ensure_ascii=False, indent=2, default=float))
    print(tab_c.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-download", action="store_true")
    args = ap.parse_args()
    E3.mkdir(parents=True, exist_ok=True)
    if not args.sem_download:
        baixar_gee()
    analisar()
