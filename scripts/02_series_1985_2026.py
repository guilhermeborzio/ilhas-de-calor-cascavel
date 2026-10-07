"""Séries de verão de temperatura de superfície (LST) e NDVI, Landsat 5, 7, 8 e 9, 1985 a 2026.

Baixa as composições de verão do Google Earth Engine e calcula a anomalia térmica em
relação à vegetação nativa estável, as tendências por pixel (Theil-Sen e Mann-Kendall),
as trajetórias de uso do MapBiomas, o estudo de evento da urbanização e a avaliação
antes-depois-controle-impacto dos parques.

    python 02_series_1985_2026.py                 # baixa do GEE e analisa
    python 02_series_1985_2026.py --sem-download  # só reanalisa os rasters já baixados

Verão Y = 1º de dezembro de Y-1 a 31 de março de Y.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmedian/nanmean de fatias vazias

# 1. CONFIGURAÇÃO
OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
SUB = OUT / "etapa2_series"
EE_PROJECT = os.environ.get("EE_PROJECT")  # projeto do Google Cloud com Earth Engine habilitado

# MapBiomas: usa o primeiro asset disponível da lista.
MB_CANDIDATOS = [
    "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3",
    "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1",
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2",
]

VERAO_INI, VERAO_FIM = 1985, 2026
L7_ULTIMO_VERAO = 2019      # o L7 entra em deriva orbital a partir do verão 2020; None mantém todas as cenas
MAX_NUVEM_CENA = 70        # % de nuvem por cena (o mascaramento por pixel faz o resto)
COBERTURA_MIN = 0.5        # fração mínima de pixels válidos para o verão entrar na série
MIN_ANOS_TENDENCIA = 20    # nº mínimo de verões válidos por pixel para Sen/Mann-Kendall
DIST_REF_URBANO_M = 1000   # referência rural: vegetação natural estável a > 1 km da mancha urbana
LAG_PARQUE = 1             # anos após a criação excluídos do período "depois" (implantação)

# Grade de referência: EPSG:4674, 642 x 398 pixels de ~30 m
XMIN, YMIN = -53.548843580949914, -25.011253466569364
XMAX, YMAX = -53.375828057228496, -24.903994621645495
RES = 0.000269494585236
CRS = "EPSG:4674"
CRS_TRANSFORM = [RES, 0, XMIN, 0, -RES, YMAX]
# tamanho aproximado do pixel em metros (lat ~ -24,96°)
DX_M = RES * 111320 * np.cos(np.radians(24.96))
DY_M = RES * 110574

# Agrupamento das classes do MapBiomas
GRUPOS = {1: ([3, 4, 5, 6, 49], "Vegetação natural"),
          2: ([9], "Silvicultura"),
          3: ([11, 12, 32, 50], "Campo/Área úmida"),
          4: ([15], "Pastagem"),
          5: ([18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48], "Agricultura"),
          6: ([21], "Mosaico de usos"),
          7: ([24], "Área urbanizada"),
          8: ([23, 25, 29, 30], "Outras não vegetadas"),
          9: ([31, 33], "Água")}
LUT = np.zeros(256, dtype=np.uint8)
for g, (cls, _) in GRUPOS.items():
    LUT[cls] = g


# 2. GOOGLE EARTH ENGINE — composições de verão e downloads
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


def baixar_dados_gee():
    import ee
    import requests

    ee.Initialize(project=EE_PROJECT)
    # meio pixel para dentro: força exatamente 642 x 398 pixels na grade de referência
    regiao = ee.Geometry.Rectangle([XMIN + RES / 2, YMIN + RES / 2, XMAX - RES / 2, YMAX - RES / 2], CRS, False)

    def prep(sensor):
        tm = sensor in ("L5", "L7")
        red, nir, st = ("SR_B3", "SR_B4", "ST_B6") if tm else ("SR_B4", "SR_B5", "ST_B10")

        def f(img):
            qa = img.select("QA_PIXEL")
            limpo = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                     .And(qa.bitwiseAnd(1 << 4).eq(0)).And(qa.bitwiseAnd(1 << 5).eq(0)))
            rn = img.select([red, nir], ["red", "nir"]).multiply(0.0000275).add(-0.2)
            if tm:  # harmonização TM/ETM+ -> OLI (Roy et al., 2016, OLS)
                rn = rn.multiply(ee.Image.constant([0.9047, 0.8462])).add(
                    ee.Image.constant([0.0061, 0.0412])).rename(["red", "nir"])
            ndvi = rn.normalizedDifference(["nir", "red"]).rename("NDVI")
            lst = img.select(st).multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
            ok = limpo.And(rn.select("red").gt(0)).And(rn.select("nir").gt(0)).And(lst.gt(5)).And(lst.lt(70))
            return ndvi.addBands(lst).updateMask(ok).copyProperties(img, ["system:time_start", "SPACECRAFT_ID"])
        return f

    cols = {"L5": "LANDSAT/LT05/C02/T1_L2", "L7": "LANDSAT/LE07/C02/T1_L2",
            "L8": "LANDSAT/LC08/C02/T1_L2", "L9": "LANDSAT/LC09/C02/T1_L2"}
    filtro_meses = ee.Filter.Or(ee.Filter.calendarRange(12, 12, "month"), ee.Filter.calendarRange(1, 3, "month"))
    full = None
    for s, cid in cols.items():
        c = (ee.ImageCollection(cid).filterBounds(regiao).filter(filtro_meses)
             .filter(ee.Filter.lt("CLOUD_COVER", MAX_NUVEM_CENA)).map(prep(s)))
        if s == "L7" and L7_ULTIMO_VERAO:
            c = c.filterDate("1999-01-01", f"{L7_ULTIMO_VERAO}-04-01")
        full = c if full is None else full.merge(c)

    # inventário de cenas (data, satélite)
    inv = full.reduceColumns(ee.Reducer.toList(2), ["system:time_start", "SPACECRAFT_ID"]).get("list").getInfo()
    df = pd.DataFrame(inv, columns=["t", "sat"])
    df["data"] = pd.to_datetime(df["t"], unit="ms")
    df["verao"] = np.where(df["data"].dt.month == 12, df["data"].dt.year + 1, df["data"].dt.year)
    df[["data", "sat", "verao"]].sort_values("data").to_csv(SUB / "inventario_cenas.csv", index=False)

    anos = list(range(VERAO_INI, VERAO_FIM + 1))
    vazio = ee.Image.constant([0, 0, 0]).rename(["NDVI", "LST", "N"]).toFloat().updateMask(ee.Image(0))
    nd_b, lst_b, n_b = [], [], []
    for y in anos:
        col = full.filterDate(f"{y-1}-12-01", f"{y}-04-01")
        comp = ee.Image(ee.Algorithms.If(
            col.size().gt(0),
            col.median().addBands(col.select("LST").count().rename("N")).select(["NDVI", "LST", "N"]).toFloat(),
            vazio))
        nd_b.append(comp.select("NDVI").multiply(10000).round().unmask(-32768).toInt16().rename(f"NDVI_{y}"))
        lst_b.append(comp.select("LST").multiply(100).round().unmask(-32768).toInt16().rename(f"LST_{y}"))
        n_b.append(comp.select("N").unmask(0).toUint8().rename(f"N_{y}"))

    mb_asset = None
    for a in MB_CANDIDATOS:
        try:
            ee.data.getAsset(a)
            mb_asset = a
            break
        except Exception:
            continue
    if mb_asset is None:
        raise RuntimeError("Nenhum asset MapBiomas de MB_CANDIDATOS foi encontrado no GEE.")
    mb = ee.Image(mb_asset)
    mb_bandas = mb.bandNames().getInfo()
    mb_anos = [int(b.split("_")[-1]) for b in mb_bandas]
    print(f"MapBiomas: {mb_asset} ({mb_anos[0]}–{mb_anos[-1]})")

    def baixar(img, nome):
        p = SUB / nome
        if p.exists():
            print("já existe:", p.name)
            return
        url = img.getDownloadURL({"region": regiao, "crs": CRS, "crs_transform": CRS_TRANSFORM, "format": "GEO_TIFF"})
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        p.write_bytes(r.content)
        print("baixado:", p.name, f"{len(r.content)/1e6:.1f} MB")

    baixar(ee.Image.cat(nd_b), "ndvi_veroes_x10000.tif")
    baixar(ee.Image.cat(lst_b), "lst_veroes_x100.tif")
    baixar(ee.Image.cat(n_b), "nobs_veroes.tif")
    baixar(mb.toUint8(), "mapbiomas_anual.tif")
    json.dump({"anos_verao": anos, "mapbiomas_asset": mb_asset, "mapbiomas_anos": mb_anos},
              open(SUB / "metadados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# 3. LEITURA
def ler():
    import rasterio
    meta = json.load(open(SUB / "metadados.json", encoding="utf-8"))
    with rasterio.open(SUB / "ndvi_veroes_x10000.tif") as s:
        nd = s.read().astype(np.float32)
        perfil, transform = s.profile, s.transform
    with rasterio.open(SUB / "lst_veroes_x100.tif") as s:
        lst = s.read().astype(np.float32)
    with rasterio.open(SUB / "nobs_veroes.tif") as s:
        nobs = s.read().astype(np.float32)
    with rasterio.open(SUB / "mapbiomas_anual.tif") as s:
        mb = s.read()
    nd[nd == -32768] = np.nan
    lst[lst == -32768] = np.nan
    nd /= 10000.0
    lst /= 100.0
    return meta, nd, lst, nobs, mb, perfil, transform


def salvar_tif(arr, nome, perfil):
    import rasterio
    p = perfil.copy()
    p.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(SUB / nome, "w", **p) as d:
        d.write(arr.astype(np.float32), 1)


# 4. FUNÇÕES DE ANÁLISE
def sen_mann_kendall(Y, t, min_n=MIN_ANOS_TENDENCIA, bloco=15000):
    """Declive de Theil-Sen e p-valor de Mann-Kendall por pixel. Y: (T, H, W)."""
    from scipy.stats import norm
    T, H, W = Y.shape
    Yf = Y.reshape(T, -1)
    P = Yf.shape[1]
    nval = np.isfinite(Yf).sum(0)
    i, j = np.triu_indices(T, 1)
    dt = (t[j] - t[i]).astype(np.float32)
    slope = np.full(P, np.nan, np.float32)
    pval = np.full(P, np.nan, np.float32)
    for s in range(0, P, bloco):
        y = Yf[:, s:s + bloco]
        d = y[j] - y[i]
        slope[s:s + bloco] = np.nanmedian(d / dt[:, None], axis=0)
        S = np.nansum(np.sign(d), axis=0)
        n = nval[s:s + bloco].astype(np.float64)
        var = n * (n - 1) * (2 * n + 5) / 18.0
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(S > 0, (S - 1) / np.sqrt(var), np.where(S < 0, (S + 1) / np.sqrt(var), 0.0))
        pval[s:s + bloco] = 2 * (1 - norm.cdf(np.abs(z)))
    ruim = nval < min_n
    slope[ruim] = np.nan
    pval[ruim] = np.nan
    return slope.reshape(H, W), pval.reshape(H, W)


def ano_conversao_urbana(U, mb_anos):
    """Primeiro ano de uma sequência ininterrupta de 'urbano' até o último ano MapBiomas."""
    c = np.full(U.shape[1:], np.nan, np.float32)
    seguindo = np.ones(U.shape[1:], bool)
    for k in range(len(mb_anos) - 1, -1, -1):
        seguindo &= U[k]
        c[seguindo] = mb_anos[k]
    return c


def mascara_poligonos(geoms, transform, shape):
    from rasterio.features import geometry_mask
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, bool)
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


# 5. ANÁLISE PRINCIPAL
def analisar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import distance_transform_edt

    meta, nd, lst, nobs, mb, perfil, transform = ler()
    anos = np.array(meta["anos_verao"])
    mb_anos = list(meta["mapbiomas_anos"])
    H, W = nd.shape[1:]
    G = LUT[mb]                       # grupos MapBiomas (n_mb, H, W)
    U = G == 7
    ult = G[-1]
    resumo = {}

    # ---------- 5.1 Qualidade anual ----------
    cobertura = np.isfinite(lst).reshape(len(anos), -1).mean(1)
    qual = pd.DataFrame({"verao": anos, "cobertura_lst": cobertura,
                         "cobertura_ndvi": np.isfinite(nd).reshape(len(anos), -1).mean(1),
                         "mediana_cenas_pixel": np.nanmedian(np.where(nobs > 0, nobs, np.nan).reshape(len(anos), -1), 1)})
    qual["usado"] = qual["cobertura_lst"] >= COBERTURA_MIN
    excl = ~qual["usado"].values
    lst[excl] = np.nan
    nd[excl] = np.nan

    # ---------- 5.2 Referência rural e anomalia anual ----------
    nat_estavel = (G == 1).all(0)
    dist_urb = distance_transform_edt(~U[-1], sampling=(DY_M, DX_M))
    ref = nat_estavel & (dist_urb > DIST_REF_URBANO_M)
    ref_t = np.array([np.nanmedian(lst[k][ref]) if np.isfinite(lst[k][ref]).sum() >= 50 else np.nan
                      for k in range(len(anos))])
    anom = lst - ref_t[:, None, None]
    qual["lst_ref_rural"] = ref_t
    qual["n_pixels_ref"] = int(ref.sum())
    qual.to_csv(SUB / "qualidade_e_referencia_anual.csv", index=False)
    resumo["veroes_usados"] = int(qual["usado"].sum())
    resumo["veroes_excluidos"] = qual.loc[~qual["usado"], "verao"].tolist()
    resumo["pixels_referencia_rural"] = int(ref.sum())

    # ---------- 5.3 Tendências por pixel ----------
    nd_slope, nd_p = sen_mann_kendall(nd, anos)
    an_slope, an_p = sen_mann_kendall(anom, anos)
    salvar_tif(nd_slope * 10, "ndvi_sen_por_decada.tif", perfil)
    salvar_tif(nd_p, "ndvi_mannkendall_p.tif", perfil)
    salvar_tif(an_slope * 10, "anomalia_sen_C_por_decada.tif", perfil)
    salvar_tif(an_p, "anomalia_mannkendall_p.tif", perfil)

    # ---------- 5.4 Trajetórias MapBiomas ----------
    c = ano_conversao_urbana(U, mb_anos)
    idx_c = {a: k for k, a in enumerate(mb_anos)}
    origem = np.zeros((H, W), np.uint8)
    for a in range(mb_anos[1], mb_anos[-1] + 1):
        sel = c == a
        origem[sel] = G[idx_c[a] - 1][sel]
    traj = np.zeros((H, W), np.uint8)
    traj[(c == mb_anos[0])] = 1                                   # urbano já em 1985
    traj[(c > mb_anos[0]) & (origem == 1)] = 2                    # vegetação natural -> urbano
    traj[(c > mb_anos[0]) & np.isin(origem, [4, 5, 6])] = 3       # agropecuária -> urbano
    traj[(c > mb_anos[0]) & ~np.isin(origem, [1, 4, 5, 6])] = 4   # outras origens -> urbano
    traj[nat_estavel] = 5                                         # vegetação natural estável
    traj[(G[0] == 1) & (ult != 1) & (ult != 7)] = 6               # vegetação perdida p/ uso não urbano
    traj[(G[0] != 1) & (ult == 1)] = 7                            # vegetação regenerada
    nomes_traj = {1: "Urbano consolidado (pré-1985)", 2: "Vegetação natural → urbano",
                  3: "Agropecuária → urbano", 4: "Outras origens → urbano",
                  5: "Vegetação natural estável", 6: "Vegetação → uso não urbano",
                  7: "Vegetação regenerada"}
    salvar_tif(traj.astype(np.float32), "trajetorias_mapbiomas.tif", perfil)

    k5i = np.isfinite(ref_t) & (anos <= anos[0] + 9)   # primeira década válida
    k5f = np.isfinite(ref_t) & (anos >= anos[-1] - 4)  # últimos 5 verões
    nd_i, nd_f = np.nanmean(nd[k5i], 0), np.nanmean(nd[k5f], 0)
    an_i, an_f = np.nanmean(anom[k5i], 0), np.nanmean(anom[k5f], 0)
    linhas = []
    for k, nome in nomes_traj.items():
        m = traj == k
        if m.sum() == 0:
            continue
        sig = np.isfinite(nd_p) & m
        linhas.append({"trajetoria": nome, "pixels": int(m.sum()),
                       "anom_inicio": np.nanmean(an_i[m]), "anom_fim": np.nanmean(an_f[m]),
                       "delta_anom": np.nanmean((an_f - an_i)[m]),
                       "ndvi_inicio": np.nanmean(nd_i[m]), "ndvi_fim": np.nanmean(nd_f[m]),
                       "ndvi_sen_decada": np.nanmean(nd_slope[m]) * 10,
                       "anom_sen_decada": np.nanmean(an_slope[m]) * 10,
                       "pct_esverdeamento_sig": 100 * np.mean((nd_slope[sig] > 0) & (nd_p[sig] < 0.05)) if sig.any() else np.nan,
                       "pct_acinzentamento_sig": 100 * np.mean((nd_slope[sig] < 0) & (nd_p[sig] < 0.05)) if sig.any() else np.nan})
    tab_traj = pd.DataFrame(linhas)
    tab_traj.to_csv(SUB / "trajetorias_ndvi_anomalia.csv", index=False)

    # urbano consolidado: variação do NDVI x variação da anomalia
    m = (traj == 1) & np.isfinite(nd_f - nd_i) & np.isfinite(an_f - an_i)
    rho, p_rho = stats.spearmanr((nd_f - nd_i)[m], (an_f - an_i)[m])
    resumo["urbano_consolidado_spearman_dNDVI_dAnom"] = {"rho": float(rho), "p": float(p_rho), "n": int(m.sum())}
    s1 = (traj == 1) & np.isfinite(nd_p)
    resumo["urbano_consolidado_pct_esverdeou_sig"] = float(100 * np.mean((nd_slope[s1] > 0) & (nd_p[s1] < 0.05)))
    resumo["urbano_consolidado_pct_acinzentou_sig"] = float(100 * np.mean((nd_slope[s1] < 0) & (nd_p[s1] < 0.05)))

    # ---------- 5.5 Estudo de evento (diferença em diferenças) ----------
    janela = np.arange(-8, 13)
    ev_linhas = []
    for org, rotulo, cod in [(1, "Vegetação natural → urbano", 2), (None, "Agropecuária → urbano", 3)]:
        ctrl = nat_estavel if org == 1 else np.isin(G, [4, 5, 6]).all(0)
        ctrl_an = np.nanmean(anom[:, ctrl], axis=1)
        ctrl_nd = np.nanmean(nd[:, ctrl], axis=1)
        sel = (traj == cod) & (c >= anos[0] + 5) & (c <= min(mb_anos[-1], anos[-1]) - 3)
        cc = c[sel]
        ra = anom[:, sel] - ctrl_an[:, None]
        rn = nd[:, sel] - ctrl_nd[:, None]
        rel = anos[:, None] - cc[None, :]
        for var, R in (("anomalia", ra), ("ndvi", rn)):
            base = np.nanmean(np.where((rel >= -5) & (rel <= -1), R, np.nan), axis=0)
            nb = np.isfinite(np.where((rel >= -5) & (rel <= -1), R, np.nan)).sum(0)
            Rb = R - base[None, :]
            Rb[:, nb < 2] = np.nan
            for k in janela:
                v = Rb[rel == k]
                v = v[np.isfinite(v)]
                if v.size >= 30:
                    ev_linhas.append({"trajetoria": rotulo, "variavel": var, "k": int(k), "n": int(v.size),
                                      "media": float(v.mean()), "ic95": float(1.96 * v.std(ddof=1) / np.sqrt(v.size))})
    ev = pd.DataFrame(ev_linhas)
    ev.to_csv(SUB / "estudo_evento_urbanizacao.csv", index=False)
    for (tr, var), d in ev.groupby(["trajetoria", "variavel"]):
        pos = d[(d.k >= 3) & (d.k <= 10)]
        resumo[f"efeito_{var}_{tr}"] = float(np.average(pos.media, weights=pos.n)) if len(pos) else None

    # ---------- 5.6 Parques: BACI e perfil de resfriamento ----------
    import geopandas as gpd
    pq_path = OUT / "parques_cascavel.geojson"
    pq = None
    if pq_path.exists():
        pq = gpd.read_file(pq_path)
        if "na_area_estudo" in pq.columns:  # camada oficial GeoCascavel: só parques dentro do recorte
            pq = pq[pq["na_area_estudo"].astype(bool)].copy()
        pq["ano"] = pd.to_numeric(pq["ano"], errors="coerce").fillna(-1).astype(int)  # -1 = ano incerto
        pq["geometry"] = pq.geometry.make_valid()
        pq = pq.dissolve(by=["nome", "ano"], as_index=False)
    if pq is not None:
        pq = pq.copy()
        pq["nome_base"] = pq["nome"].str.replace(r" \(.*\)$", "", regex=True)
        pq = pq.dissolve(by=["nome_base", "ano"], as_index=False)
        pm = pq.to_crs(31982)
        todos = pm.union_all() if hasattr(pm, "union_all") else pm.unary_union
        exclui = mascara_poligonos([gpd.GeoSeries([todos.buffer(300)], crs=31982).to_crs(CRS).iloc[0]], transform, (H, W))
        urb_ult = U[-1]
        baci, perfis, series = [], [], []
        for _, row in pm.iterrows():
            nome, ano_c = row["nome_base"], int(row["ano"])
            geo = row.geometry
            z = {"parque": geo, "0-150 m": geo.buffer(150).difference(geo),
                 "150-300 m": geo.buffer(300).difference(geo.buffer(150)),
                 "controle 500-1000 m": geo.buffer(1000).difference(geo.buffer(500))}
            mz = {k: mascara_poligonos([gpd.GeoSeries([g], crs=31982).to_crs(CRS).iloc[0]], transform, (H, W))
                  for k, g in z.items()}
            mz["controle 500-1000 m"] &= urb_ult & ~exclui
            for k, mk in mz.items():
                for t_i, y in enumerate(anos):
                    series.append({"parque": nome, "ano_criacao": ano_c, "zona": k, "verao": int(y),
                                   "anomalia": np.nanmean(anom[t_i][mk]) if mk.any() else np.nan,
                                   "ndvi": np.nanmean(nd[t_i][mk]) if mk.any() else np.nan,
                                   "pixels": int(mk.sum())})
            d_an = np.array([np.nanmean(anom[t][mz["parque"]]) - np.nanmean(anom[t][mz["controle 500-1000 m"]])
                             for t in range(len(anos))])
            d_nd = np.array([np.nanmean(nd[t][mz["parque"]]) - np.nanmean(nd[t][mz["controle 500-1000 m"]])
                             for t in range(len(anos))])
            antes = (anos < ano_c) & np.isfinite(d_an)
            depois = (anos > ano_c + LAG_PARQUE) & np.isfinite(d_an)
            reg = {"parque": nome, "ano_criacao": ano_c, "pixels_parque": int(mz["parque"].sum()),
                   "pixels_controle": int(mz["controle 500-1000 m"].sum()),
                   "n_antes": int(antes.sum()), "n_depois": int(depois.sum())}
            if antes.sum() >= 2 and depois.sum() >= 2:
                reg.update({"dif_anom_antes": d_an[antes].mean(), "dif_anom_depois": d_an[depois].mean(),
                            "efeito_BACI_anom_C": d_an[depois].mean() - d_an[antes].mean(),
                            "p_welch": stats.ttest_ind(d_an[depois], d_an[antes], equal_var=False).pvalue,
                            "efeito_BACI_ndvi": d_nd[depois].mean() - d_nd[antes].mean()})
                reg["tipo"] = "BACI" if depois.sum() >= 5 else "BACI preliminar (poucos verões após)"
            else:
                reg.update({"dif_anom_periodo_todo": np.nanmean(d_an), "dif_ndvi_periodo_todo": np.nanmean(d_nd)})
                if ano_c < 0:
                    reg["tipo"] = "ano de criação incerto — só nível e tendência"
                elif antes.sum() < 2:
                    reg["tipo"] = "sem período 'antes' na série Landsat — só nível e tendência"
                else:
                    reg["tipo"] = "parque recente — verões 'depois' insuficientes (linha de base pré-parque)"
            ok = np.isfinite(d_an) & (anos > ano_c + LAG_PARQUE)
            if ok.sum() >= 5:
                reg["tendencia_dif_anom_pos_C_decada"] = stats.theilslopes(d_an[ok], anos[ok])[0] * 10
            baci.append(reg)
            # perfil de resfriamento (últimos 5 verões), por distância à borda do parque
            dist = distance_transform_edt(~mz["parque"], sampling=(DY_M, DX_M))
            for b0 in range(0, 900, 60):
                mk = (dist > b0) & (dist <= b0 + 60) & ~np.isin(ult, [9])
                perfis.append({"parque": nome, "dist_m": b0 + 30, "anomalia": np.nanmean(an_f[mk]), "pixels": int(mk.sum())})
            perfis.append({"parque": nome, "dist_m": 0, "anomalia": np.nanmean(an_f[mz["parque"]]),
                           "pixels": int(mz["parque"].sum())})
        pd.DataFrame(baci).to_csv(SUB / "parques_baci.csv", index=False)
        pd.DataFrame(series).to_csv(SUB / "parques_series.csv", index=False)
        pd.DataFrame(perfis).sort_values(["parque", "dist_m"]).to_csv(SUB / "parques_perfil_resfriamento.csv", index=False)

    # ---------- 5.7 Figuras ----------
    ext = [XMIN, XMAX, YMIN, YMAX]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(anos, ref_t, "o-", c="tab:green")
    ax[0].set_ylabel("LST referência rural (°C)")
    for k, cor in [(1, "darkred"), (2, "orange"), (3, "goldenrod"), (5, "darkgreen")]:
        ax[1].plot(anos, [np.nanmean(anom[t][traj == k]) for t in range(len(anos))], "o-", ms=3, c=cor, label=nomes_traj[k])
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].set_ylabel("Anomalia térmica (°C)")
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel("Verão")
    fig.tight_layout()
    fig.savefig(SUB / "fig1_series_anomalia_trajetorias.png", dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, arr, tit, cm, lim in [(ax[0], nd_slope * 10, "NDVI — Sen (por década), 1985–2026", "BrBG", 0.15),
                                 (ax[1], an_slope * 10, "Anomalia térmica — Sen (°C por década)", "RdBu_r", 2.0)]:
        im = a.imshow(arr, extent=ext, cmap=cm, vmin=-lim, vmax=lim)
        a.set_title(tit)
        plt.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(SUB / "fig2_tendencias_pixel.png", dpi=200)

    if len(ev):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for a, var, yl in [(ax[0], "anomalia", "Δ anomalia vs. controle (°C)"), (ax[1], "ndvi", "Δ NDVI vs. controle")]:
            for tr, d in ev[ev.variavel == var].groupby("trajetoria"):
                a.errorbar(d.k, d.media, yerr=d.ic95, fmt="o-", ms=3, capsize=2, label=tr)
            a.axvline(0, c="k", ls="--", lw=0.8)
            a.axhline(0, c="k", lw=0.5)
            a.set_xlabel("Anos desde a conversão para urbano (MapBiomas)")
            a.set_ylabel(yl)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(SUB / "fig3_estudo_evento.png", dpi=200)

    if pq is not None:
        ser = pd.DataFrame(series)
        parques = ser.parque.unique()
        fig, axs = plt.subplots(len(parques), 1, figsize=(10, 2.6 * len(parques)), sharex=True)
        for a, p in zip(np.atleast_1d(axs), parques):
            d = ser[ser.parque == p].pivot(index="verao", columns="zona", values="anomalia")
            a.plot(d.index, d["parque"] - d["controle 500-1000 m"], "o-", ms=3)
            ac = ser[ser.parque == p].ano_criacao.iloc[0]
            if anos[0] <= ac <= anos[-1]:
                a.axvline(ac, c="green", ls="--")
            a.axhline(0, c="k", lw=0.5)
            a.set_title(p, fontsize=9)
            a.set_ylabel("parque − controle (°C)", fontsize=8)
        fig.tight_layout()
        fig.savefig(SUB / "fig4_parques_baci.png", dpi=200)
        per = pd.DataFrame(perfis)
        fig, a = plt.subplots(figsize=(8, 4.5))
        for p, d in per.sort_values("dist_m").groupby("parque"):
            a.plot(d.dist_m, d.anomalia, "o-", ms=3, label=p)
        a.set_xlabel("Distância da borda do parque (m)  [0 = interior]")
        a.set_ylabel("Anomalia média, últimos 5 verões (°C)")
        a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(SUB / "fig5_perfil_resfriamento.png", dpi=200)

    json.dump(sem_nan(resumo), open(SUB / "resumo_resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(sem_nan(resumo), ensure_ascii=False, indent=2, allow_nan=False))
    print(tab_traj.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-download", action="store_true", help="pula o GEE e só analisa os .tif existentes")
    args = ap.parse_args()
    SUB.mkdir(parents=True, exist_ok=True)
    if not args.sem_download:
        baixar_dados_gee()
    analisar()
