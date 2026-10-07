"""Baixa os dados públicos usados no estudo, exceto as imagens Landsat e o MapBiomas,
que são lidos diretamente do Google Earth Engine pelos scripts 02 e 03.

- GeoCascavel (Prefeitura de Cascavel / IPC), via WFS: parques, praças, bairros,
  edificações de 1995 e 2007, geometria do cadastro atual de edificações e inventário
  de árvores. Do cadastro atual só se baixa a geometria; nenhum dado pessoal é usado.
- IBGE, Censo Demográfico 2022: malha de setores do Paraná com atributos e os
  agregados de demografia e de renda do responsável, recortados para Cascavel.

    python 00_baixar_dados.py
"""
import io
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

OUT = Path(os.environ.get("ILHAS_CALOR_DIR", Path(__file__).resolve().parents[1] / "dados"))
GEO = OUT / "geocascavel"
CENSO = OUT / "censo2022"
COD_MUN = "4104808"  # Cascavel

WFS = ("https://geoserver.cascavel.pr.gov.br:8181/geoserver/wfs?version=1.0.0&request=GetFeature"
       "&typeName=ctmgeo_22s:{}&outputFormat=application/json&srsName=EPSG:31982{}")
CAMADAS = {
    "ParquesCascavel": ("parques_ambientais", ""),
    "Pracas": ("pracas", ""),
    "BairrosCamadas": ("bairros", ""),
    "Edificacoes1995": ("edificacoes_1995", ""),
    "Edificacoes2007": ("edificacoes_2007", ""),
    "vw_edificacoes": ("edificacoes_atual_geometria", "&propertyName=wkb_geometry,nivel"),
    "vw_arvores": ("arvores_inventario", ""),
}

IBGE = "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
MALHA = IBGE + "Agregados_por_Setores_Censitarios/malha_com_atributos/setores/gpkg/UF/PR/PR_setores_CD2022.gpkg"
DEMOGRAFIA = IBGE + "Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/Agregados_por_setores_demografia_BR.zip"
RENDA = (IBGE + "Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/"
         "Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip")


def baixar(url, destino=None, contexto=None):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=900, context=contexto) as r:
        dados = r.read()
    if destino:
        destino.write_bytes(dados)
    return dados


def geocascavel():
    GEO.mkdir(parents=True, exist_ok=True)
    # o certificado do servidor da prefeitura nem sempre é reconhecido
    ctx = ssl._create_unverified_context()
    for camada, (nome, extra) in CAMADAS.items():
        p = GEO / f"{nome}.geojson"
        if p.exists():
            continue
        baixar(WFS.format(camada, extra), p, ctx)
        print(f"{nome}: {p.stat().st_size / 1e6:.1f} MB")


def ler_csv_zip(dados, colunas):
    zf = zipfile.ZipFile(io.BytesIO(dados))
    partes = pd.read_csv(zf.open(zf.namelist()[0]), sep=";", dtype=str, encoding="latin-1", chunksize=200_000)
    tab = pd.concat(p[p.iloc[:, 0].str.startswith(COD_MUN)] for p in partes)
    tab = tab.rename(columns={tab.columns[0]: "CD_SETOR"})
    num = {c: pd.to_numeric(tab[c].str.replace(",", ".", regex=False), errors="coerce") for c in colunas}
    return pd.DataFrame({"CD_SETOR": tab["CD_SETOR"], **num})


def censo():
    CENSO.mkdir(parents=True, exist_ok=True)
    malha = CENSO / "PR_setores_CD2022.gpkg"
    if not malha.exists():
        baixar(MALHA, malha)
    setores = gpd.read_file(malha)
    setores = setores[setores["CD_MUN"] == COD_MUN]

    dem = ler_csv_zip(baixar(DEMOGRAFIA), ["V01031", "V01032", "V01040", "V01041"])
    dem["pop_0a9"] = dem["V01031"] + dem["V01032"]
    dem["pop_60mais"] = dem["V01040"] + dem["V01041"]
    dem["pop_70mais"] = dem["V01041"]
    ren = ler_csv_zip(baixar(RENDA), ["V06004", "V06006"])
    ren = ren.rename(columns={"V06004": "renda_media_resp", "V06006": "renda_mediana_resp"})

    tab = (setores[["CD_SETOR", "SITUACAO", "NM_BAIRRO", "AREA_KM2", "v0001", "v0007", "geometry"]]
           .rename(columns={"v0001": "pop_total", "v0007": "domic_ocup"})
           .merge(dem[["CD_SETOR", "pop_0a9", "pop_60mais", "pop_70mais"]], on="CD_SETOR", how="left")
           .merge(ren, on="CD_SETOR", how="left"))
    tab.to_file(CENSO / "cascavel_setores_censo2022.gpkg", layer="setores_cascavel", driver="GPKG")
    print(f"setores de Cascavel: {len(tab)} | população: {int(tab['pop_total'].sum())}")


if __name__ == "__main__":
    geocascavel()
    censo()
