# Ilhas de calor urbanas em Cascavel (PR), 1985–2026

Código do estudo *Ilhas de calor urbanas em Cascavel (PR): perda de vegetação, urbanização e o papel dos parques*, submetido à 9ª edição do Prêmio MapBiomas.

O estudo reconstrói a temperatura de superfície de verão de Cascavel entre 1985 e 2026 com imagens Landsat, relaciona o calor às trajetórias de uso da terra do MapBiomas, estima o efeito térmico da urbanização e dos parques urbanos e cruza os resultados com o Censo 2022 para identificar onde a arborização é mais urgente.

## Scripts

| Script | O que faz |
|---|---|
| `scripts/00_baixar_dados.py` | Baixa as camadas públicas do GeoCascavel e os dados do Censo 2022 |
| `scripts/01_anomalia_mapbiomas.js` | Anomalia térmica dos verões recentes por classe do MapBiomas (Earth Engine Code Editor) |
| `scripts/02_series_1985_2026.py` | Séries de LST e NDVI, tendências, trajetórias de uso, estudo de evento e avaliação dos parques |
| `scripts/03_robustez_sensores.py` | Viés entre sensores Landsat, horário de passagem e cenários de tendência |
| `scripts/04_exposicao_censo.py` | Exposição da população, acesso a parques e praças e índice de prioridade para arborização |
| `scripts/05_edificacoes_arvores.py` | Cobertura construída (1995, 2007 e atual) e arborização viária |
| `scripts/06_figuras.py` | Gráficos do artigo e rasters para os mapas |

## Como reproduzir

Requer Python 3.10 ou superior e uma conta no Google Earth Engine vinculada a um projeto do Google Cloud.

```
pip install -r requirements.txt
earthengine authenticate
export EE_PROJECT=seu-projeto        # no Windows: set EE_PROJECT=seu-projeto

python scripts/00_baixar_dados.py
python scripts/02_series_1985_2026.py
python scripts/03_robustez_sensores.py
python scripts/04_exposicao_censo.py
python scripts/05_edificacoes_arvores.py
python scripts/06_figuras.py
```

Os dados e as saídas ficam em `dados/`. Para usar outra pasta, defina a variável `ILHAS_CALOR_DIR`. No Windows, se o terminal não exibir acentos, rode antes `set PYTHONIOENCODING=utf-8`.

O script `01_anomalia_mapbiomas.js` roda no [Code Editor do Earth Engine](https://code.earthengine.google.com) e exporta os resultados para o Google Drive.

Os mapas das figuras 2, 3b e 7 foram compostos no QGIS a partir dos rasters gerados por `06_figuras.py` e das camadas de `dados/`.

## Dados

| Fonte | Uso |
|---|---|
| Landsat 5, 7, 8 e 9, Coleção 2, Nível 2 (USGS), via Google Earth Engine | Temperatura de superfície e NDVI |
| MapBiomas Brasil, Coleção 11 | Cobertura e uso da terra, 1985–2025 |
| GeoCascavel / Instituto de Planejamento de Cascavel | Parques, praças, bairros, edificações e inventário de árvores |
| IBGE, Censo Demográfico 2022 | População, idade e renda por setor censitário |

`dados/parques_cascavel.geojson` reúne os parques da camada oficial do GeoCascavel com os anos de criação levantados para o estudo (campos `ano`, `status_ano` e `fonte_ano`). Dois polígonos foram delimitados pelo autor: o território completo do Parque Ecológico Paulo Gorski (lago, parque e área militar contígua) e o Bosque Elias Lopuch. Quando não foi encontrada fonte formal, o ano ficou em branco.

## Observações

- A série final exclui as cenas do Landsat 7 a partir do verão de 2020, quando o satélite entrou em deriva orbital (`L7_ULTIMO_VERAO` em `02_series_1985_2026.py`). Os testes de robustez relatados no artigo foram feitos sobre a série completa; para reproduzi-los exatamente, rode o script 02 com `L7_ULTIMO_VERAO = None` antes do script 03.
- O acervo Landsat e as coleções do MapBiomas são reprocessados periodicamente, o que pode gerar pequenas diferenças em relação aos números do artigo.
- O cadastro atual de edificações tem lacunas espaciais; por isso a variação entre 2007 e o presente não é usada para inferência.

## Licença

O código está sob a licença MIT. Os dados seguem as condições de cada fonte: o MapBiomas é de uso livre com citação, as imagens Landsat são de domínio público e os dados do IBGE e do GeoCascavel são públicos.

## Como citar

Rodrigues, G. B. (2026). *Ilhas de calor urbanas em Cascavel (PR), 1985–2026* [código-fonte]. https://github.com/USUARIO/ilhas-calor-cascavel
