// Anomalia térmica de superfície em Cascavel (PR) e sua relação com as classes do MapBiomas.
// Google Earth Engine (Code Editor). Compósito Landsat 8/9 de cinco verões (dez–mar).
//
// A área de análise e a grade de exportação reproduzem o recorte usado em todo o estudo:
// EPSG:4674, 642 x 398 pixels de 0,000269494585236 grau (~30 m).

// ---------------------------------------------------------------------
// 1. PARÂMETROS
// ---------------------------------------------------------------------
var PARAMS = {
  // MapBiomas Brasil, Coleção 11
  mbAsset: 'projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3',
  anoMapBiomas: 2025,
  anoBase: 1985,

  // Cinco verões; veraoIni é o ano do dezembro do primeiro verão (2021 = verões 2021/22 a 2025/26)
  veraoIni: 2021,
  veraoFim: 2025,
  maxNuvem: 40,

  // Anomalia: 'rural'  = LST − mediana da vegetação natural da própria área,
  //                      afastada da mancha urbana (bufferUrbano)
  //           'zscore' = (LST − média da área) / desvio-padrão
  metodoAnomalia: 'rural',
  classesReferencia: [3, 4, 11, 12], // floresta, savana, área úmida, campo
  bufferUrbano: 1000, // m — exclui da referência a vegetação colada à cidade

  // raster de referência opcional, para conferir a consistência da LST
  refAsset: null, // ex.: 'projects/SEU_PROJETO/assets/lst_referencia'

  pastaDrive: 'ilhas_calor_cascavel'
};

// Recorte e grade de referência
var XMIN = -53.548843580949914, YMIN = -25.011253466569364;
var XMAX = -53.375828057228496, YMAX = -24.903994621645495;
var RES = 0.000269494585236;
var CRS = 'EPSG:4674';
var CRS_TRANSFORM = [RES, 0, XMIN, 0, -RES, YMAX];

var area = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
var aoi = area.buffer(3000); // margem para operações focais

// ---------------------------------------------------------------------
// 2. MAPBIOMAS — uso atual, agrupamento e histórico de urbanização
// ---------------------------------------------------------------------
var mb = ee.Image(PARAMS.mbAsset);
var lulc = mb.select('classification_' + PARAMS.anoMapBiomas).clip(aoi);

var DE = [3,4,5,6,49, 9, 11,12,32,50, 15, 18,19,39,20,40,62,41,36,46,47,35,48, 21, 24, 23,25,29,30, 31,33];
var PARA = [1,1,1,1,1, 2, 3,3,3,3, 4, 5,5,5,5,5,5,5,5,5,5,5,5, 6, 7, 8,8,8,8, 9,9];
var NOMES = ['Vegetação natural', 'Silvicultura', 'Campo/Área úmida', 'Pastagem',
             'Agricultura', 'Mosaico de usos', 'Área urbanizada', 'Outras não vegetadas', 'Água'];
var grupo = lulc.remap(DE, PARA).rename('grupo');

// Primeiro ano em que o pixel aparece como área urbanizada
var anos = ee.List.sequence(PARAMS.anoBase, PARAMS.anoMapBiomas);
var anoUrb = ee.ImageCollection(anos.map(function (a) {
  a = ee.Number(a);
  var banda = ee.String('classification_').cat(a.format('%d'));
  return mb.select(banda).eq(24).multiply(a).selfMask().toInt().rename('ano_urb');
})).min().clip(aoi);

// Transições anoBase → anoMapBiomas
var base = mb.select('classification_' + PARAMS.anoBase).clip(aoi);
var natBase = base.remap(DE, PARA).eq(1);
var natAtual = grupo.eq(1);
var urbBase = base.eq(24);
var urbAtual = lulc.eq(24);
var transicao = ee.Image(0)
  .where(urbBase.and(urbAtual), 1)
  .where(urbBase.not().and(urbAtual), 2)
  .where(natBase.and(natAtual), 3)
  .where(natBase.and(natAtual.not()).and(urbAtual.not()), 4)
  .where(natBase.not().and(natAtual), 5)
  .selfMask().rename('transicao').clip(aoi);
var NOMES_TRANS = ['Urbano consolidado', 'Urbanizado após ' + PARAMS.anoBase,
                   'Vegetação estável', 'Vegetação perdida', 'Vegetação regenerada'];

// ---------------------------------------------------------------------
// 3. LST E NDVI — Landsat 8/9 C2 L2, cinco verões
// ---------------------------------------------------------------------
function prepLandsat(img) {
  var qa = img.select('QA_PIXEL');
  var limpo = qa.bitwiseAnd(1 << 1).eq(0)
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  var lst = img.select('ST_B10').multiply(0.00341802).add(149.0)
    .subtract(273.15).rename('LST');
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  var ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  return lst.addBands(ndvi).updateMask(limpo)
    .copyProperties(img, ['system:time_start']);
}

var filtros = [];
for (var y = PARAMS.veraoIni; y <= PARAMS.veraoFim; y++) {
  filtros.push(ee.Filter.date(y + '-12-01', (y + 1) + '-04-01'));
}
var filtroVeroes = ee.Filter.or.apply(null, filtros);

var colecao = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(area)
  .filter(filtroVeroes)
  .filter(ee.Filter.lt('CLOUD_COVER', PARAMS.maxNuvem))
  .map(prepLandsat);

print('Cenas no compósito:', colecao.size());
print('Datas:', colecao.aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }));

var composto = colecao.median().clip(aoi);
var nObs = colecao.select('LST').count().rename('n_obs').clip(aoi);
var lst = composto.select('LST');

// ---------------------------------------------------------------------
// 4. ANOMALIA TÉRMICA
// ---------------------------------------------------------------------
var pertoUrbano = urbAtual.focalMax(PARAMS.bufferUrbano, 'circle', 'meters');
var refMask = lulc.remap(PARAMS.classesReferencia,
    ee.List.repeat(1, PARAMS.classesReferencia.length), 0)
  .and(pertoUrbano.not());

var anomalia;
if (PARAMS.metodoAnomalia === 'rural') {
  var medRural = ee.Number(lst.updateMask(refMask).reduceRegion({
    reducer: ee.Reducer.median(), geometry: area,
    crs: CRS, crsTransform: CRS_TRANSFORM, maxPixels: 1e10
  }).get('LST'));
  print('LST mediana da referência rural (°C):', medRural);
  anomalia = lst.subtract(medRural).rename('anomalia');
} else {
  var st = lst.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
    geometry: area, crs: CRS, crsTransform: CRS_TRANSFORM, maxPixels: 1e10
  });
  anomalia = lst.subtract(ee.Number(st.get('LST_mean')))
    .divide(ee.Number(st.get('LST_stdDev'))).rename('anomalia');
}

// ---------------------------------------------------------------------
// 5. ESTATÍSTICAS ZONAIS
// ---------------------------------------------------------------------
function zonal(imgClasse, nomes, regiao) {
  var r = anomalia.addBands(imgClasse).reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), null, true)
      .combine(ee.Reducer.count(), null, true)
      .group({groupField: 1, groupName: 'classe'}),
    geometry: regiao, crs: CRS, crsTransform: CRS_TRANSFORM, maxPixels: 1e10
  });
  return ee.FeatureCollection(ee.List(r.get('groups')).map(function (d) {
    d = ee.Dictionary(d);
    var nome = ee.List(nomes).get(ee.Number(d.get('classe')).subtract(1));
    return ee.Feature(null, d.set('nome', nome));
  }));
}

var statsGrupo = zonal(grupo, NOMES, area);
var statsTrans = zonal(transicao, NOMES_TRANS, area);
print('Anomalia por classe MapBiomas:', statsGrupo);
print('Anomalia por transição:', statsTrans);

// Consistência com o raster de referência (opcional)
if (PARAMS.refAsset) {
  var ref = ee.Image(PARAMS.refAsset).rename('LST_ref');
  var dif = lst.subtract(ref).rename('dif');
  print('LST nova − referência (média, dp, °C):', dif.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
    geometry: area, crs: CRS, crsTransform: CRS_TRANSFORM, maxPixels: 1e10
  }));
}

// Pilha final e amostra estratificada para análise em Python
var pilha = anomalia.addBands(lst).addBands(composto.select('NDVI'))
  .addBands(grupo).addBands(transicao.unmask(0)).addBands(anoUrb.unmask(0))
  .addBands(nObs).clip(area);

var amostra = pilha.stratifiedSample({
  numPoints: 500, classBand: 'grupo', region: area,
  projection: ee.Projection(CRS, CRS_TRANSFORM),
  geometries: true, seed: 42
});

// ---------------------------------------------------------------------
// 6. VISUALIZAÇÃO
// ---------------------------------------------------------------------
Map.centerObject(area, 13);
var palTermica = ['#313695', '#4575b4', '#abd9e9', '#ffffbf', '#fdae61', '#d73027', '#a50026'];
Map.addLayer(grupo.clip(area), {min: 1, max: 9, palette: ['1f8d49', '7a5900', 'd6bc74', 'edde8e',
  'e974ed', 'ffefc3', 'd4271e', 'db4d4f', '2532e4']}, 'Classes MapBiomas (agrupadas)', false);
Map.addLayer(transicao.clip(area), {min: 1, max: 5,
  palette: ['7f0000', 'ff4500', '1f8d49', 'ffd700', '7fffd4']},
  'Transições ' + PARAMS.anoBase + '–' + PARAMS.anoMapBiomas, false);
Map.addLayer(anoUrb.clip(area), {min: PARAMS.anoBase, max: PARAMS.anoMapBiomas,
  palette: ['440154', '3b528b', '21918c', '5ec962', 'fde725']}, 'Ano de urbanização', false);
Map.addLayer(refMask.selfMask().clip(area), {palette: ['00ff00']}, 'Pixels de referência rural', false);
Map.addLayer(anomalia.clip(area), {
  min: PARAMS.metodoAnomalia === 'rural' ? -4 : -2.5,
  max: PARAMS.metodoAnomalia === 'rural' ? 12 : 2.5,
  palette: palTermica}, 'Anomalia térmica');
Map.addLayer(ee.Image().paint(area, 0, 2), {palette: '000000'}, 'Área de análise');

// ---------------------------------------------------------------------
// 7. EXPORTAÇÕES
// ---------------------------------------------------------------------
Export.image.toDrive({
  image: pilha.toFloat(), description: 'cascavel_pilha_anomalia_mapbiomas',
  folder: PARAMS.pastaDrive, region: area,
  crs: CRS, crsTransform: CRS_TRANSFORM, maxPixels: 1e10
});
Export.table.toDrive({collection: statsGrupo, description: 'stats_anomalia_por_classe',
  folder: PARAMS.pastaDrive, fileFormat: 'CSV'});
Export.table.toDrive({collection: statsTrans, description: 'stats_anomalia_por_transicao',
  folder: PARAMS.pastaDrive, fileFormat: 'CSV'});
Export.table.toDrive({collection: amostra, description: 'amostra_estratificada',
  folder: PARAMS.pastaDrive, fileFormat: 'CSV'});
