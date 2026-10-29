"""Higienização e codificação das features (primeira configuração, 33 features).

Gera, em dados/processed/DADOS/:
  1. base_modelagem_mt_limpa.csv      ID + 33 features com os códigos originais + alvo
  2. base_modelagem_mt_numerica.csv   mesma base com codificação numérica determinística
  3. desenvolvimento_80_numerico.csv e teste_20_numerico.csv
     (a numérica separada pelos IDs de dados/splits/, sem refazer a divisão)

Regras: só os registros sem NU_NOTA_MT saem; treineiros, notas zero e outliers ficam;
ausências nas features NÃO são imputadas aqui (imputação e padronização ficam no
Pipeline, ajustadas no treino de cada dobra). A codificação usa mapas fixos do
dicionário do ENEM 2023, então nada é aprendido dos dados e não há vazamento.
Arquivos já existentes não são sobrescritos.

Uso:
    python src/higienizacao.py [--saida DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ALVO, CAMINHO_IDS_DESENVOLVIMENTO, CAMINHO_IDS_TESTE, CHAVE_ID,
                    COLUNAS_SAIDA_LIMPA, DIR_PROCESSED, FEATURES_MANTIDAS, FEATURES_REMOVIDAS,
                    TAMANHO_BLOCO, ler_csv_bruto, localizar_csv_bruto, verificar_sem_vazamento)

# --- Codificação determinística, baseada no dicionário do ENEM 2023 ----------------
# Categorias nominais: one-hot (a primeira categoria é a referência).
CATEGORIAS_NOMINAIS = {
    'TP_FAIXA_ETARIA': [str(i) for i in range(1, 21)],
    'TP_SEXO': ['F', 'M'],
    'TP_ESTADO_CIVIL': [str(i) for i in range(0, 5)],
    'TP_COR_RACA': [str(i) for i in range(0, 7)],
    'TP_ST_CONCLUSAO': [str(i) for i in range(1, 5)],
    'TP_ESCOLA': [str(i) for i in range(1, 4)],
    'TP_ENSINO': ['1', '2'],
    'Q003': list('ABCDEF'),
    'Q004': list('ABCDEF'),
}
MAPA_ESCOLARIDADE = {letra: nivel for nivel, letra in enumerate('ABCDEFG')}
# Ponto médio da faixa em salários mínimos; Q (> 20 SM) usa o piso 20.
MAPA_RENDA_SM = {
    'A': 0.0, 'B': 0.5, 'C': 1.25, 'D': 1.75, 'E': 2.25, 'F': 2.75,
    'G': 3.5, 'H': 4.5, 'I': 5.5, 'J': 6.5, 'K': 7.5, 'L': 8.5,
    'M': 9.5, 'N': 11.0, 'O': 13.5, 'P': 17.5, 'Q': 20.0,
}
MAPA_CONTAGEM = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}  # E = 4 ou mais (piso)
# Q007: A = não, B = 1-2 dias, C = 3-4 dias, D = 5 ou mais dias por semana.
# Os valores 0 a 3 são NÍVEIS ordinais da resposta, não a quantidade de dias.
MAPA_NIVEL_EMPREGADA = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
COLUNAS_CONTAGEM = ['Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016',
                    'Q017', 'Q019', 'Q022', 'Q024']
COLUNAS_BINARIAS = ['Q018', 'Q020', 'Q021', 'Q023', 'Q025']


def normalizar_codigo(serie):
    """Normaliza códigos como 1, 1.0 ou letras, sem interpretar o código como medida."""
    return serie.astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)


def codificar_categoria_nominal(saida, coluna, serie, categorias):
    valores = normalizar_codigo(serie)
    for categoria in categorias[1:]:
        saida[f'{coluna}_{categoria}'] = valores.eq(categoria).fillna(False).astype('int8')
    saida[f'{coluna}__AUSENTE'] = valores.isna().astype('int8')
    saida[f'{coluna}__OUTRO'] = (valores.notna() & ~valores.isin(categorias)).astype('int8')


def codificar_valor_ordenado(saida, coluna_saida, serie, mapa, codigos_validos):
    valores = normalizar_codigo(serie)
    saida[coluna_saida] = valores.map(mapa).astype('float32')
    saida[f'{coluna_saida}__AUSENTE'] = valores.isna().astype('int8')
    saida[f'{coluna_saida}__OUTRO'] = (valores.notna() & ~valores.isin(codigos_validos)).astype('int8')


def transformar_bloco_para_numerico(bloco):
    """Devolve ID + preditoras numéricas + alvo; nenhum rótulo textual é exportado."""
    saida = {}
    id_numerico = pd.to_numeric(bloco[CHAVE_ID], errors='coerce')
    if id_numerico.isna().any():
        raise ValueError('Há NU_INSCRICAO ausente ou não numérica; verifique a base antes de exportar.')
    saida[CHAVE_ID] = id_numerico.astype('int64')

    for coluna, categorias in CATEGORIAS_NOMINAIS.items():
        codificar_categoria_nominal(saida, coluna, bloco[coluna], categorias)

    # Escolaridade: A-G são níveis ordenados; H ('não sei') é sinalizado à parte, não é um nível.
    for coluna in ('Q001', 'Q002'):
        valores = normalizar_codigo(bloco[coluna])
        saida[f'{coluna}_NIVEL_ESCOLARIDADE'] = valores.map(MAPA_ESCOLARIDADE).astype('float32')
        saida[f'{coluna}_DESCONHECE'] = valores.eq('H').fillna(False).astype('int8')
        saida[f'{coluna}__AUSENTE'] = valores.isna().astype('int8')
        saida[f'{coluna}__OUTRO'] = (valores.notna() & ~valores.isin(list('ABCDEFGH'))).astype('int8')

    # Treineiro é preservado como 0/1; nenhuma linha é filtrada por ele.
    codificar_valor_ordenado(saida, 'IN_TREINEIRO', bloco['IN_TREINEIRO'], {'0': 0, '1': 1}, ['0', '1'])

    moradores_txt = normalizar_codigo(bloco['Q005'])
    moradores = pd.to_numeric(moradores_txt, errors='coerce')
    valido = moradores.between(1, 20)
    saida['Q005_MORADORES'] = moradores.where(valido).astype('float32')
    saida['Q005_MORADORES__AUSENTE'] = moradores_txt.isna().astype('int8')
    saida['Q005_MORADORES__FORA_DOMINIO'] = (moradores_txt.notna() & ~valido).astype('int8')

    codificar_valor_ordenado(saida, 'Q006_RENDA_SM_ESTIMADA', bloco['Q006'], MAPA_RENDA_SM, list(MAPA_RENDA_SM))
    codificar_valor_ordenado(saida, 'Q007_DIAS_DOMESTICA_ESTIMADOS', bloco['Q007'],
                             MAPA_NIVEL_EMPREGADA, list(MAPA_NIVEL_EMPREGADA))
    for coluna in COLUNAS_CONTAGEM:
        codificar_valor_ordenado(saida, f'{coluna}_CONTAGEM_ESTIMADA', bloco[coluna], MAPA_CONTAGEM, list(MAPA_CONTAGEM))
    for coluna in COLUNAS_BINARIAS:
        codificar_valor_ordenado(saida, f'{coluna}_BINARIA', bloco[coluna], {'A': 0, 'B': 1}, ['A', 'B'])

    saida[ALVO] = pd.to_numeric(bloco[ALVO], errors='coerce').astype('float32')
    saida = pd.DataFrame(saida, index=bloco.index)
    if not all(pd.api.types.is_numeric_dtype(t) for t in saida.dtypes):
        raise TypeError('A saída numérica contém uma coluna não numérica.')
    return saida


def _preparar_saida(caminho):
    """Devolve o caminho temporário; None se o arquivo final já existe (não sobrescreve)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        print(f'O arquivo já existe; não foi sobrescrito: {caminho}')
        return None
    tmp = caminho.with_name(caminho.name + '.tmp')
    if tmp.exists():
        raise FileExistsError(f'Arquivo temporário já existe: {tmp}')
    return tmp


def _ler_blocos_brutos(caminho_base):
    return ler_csv_bruto(caminho_base, usecols=COLUNAS_SAIDA_LIMPA, chunksize=TAMANHO_BLOCO,
                         dtype={c: 'string' for c in [CHAVE_ID, *FEATURES_MANTIDAS]})


def exportar_bases(caminho_base, pasta_saida):
    """Uma única leitura da base bruta grava a versão limpa (códigos originais) e a numérica."""
    destino_limpa = pasta_saida / 'base_modelagem_mt_limpa.csv'
    destino_num = pasta_saida / 'base_modelagem_mt_numerica.csv'
    tmp_limpa, tmp_num = _preparar_saida(destino_limpa), _preparar_saida(destino_num)
    if tmp_limpa is None and tmp_num is None:
        return destino_num
    if (tmp_limpa is None) != (tmp_num is None):
        raise FileExistsError('Só uma das bases existe; confira os arquivos antes de continuar.')

    lidos = removidos = mantidos = treineiros = zeros = 0
    primeiro = True
    try:
        for bloco in _ler_blocos_brutos(caminho_base):
            lidos += len(bloco)
            alvo = pd.to_numeric(bloco[ALVO], errors='coerce')
            valido = alvo.notna()
            removidos += int((~valido).sum())
            bloco = bloco.loc[valido].copy()
            mantidos += len(bloco)
            treineiros += int(normalizar_codigo(bloco['IN_TREINEIRO']).eq('1').sum())
            zeros += int(pd.to_numeric(bloco[ALVO], errors='coerce').eq(0).sum())
            modo = dict(mode='w' if primeiro else 'a', header=primeiro, index=False)
            bloco[COLUNAS_SAIDA_LIMPA].to_csv(tmp_limpa, **modo)
            numerico = transformar_bloco_para_numerico(bloco)
            numerico.to_csv(tmp_num, **modo)
            primeiro = False
    except Exception:
        for tmp in (tmp_limpa, tmp_num):
            if tmp is not None and tmp.exists():
                tmp.unlink()
        raise
    tmp_limpa.replace(destino_limpa)
    tmp_num.replace(destino_num)

    print(f'Registros lidos: {lidos:,} | removidos por alvo ausente: {removidos:,} | mantidos: {mantidos:,}')
    print(f'Treineiros mantidos: {treineiros:,} | notas zero mantidas: {zeros:,}')
    print(f'Features em X: {len(FEATURES_MANTIDAS)} | candidatas fora de X: {len(FEATURES_REMOVIDAS)}')
    print(f'Colunas numéricas na saída (ID e alvo incluídos): {numerico.shape[1]}')
    print(f'Salvos: {destino_limpa.name} e {destino_num.name}')
    return destino_num


def separar_numerica_por_ids(pasta_saida):
    """Separa a base numérica em desenvolvimento e teste pelos IDs de dados/splits/."""
    origem = pasta_saida / 'base_modelagem_mt_numerica.csv'
    destino_dev = pasta_saida / 'desenvolvimento_80_numerico.csv'
    destino_teste = pasta_saida / 'teste_20_numerico.csv'
    if destino_dev.exists() and destino_teste.exists():
        print('Os dois CSVs particionados já existem; nenhum arquivo foi sobrescrito.')
        return
    if destino_dev.exists() or destino_teste.exists():
        raise FileExistsError('Só uma das saídas existe; confira antes de continuar (partição incompleta).')
    if not (CAMINHO_IDS_DESENVOLVIMENTO.exists() and CAMINHO_IDS_TESTE.exists()):
        raise FileNotFoundError("IDs não encontrados em dados/splits/. Execute 'python src/preparacao_dados.py'.")

    ids_dev = pd.read_csv(CAMINHO_IDS_DESENVOLVIMENTO, dtype={CHAVE_ID: 'int64'})[CHAVE_ID].to_numpy()
    ids_teste = pd.read_csv(CAMINHO_IDS_TESTE, dtype={CHAVE_ID: 'int64'})[CHAVE_ID].to_numpy()
    if np.intersect1d(ids_dev, ids_teste).size:
        raise ValueError('Há IDs repetidos entre desenvolvimento e teste.')

    tmp_dev = destino_dev.with_name(destino_dev.name + '.tmp')
    tmp_teste = destino_teste.with_name(destino_teste.name + '.tmp')
    if tmp_dev.exists() or tmp_teste.exists():
        raise FileExistsError('Há arquivo temporário de partição anterior; verifique antes de continuar.')

    n_dev = n_teste = n_base = 0
    primeiro_dev = primeiro_teste = True
    try:
        for bloco in pd.read_csv(origem, chunksize=TAMANHO_BLOCO):
            n_base += len(bloco)
            em_dev = bloco[CHAVE_ID].isin(ids_dev)
            em_teste = bloco[CHAVE_ID].isin(ids_teste)
            if (em_dev & em_teste).any():
                raise ValueError('Um mesmo registro caiu em desenvolvimento e teste.')
            if (~em_dev & ~em_teste).any():
                raise ValueError('Há registro numérico sem ID em nenhum split.')
            if em_dev.any():
                bloco.loc[em_dev].to_csv(tmp_dev, mode='w' if primeiro_dev else 'a', header=primeiro_dev, index=False)
                n_dev += int(em_dev.sum())
                primeiro_dev = False
            if em_teste.any():
                bloco.loc[em_teste].to_csv(tmp_teste, mode='w' if primeiro_teste else 'a', header=primeiro_teste, index=False)
                n_teste += int(em_teste.sum())
                primeiro_teste = False
        if n_dev != len(ids_dev) or n_teste != len(ids_teste):
            raise ValueError(f'Contagens não conferem: dev {n_dev:,}/{len(ids_dev):,}; teste {n_teste:,}/{len(ids_teste):,}.')
        tmp_dev.replace(destino_dev)
        tmp_teste.replace(destino_teste)
    except Exception:
        for tmp in (tmp_dev, tmp_teste):
            if tmp.exists():
                tmp.unlink()
        raise
    print(f'Desenvolvimento: {n_dev:,} ({n_dev / n_base:.2%}) -> {destino_dev.name}')
    print(f'Teste isolado:   {n_teste:,} ({n_teste / n_base:.2%}) -> {destino_teste.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', type=Path, default=DIR_PROCESSED, help='pasta de saída (padrão: dados/processed/DADOS)')
    args = ap.parse_args()

    verificar_sem_vazamento()
    caminho_base = localizar_csv_bruto()
    print(f'Base bruta: {caminho_base}')
    exportar_bases(caminho_base, args.saida)
    separar_numerica_por_ids(args.saida)


if __name__ == '__main__':
    main()
