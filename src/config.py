"""Caminhos e constantes compartilhados pelos scripts de src/.

Concentra o que antes estava repetido nos notebooks: localização dos arquivos,
semente, alvo, lista de features mantidas/removidas e colunas proibidas (vazamento).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_RAW = BASE_DIR / 'dados' / 'raw'
DIR_PROCESSED = BASE_DIR / 'dados' / 'processed' / 'DADOS'
DIR_SPLITS = BASE_DIR / 'dados' / 'splits'
DIR_DOCS = BASE_DIR / 'docs'
DIR_RESULTADOS = BASE_DIR / 'resultados'

SEMENTE = 42
TAMANHO_BLOCO = 100_000
ALVO = 'NU_NOTA_MT'
CHAVE_ID = 'NU_INSCRICAO'

CAMINHO_IDS_DESENVOLVIMENTO = DIR_SPLITS / 'desenvolvimento_ids.csv'
CAMINHO_IDS_TESTE = DIR_SPLITS / 'teste_ids.csv'
CAMINHO_BASE_LIMPA = DIR_PROCESSED / 'base_modelagem_mt_limpa.csv'
CAMINHO_BASE_NUMERICA = DIR_PROCESSED / 'base_modelagem_mt_numerica.csv'
CAMINHO_DESENVOLVIMENTO_NUMERICO = DIR_PROCESSED / 'desenvolvimento_80_numerico.csv'
CAMINHO_TESTE_NUMERICO = DIR_PROCESSED / 'teste_20_numerico.csv'

# --- Features -----------------------------------------------------------------
# 46 candidatas registradas na tabela de decisões; 33 entram na primeira configuração.
FEATURES_PERFIL_ESCOLAR = [
    'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
    'TP_ST_CONCLUSAO', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO',
]
FEATURES_SOCIOECONOMICAS = [f'Q{i:03d}' for i in range(1, 26)]
FEATURES_MANTIDAS = FEATURES_PERFIL_ESCOLAR + FEATURES_SOCIOECONOMICAS

FEATURES_CANDIDATAS = [
    'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
    'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA',
    'TP_ENSINO', 'IN_TREINEIRO', 'CO_MUNICIPIO_ESC', 'NO_MUNICIPIO_ESC',
    'CO_UF_ESC', 'SG_UF_ESC', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC',
    'TP_SIT_FUNC_ESC', 'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA',
    'CO_UF_PROVA', 'SG_UF_PROVA', *FEATURES_SOCIOECONOMICAS,
]
FEATURES_REMOVIDAS = [c for c in FEATURES_CANDIDATAS if c not in FEATURES_MANTIDAS]

COLUNAS_SAIDA_LIMPA = [CHAVE_ID, *FEATURES_MANTIDAS, ALVO]
COLUNAS_DIAGNOSTICO = [CHAVE_ID, *FEATURES_CANDIDATAS, ALVO]

# Colunas que nunca podem entrar em X (vazamento ou sem informação).
COLUNAS_PROIBIDAS_LEAKAGE = [
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
    'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',
    'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT',
    'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT',
    'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
    'TP_STATUS_REDACAO', 'NU_ANO', ALVO, CHAVE_ID,
]


def verificar_sem_vazamento(colunas_x=FEATURES_MANTIDAS):
    """Interrompe a execução se X contiver coluna proibida, o alvo ou o identificador."""
    proibidas = set(colunas_x) & set(COLUNAS_PROIBIDAS_LEAKAGE)
    assert not proibidas, f'ERRO DE VAZAMENTO: colunas proibidas em X: {sorted(proibidas)}'


def localizar_csv_bruto():
    """Aceita o CSV em dados/raw/, em dados/raw/DADOS/ ou numa pasta de zip descompactado."""
    candidatos = [
        DIR_RAW / 'MICRODADOS_ENEM_2023.csv',
        DIR_RAW / 'DADOS' / 'MICRODADOS_ENEM_2023.csv',
        *sorted(DIR_RAW.glob('*/DADOS/MICRODADOS_ENEM_2023.csv')),
    ]
    for caminho in candidatos:
        if caminho.exists():
            return caminho
    raise FileNotFoundError(
        'MICRODADOS_ENEM_2023.csv não encontrado em dados/raw/. '
        'Veja as instruções de obtenção no README.md.'
    )


def ler_csv_bruto(caminho, **kwargs):
    """Leitura padrão do CSV do Inep (separador ';', codificação latin1)."""
    import pandas as pd
    return pd.read_csv(caminho, sep=';', encoding='latin1', **kwargs)
