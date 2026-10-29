from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
_RAW = BASE_DIR / 'dados' / 'raw'
# Aceita o CSV solto em dados/raw/ ou dentro da pasta do zip do Inep descompactado.
caminho_base = next(
    (c for c in [_RAW / 'MICRODADOS_ENEM_2023.csv',
                 _RAW / 'DADOS' / 'MICRODADOS_ENEM_2023.csv',
                 *_RAW.glob('*/DADOS/MICRODADOS_ENEM_2023.csv')] if c.exists()),
    _RAW / 'MICRODADOS_ENEM_2023.csv',
)

pasta_splits = BASE_DIR / 'dados' / 'splits'
pasta_processed = BASE_DIR / 'dados' / 'processed'
pasta_docs = BASE_DIR / 'docs'

COLUNAS_PROIBIDAS_LEAKAGE = [
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
    'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',
    'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT',
    'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT',
    'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
    'TP_STATUS_REDACAO',
    'NU_ANO'
]

FEATURES_DEMOGRAFICAS = [
    'TP_FAIXA_ETARIA',
    'TP_SEXO',
    'TP_ESTADO_CIVIL',
    'TP_COR_RACA',
    'TP_NACIONALIDADE',
    'TP_ST_CONCLUSAO',
    'TP_ANO_CONCLUIU',
    'TP_ESCOLA',
    'TP_ENSINO',
    'IN_TREINEIRO'
]

FEATURES_PROVA_PREVIA = [
    'TP_LINGUA',
    'SG_UF_PROVA'
]

FEATURES_SOCIOECONOMICAS = [f'Q{str(i).zfill(3)}' for i in range(1, 26)]

COLUNAS_PREDITORAS = FEATURES_DEMOGRAFICAS + FEATURES_PROVA_PREVIA + FEATURES_SOCIOECONOMICAS
ALVO = 'NU_NOTA_MT'
CHAVE_ID = 'NU_INSCRICAO'

COLUNAS_LEITURA = [CHAVE_ID, ALVO] + COLUNAS_PREDITORAS


def carregar_e_filtrar_dados():
    print("-> Carregando base bruta com seleção de colunas legítimas...")
    
    intersecao_proibidas = set(COLUNAS_PREDITORAS).intersection(set(COLUNAS_PROIBIDAS_LEAKAGE))
    assert len(intersecao_proibidas) == 0, f"ERRO DE VAZAMENTO: Colunas proibidas detectadas em X: {intersecao_proibidas}"
    
    df = pd.read_csv(
        caminho_base,
        sep=';',
        encoding='latin1',
        usecols=COLUNAS_LEITURA
    )
    print(f"   Base carregada com {len(df):,} linhas e {len(df.columns)} colunas.")
    
    df = df.dropna(subset=[ALVO]).copy()
    print(f"   Participantes presentes com nota de Matemática: {len(df):,}")
    
    return df


def carregar_identificadores_splits():
    print("-> Carregando identificadores das partições (splits)...")
    arquivo_dev = pasta_splits / 'desenvolvimento_ids.csv'
    arquivo_teste = pasta_splits / 'teste_ids.csv'
    
    if not arquivo_dev.exists() or not arquivo_teste.exists():
        raise FileNotFoundError(
            "Arquivos de splits não encontrados em dados/splits/. "
            "Execute 'python src/preparacao_dados.py' primeiro."
        )
    
    dev_ids = pd.read_csv(arquivo_dev)[CHAVE_ID]
    teste_ids = pd.read_csv(arquivo_teste)[CHAVE_ID]
    
    print(f"   IDs carregados: Desenvolvimento = {len(dev_ids):,}, Teste = {len(teste_ids):,}")
    return set(dev_ids), set(teste_ids)


def higienizar_perfil_sociodemografico(df_dev, df_teste):
    print("-> Higienizando perfil sociodemográfico e tratando ausências...")
    
    # TP_ESTADO_CIVIL: 0 = Não informado
    df_dev['TP_ESTADO_CIVIL'] = df_dev['TP_ESTADO_CIVIL'].fillna(0).astype(int)
    df_teste['TP_ESTADO_CIVIL'] = df_teste['TP_ESTADO_CIVIL'].fillna(0).astype(int)

    # TP_COR_RACA: 0 = Não declarado
    df_dev['TP_COR_RACA'] = df_dev['TP_COR_RACA'].fillna(0).astype(int)
    df_teste['TP_COR_RACA'] = df_teste['TP_COR_RACA'].fillna(0).astype(int)

    # TP_ESCOLA: 1 = Não respondeu
    df_dev['TP_ESCOLA'] = df_dev['TP_ESCOLA'].fillna(1).astype(int)
    df_teste['TP_ESCOLA'] = df_teste['TP_ESCOLA'].fillna(1).astype(int)

    # TP_ENSINO: 0 = Não se aplica / Já concluiu
    df_dev['TP_ENSINO'] = df_dev['TP_ENSINO'].fillna(0).astype(int)
    df_teste['TP_ENSINO'] = df_teste['TP_ENSINO'].fillna(0).astype(int)

    # TP_ANO_CONCLUIU: 0 = Não informado / Não concluído
    df_dev['TP_ANO_CONCLUIU'] = df_dev['TP_ANO_CONCLUIU'].fillna(0).astype(int)
    df_teste['TP_ANO_CONCLUIU'] = df_teste['TP_ANO_CONCLUIU'].fillna(0).astype(int)

    # Questionário socioeconômico: preenchimento constante para evitar vazamento em validação cruzada
    for col in FEATURES_SOCIOECONOMICAS:
        df_dev[col] = df_dev[col].fillna('Nao_Informado').astype(str)
        df_teste[col] = df_teste[col].fillna('Nao_Informado').astype(str)

    colunas_categoricas_str = ['TP_SEXO', 'SG_UF_PROVA']
    for col in colunas_categoricas_str:
        df_dev[col] = df_dev[col].astype(str)
        df_teste[col] = df_teste[col].astype(str)

    print("   Tratamento de ausências e tipagens concluído.")
    return df_dev, df_teste


def exportar_dados_higienizados(df_dev, df_teste):
    pasta_processed.mkdir(parents=True, exist_ok=True)
    caminho_dev = pasta_processed / 'desenvolvimento_higienizado.csv'
    caminho_teste = pasta_processed / 'teste_higienizado.csv'
    
    # Reduzindo o peso da base na memória
    for col in df_dev.columns:
        if df_dev[col].dtype == 'int64':
            df_dev[col] = pd.to_numeric(df_dev[col], downcast='integer')
            df_teste[col] = pd.to_numeric(df_teste[col], downcast='integer')
        elif df_dev[col].dtype == 'object':
            df_dev[col] = df_dev[col].astype('category')
            df_teste[col] = df_teste[col].astype('category')

    print("-> Exportando datasets higienizados em dados/processed/...")
    df_dev.to_csv(caminho_dev, index=False)
    df_teste.to_csv(caminho_teste, index=False)
    
    print(f"   Partição de desenvolvimento salva: {caminho_dev}")
    print(f"   Partição de teste salva: {caminho_teste}")


def main():
    print("=" * 80)
    print("INICIANDO HIGIENIZAÇÃO E TRATAMENTO DE DATA LEAKAGE")
    print("=" * 80)
    
    df = carregar_e_filtrar_dados()
    dev_ids, teste_ids = carregar_identificadores_splits()
    
    print("-> Particionando a base higienizada...")
    df_dev = df[df[CHAVE_ID].isin(dev_ids)].copy()
    df_teste = df[df[CHAVE_ID].isin(teste_ids)].copy()
    
    print(f"   Desenvolvimento: {len(df_dev):,} registros")
    print(f"   Teste: {len(df_teste):,} registros")
    
    df_dev, df_teste = higienizar_perfil_sociodemografico(df_dev, df_teste)
    
    exportar_dados_higienizados(df_dev, df_teste)
    
    print("=" * 80)
    print("HIGIENIZAÇÃO E TRATAMENTO CONCLUÍDOS COM SUCESSO!")
    print("=" * 80)


if __name__ == '__main__':
    main()
