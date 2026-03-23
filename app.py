"""
App Streamlit — Preditor de Público e Renda (Brasileirão)
==========================================================
Como rodar:
    pip install streamlit pandas openpyxl scikit-learn numpy
    streamlit run app.py

O arquivo consolidado_formatado.xlsx deve estar na mesma pasta.
O modelo é treinado automaticamente na primeira execução e salvo
em modelo_predicao.pkl para as execuções seguintes.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Configurações ────────────────────────────────────────────────────────────

ARQUIVO_DADOS  = "consolidado_formatado.xlsx"
ARQUIVO_MODELO = "modelo_predicao.pkl"

DIAS_SEMANA = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo",
}
DIAS_SEMANA_INV = {v: k for k, v in DIAS_SEMANA.items()}

# ─── Funções do modelo ────────────────────────────────────────────────────────

def carregar_dados(caminho):
    df = pd.read_excel(caminho)
    df["Hora"] = df["Hora"].fillna("Não informado").astype(str)
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
    df["DiaSemana"] = df["Data"].dt.dayofweek
    return df


def construir_features(df, le_team, le_hora, avg_pub, avg_ren, fallback_pub, fallback_ren):
    df = df.copy()
    df["MandanteAvgPub"] = df["Mandante"].map(avg_pub).fillna(fallback_pub)
    df["MandanteAvgRen"] = df["Mandante"].map(avg_ren).fillna(fallback_ren)
    df["Mandante_enc"]   = le_team.transform(df["Mandante"])
    df["Visitante_enc"]  = le_team.transform(df["Visitante"])
    df["Hora_enc"]       = le_hora.transform(df["Hora"])
    return df[["Mandante_enc", "Visitante_enc", "Hora_enc",
               "DiaSemana", "Ano", "MandanteAvgPub", "MandanteAvgRen"]]


def treinar(df):
    all_teams = sorted(set(df["Mandante"]) | set(df["Visitante"]))
    hora_cats = sorted(df["Hora"].unique())
    le_team = LabelEncoder().fit(all_teams)
    le_hora = LabelEncoder().fit(hora_cats)
    avg_pub = df.groupby("Mandante")["Pagante"].mean().to_dict()
    avg_ren = df.groupby("Mandante")["Renda"].mean().to_dict()
    fallback_pub = df["Pagante"].mean()
    fallback_ren = df["Renda"].mean()

    X = construir_features(df, le_team, le_hora, avg_pub, avg_ren, fallback_pub, fallback_ren)
    params = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, random_state=42)
    mod_pag  = GradientBoostingRegressor(**params).fit(X, df["Pagante"])
    mod_rend = GradientBoostingRegressor(**params).fit(X, df["Renda"])

    pacote = dict(
        mod_pag=mod_pag, mod_rend=mod_rend,
        le_team=le_team, le_hora=le_hora,
        avg_pub=avg_pub, avg_ren=avg_ren,
        fallback_pub=fallback_pub, fallback_ren=fallback_ren,
        all_teams=all_teams, hora_cats=hora_cats,
        n_jogos=len(df), ano_min=int(df["Ano"].min()), ano_max=int(df["Ano"].max()),
    )
    with open(ARQUIVO_MODELO, "wb") as f:
        pickle.dump(pacote, f)
    return pacote


def prever(pacote, mandante, visitante, hora, dia_semana, ano):
    row = pd.DataFrame([{
        "Mandante_enc":   pacote["le_team"].transform([mandante])[0],
        "Visitante_enc":  pacote["le_team"].transform([visitante])[0],
        "Hora_enc":       pacote["le_hora"].transform([hora])[0],
        "DiaSemana":      dia_semana,
        "Ano":            ano,
        "MandanteAvgPub": pacote["avg_pub"].get(mandante, pacote["fallback_pub"]),
        "MandanteAvgRen": pacote["avg_ren"].get(mandante, pacote["fallback_ren"]),
    }])
    colunas = ["Mandante_enc", "Visitante_enc", "Hora_enc",
               "DiaSemana", "Ano", "MandanteAvgPub", "MandanteAvgRen"]
    pub  = max(0, int(pacote["mod_pag"].predict(row[colunas])[0]))
    rend = max(0, pacote["mod_rend"].predict(row[colunas])[0])
    return pub, rend


# ─── Cache do modelo ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Treinando modelo, aguarde...")
def carregar_ou_treinar():
    if os.path.exists(ARQUIVO_MODELO):
        with open(ARQUIVO_MODELO, "rb") as f:
            return pickle.load(f)
    df = carregar_dados(ARQUIVO_DADOS)
    return treinar(df)


# ─── Interface ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Preditor Brasileirão", page_icon="⚽", layout="centered")

st.title("⚽ Preditor de Público e Renda")
st.caption("Brasileirão Série A — modelo treinado com dados históricos")

# Carrega modelo
if not os.path.exists(ARQUIVO_DADOS) and not os.path.exists(ARQUIVO_MODELO):
    st.error(f"Arquivo `{ARQUIVO_DADOS}` não encontrado. Coloque-o na mesma pasta que `app.py`.")
    st.stop()

pacote = carregar_ou_treinar()

st.caption(
    f"Modelo treinado com **{pacote['n_jogos']:,} jogos** "
    f"({pacote['ano_min']}–{pacote['ano_max']})"
)

st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    mandante = st.selectbox("🏠 Time mandante", pacote["all_teams"])

with col2:
    visitantes = [t for t in pacote["all_teams"] if t != mandante]
    visitante  = st.selectbox("✈️ Time visitante", visitantes)

col3, col4, col5 = st.columns(3)

with col3:
    dia_nome = st.selectbox("📅 Dia da semana", list(DIAS_SEMANA.values()))

with col4:
    hora = st.selectbox("🕐 Horário", pacote["hora_cats"])

with col5:
    ano = st.number_input("📆 Ano", min_value=2018, max_value=2035,
                          value=datetime.now().year, step=1)

st.divider()

# ── Botão e resultado ─────────────────────────────────────────────────────────
if st.button("🔮 Prever", use_container_width=True, type="primary"):
    dia_num = DIAS_SEMANA_INV[dia_nome]
    pub, rend = prever(pacote, mandante, visitante, hora, dia_num, ano)

    st.subheader(f"{mandante} × {visitante}")
    st.caption(f"{dia_nome} às {hora} — {ano}")

    m1, m2 = st.columns(2)
    m1.metric("👥 Público estimado", f"{pub:,.0f} pagantes")
    m2.metric("💰 Renda estimada",   f"R$ {rend:,.2f}")

    # Contexto histórico
    avg_pub_hist = pacote["avg_pub"].get(mandante, pacote["fallback_pub"])
    avg_ren_hist = pacote["avg_ren"].get(mandante, pacote["fallback_ren"])
    delta_pub = pub - avg_pub_hist
    delta_ren = rend - avg_ren_hist

    st.caption(
        f"Média histórica do {mandante} como mandante: "
        f"**{avg_pub_hist:,.0f} pagantes** · **R$ {avg_ren_hist:,.2f}**"
    )

    if delta_pub >= 0:
        st.success(f"📈 Previsão {delta_pub:,.0f} pagantes acima da média histórica do mandante")
    else:
        st.warning(f"📉 Previsão {abs(delta_pub):,.0f} pagantes abaixo da média histórica do mandante")
