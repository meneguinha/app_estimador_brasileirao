# ⚽ Preditor de Público e Renda — Brasileirão Série A

Aplicação web construída com **Streamlit** que usa Machine Learning para estimar o **público pagante** e a **renda bruta** de partidas do Brasileirão Série A, com base em dados históricos de jogos.

---

## 📸 Visão Geral

O usuário escolhe o time mandante, o visitante, o dia da semana, o horário e o ano desejado. O modelo retorna uma previsão instantânea, comparando o resultado com a média histórica do mandante.

---

## 🧠 Como Funciona

O modelo é treinado com dados reais de partidas (arquivo `.xlsx`) usando um **Gradient Boosting Regressor** com as seguintes features:

| Feature | Descrição |
|---|---|
| `Mandante_enc` | Time mandante (codificado) |
| `Visitante_enc` | Time visitante (codificado) |
| `Hora_enc` | Horário do jogo (codificado) |
| `DiaSemana` | Dia da semana (0 = Segunda … 6 = Domingo) |
| `Ano` | Ano da partida |
| `MandanteAvgPub` | Média histórica de público do mandante |
| `MandanteAvgRen` | Média histórica de renda do mandante |

Dois modelos independentes são treinados: um para **público pagante** e outro para **renda bruta**.

O modelo treinado é salvo em disco (`modelo_predicao.pkl`) e reutilizado nas execuções seguintes para evitar retreinamento desnecessário.

---

## 🗂️ Estrutura do Projeto

```
.
├── app.py                      # Aplicação principal (Streamlit)
├── consolidado_formatado.xlsx  # Base de dados histórica (necessária para treino)
├── modelo_predicao.pkl         # Modelo serializado (gerado automaticamente)
└── README.md
```

---

## ⚙️ Pré-requisitos

- Python 3.8+
- pip

---

## 🚀 Instalação e Execução

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Instale as dependências:**
   ```bash
   pip install streamlit pandas openpyxl scikit-learn numpy
   ```

3. **Adicione o arquivo de dados:**

   Coloque o arquivo `consolidado_formatado.xlsx` na raiz do projeto (mesma pasta que `app.py`).

4. **Inicie a aplicação:**
   ```bash
   streamlit run app.py
   ```

5. Acesse no navegador: `http://localhost:8501`

> Na primeira execução, o modelo será treinado automaticamente e salvo em `modelo_predicao.pkl`. As execuções seguintes carregam o modelo do disco.

---

## 📊 Formato dos Dados (`consolidado_formatado.xlsx`)

O arquivo deve conter ao menos as seguintes colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Data` | Data (DD/MM/AAAA) | Data da partida |
| `Hora` | String | Horário do jogo (ex.: `16:00`) |
| `Mandante` | String | Nome do time mandante |
| `Visitante` | String | Nome do time visitante |
| `Pagante` | Inteiro | Público pagante |
| `Renda` | Float | Renda bruta da partida |
| `Ano` | Inteiro | Ano da temporada |

---

## 🖥️ Interface

A aplicação oferece os seguintes campos de entrada:

- 🏠 **Time Mandante** — seleção a partir dos times presentes nos dados
- ✈️ **Time Visitante** — seleção excluindo o mandante escolhido
- 📅 **Dia da Semana** — de Segunda-feira a Domingo
- 🕐 **Horário** — horários presentes na base de dados
- 📆 **Ano** — entre 2018 e 2035

Após clicar em **🔮 Prever**, a aplicação exibe:

- 👥 Público estimado (pagantes)
- 💰 Renda estimada (R$)
- 📈/📉 Comparação com a média histórica do mandante

---

## 🔧 Parâmetros do Modelo

```python
GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```

Para ajustar a performance, edite o dicionário `params` dentro da função `treinar()` em `app.py`.

---

## 📝 Licença

Este projeto é de uso livre para fins educacionais e de pesquisa.

---

## 🤝 Contribuições

Pull requests são bem-vindos! Para mudanças maiores, abra uma *issue* primeiro para discutir o que você gostaria de alterar.
