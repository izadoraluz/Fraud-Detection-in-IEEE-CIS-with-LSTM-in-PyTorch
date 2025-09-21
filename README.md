# Credit Card Fraud Detection com LSTM (TensorFlow/Keras)

## **👤 Integrante**

* [Izadora Luz](https://www.linkedin.com/in/izadoraluz-rsn/)

## **👨‍🏫 Professores**

* [Cesar Almiñana](https://www.linkedin.com/in/ccalminana/) — Professor orientador
* [Crishna Irion](https://www.linkedin.com/in/crishna-irion-7b5aa311/) — Professora de redes neurais


## **📝 Descrição**

Este projeto implementa um pipeline completo para **detecção de fraudes** em transações de cartão de crédito usando **LSTM** (TensorFlow/Keras). O dataset é o público **Credit Card Fraud Detection (Kaggle – MLG-ULB)**, com \~**284k** transações, sendo **492 fraudes** (≈ **0,17%**).
O fluxo inclui: **EDA**, **preparo de dados** (normalização, janelas temporais), **treinamento** de uma LSTM com **validação temporal**, **ajuste de limiar** por F1, e **avaliação** com **Accuracy, Precision, Recall, F1 e AUC-ROC**, além de curvas de aprendizado, ROC e matriz de confusão.

> ⚠️ Observação importante: como o dataset não traz um ID de cliente/cartão, as sequências foram criadas na **linha do tempo global**. Em cenários reais, recomenda-se criar **sequências por entidade** (cartão/cliente) e respeitar fronteiras por ID nos splits para liberar o real potencial das RNNs.

## **🎯 Objetivos**

1. Explorar e entender o dataset (distribuições, correlações, outliers).
2. Preparar dados para uma **rede LSTM** (normalização e janelas temporais).
3. **Treinar** e **validar** o modelo com corte temporal.
4. **Avaliar** o desempenho e inspecionar **overfitting/underfitting** por curvas de aprendizado.
5. Documentar **decisões, limitações** e **próximos passos**.


## **⚙️ Funcionalidades implementadas**

### **EDA**

* Checagem de nulos (nenhum encontrado).
* Estatísticas descritivas e distribuição de classes (forte desbalanceamento).
* Histogramas/boxplots de `Amount` (inclui `log1p`).
* Correlações (ponto-biserial com `Class` + heatmap entre features PCA).

### **Preparo de dados**

* **Split temporal**: Train/Val/Test separados por tempo (evita vazamento futuro→passado).
* **StandardScaler** (fit no treino; aplicação em val/test).
* **Sequências deslizantes** com `LOOKBACK=10` passos.
* **Undersampling** leve no treino (\~10:1) + **class weights** para lidar com o desbalanceamento.

### **Modelo LSTM**

* Arquitetura enxuta (\~38k parâmetros): `LSTM(64, return_sequences) → BN → Dropout → LSTM(32) → Dropout → Dense(16) → Sigmoid`.
* **Callbacks**: `EarlyStopping` (monitor AUC), `ReduceLROnPlateau`.
* Otimização: `Adam` (lr inicial 1e-3).

### **Avaliação**

* **Métricas**: Accuracy, Precision, Recall, F1, **AUC-ROC**.
* **Tuning de limiar** na validação para **F1**.
* **Curvas**: Loss/AUC (aprendizado), ROC (teste) e **Matriz de Confusão**.


## **📁 Estrutura de pastas**

```
.
├── fraud_lstm_pipeline.py        # script principal (EDA → Preparo → LSTM → Avaliação)
├── README.md
```


## **💻 Tecnologias Utilizadas**

* **Python 3.x**
* **TensorFlow/Keras** (LSTM, métricas, callbacks)
* **Scikit-learn** (scalers, métricas, pesos de classe)
* **Pandas & NumPy** (manipulação de dados)
* **Matplotlib & Seaborn** (visualização)


## **🚀 Como rodar localmente**

1. Instale dependências (ideal usar venv/conda):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

2. Execute:

```bash
python fraud_lstm_pipeline.py
```

3. Veja saídas no terminal e gráficos na pasta `figs/`.

> Dica: se disponível, use GPU (TensorFlow com CUDA) para acelerar.


## **🧪 Experimentos e Resultados (resumo)**

**Distribuição de classes (dados do projeto):**

* Negativa: **99,827%** (284.315)
* Positiva (fraude): **0,173%** (492)

**Validação (melhor época):**

* **AUC(val)** ≈ **0,588**
* `val_precision/val_recall` muito baixos com threshold 0,5 (classe raríssima).

**Threshold escolhido pela validação (max F1):**

* **0,40** (F1(val) ≈ **0,007**)

**Teste (mantida distribuição real):**

* **Accuracy:** 0,9926
* **Precision:** 0,0028
* **Recall:** 0,0133
* **F1:** 0,0047
* **AUC-ROC:** 0,5186

**Curvas/Gráficos gerados:**

* `learning_curve_loss.png`, `learning_curve_auc.png`, `roc_curve_test.png`, `confusion_matrix_test.png` etc.


## **🔎 Diagnóstico e Lições**

* **Overfitting**: AUC treino (\~0,86) bem maior que AUC validação (\~0,59).
* **Sinal temporal fraco p/ LSTM**: sem ID de cartão/cliente, as janelas unem transações de entidades diferentes; a LSTM “vê” ruído global.
* **Desbalanceamento extremo**: mesmo com undersampling no treino e class weights, **val/test** no desequilíbrio real derrubam Precision/Recall; **AUC-PR** seria uma métrica ainda mais relevante.
* **Métricas com threshold fixo** no treino (0,5) pouco informativas; usar **PR-AUC**, **Recall\@K** e **threshold por custo** melhora a leitura.


## **🧭 Próximos passos**

1. **Sequências por entidade** (cartão/cliente) com splits respeitando fronteiras por ID e tempo.
2. **Baseline tabular forte** (LogReg / XGBoost / LightGBM) com **features derivadas** (ex.: contagens/estatísticas rolling por entidade, deltas de tempo, frequência por merchant).
3. **Focal Loss** ou apenas **class weights** (sem undersampling agressivo) para melhorar **calibração**.
4. **Métricas e decisão orientadas a custo**: **PR-AUC**, **Recall\@K**, ajuste de **threshold** visando Recall/precision-alvo.
5. **Regularização/arquitetura**: simplificar LSTM, ajustar Dropout/neurônios; considerar **GRU/1D-CNN**.
6. **Calibração de probabilidades** (Platt/Isotonic) após o treino.


## **⚙️ Parâmetros principais (no script)**

* `LOOKBACK=10` (tamanho da janela)
* `TEST_FRACTION=0.2` (corte temporal final para teste)
* `VAL_FRACTION=0.1` (parte final do treino para validação)
* `BATCH_SIZE=512`, `EPOCHS=30`, `LEARNING_RATE=1e-3`
* Undersampling no treino para \~**10:1** (neg\:pos) + `class_weight` balanceado
* EarlyStopping (monitor `val_auc`) e ReduceLROnPlateau
