# Estimativa de Frequência Cardíaca via Wi-Fi CSI: Avaliação de Modelos Leves

Investigação de modelos leves de aprendizado de máquina para estimativa sem contato da frequência cardíaca utilizando informações do estado do canal Wi-Fi (Channel State Information — CSI). Este projeto avalia sistematicamente se arquiteturas mais simples — Regressão Linear e Perceptrons Multicamadas (MLP) — podem servir como alternativas viáveis a redes neurais recorrentes computacionalmente custosas (Bi-GRU, Bi-LSTM), analisando também o impacto do tamanho da janela de observação na acurácia da estimativa.

## Contexto

Este projeto foi desenvolvido como trabalho final da disciplina **Tópicos em Sistemas de Computação**, disciplina de pós-graduação ofertada na **Universidade Federal de Goiás (UFG)**, no semestre 2026/2.

A disciplina aborda aplicações emergentes de aprendizado profundo e processamento de sinais em sensoriamento sem fio. Este trabalho explora o problema específico do monitoramento cardíaco sem contato, que explora perturbações sutis causadas pelo movimento da parede torácica no canal Wi-Fi entre um transmissor e um receptor.

## Fundamentação e Inspiração

Este projeto foi diretamente inspirado por e construído sobre dois trabalhos-chave na literatura de Wi-Fi Sensing:

- **Wi-Cardio** (SVR 2026): Propõe uma arquitetura GRU Bidirecional com atenção temporal para estimativa de frequência cardíaca a partir de dados CSI, alcançando MAE reportado de 1,22 bpm no dataset eHealth CSI. O repositório anônimo original que acompanha a submissão do artigo serviu como ponto de partida deste projeto, fornecendo o pipeline de pré-processamento base, as implementações de referência GRU/LSTM e a documentação do dataset.

- **Pulse-Fi** (IEEE Sensors Journal, 2023): Emprega uma LSTM Bidirecional para a mesma tarefa, reportando MAE de 1,84 bpm.

Ambas as abordagens dependem de arquiteturas recorrentes com milhões de parâmetros. A pergunta de pesquisa central deste projeto é: **quanta acurácia deve ser sacrificada ao substituir esses modelos complexos por alternativas mais simples e eficientes em parâmetros?**

## Evolução do Projeto

O desenvolvimento seguiu uma progressão metódica, desde a estabilização da infraestrutura até a experimentação arquitetural:

1. **Código-base inicial** (10/Out): Clonado a partir do repositório anônimo do Wi-Cardio, que incluía scripts de treinamento para Bi-GRU e Bi-LSTM, um pipeline de pré-processamento de sinais e documentação do dataset eHealth CSI. O código original era fixo para uma janela de observação de 20 segundos e possuía problemas de escalabilidade de memória.

2. **Reformulação do pré-processamento** (11/Out): Refatoração do pipeline de pré-processamento para suportar tamanhos de janela dinâmicos via parâmetros de linha de comando (`--window_sec`, `--step_sec`), adição de compatibilidade com ground truth em CSV do Polar H10, e resolução de problemas de estouro de memória (OOM) através de divisão por sujeito com coleta de lixo explícita.

3. **Validação em nuvem** (12/Out): Construção de notebook no Google Colab para validação do pipeline de pré-processamento em hardware de nuvem antes de executar experimentos de longa duração localmente.

4. **Otimização de I/O** (27/Out): Substituição da busca recursiva em disco por arquivo de ground truth por um índice hash em memória (`build_gt_index`), reduzindo o tempo de pré-processamento de horas para minutos por configuração de janela. Criação do script `preprocess_loop.py` para processamento automatizado em lote nos cinco tamanhos de janela.

5. **Linha de base linear** (28/Out): Implementação de regressão linear por Mínimos Quadrados Ordinários operando sobre a entrada CSI achatada (*flatten*). Esta abordagem demonstrou falha catastrófica pela maldição da dimensionalidade: o MAE degradou de 23,47 bpm (janela de 10s, 21.248 features) para 167,95 bpm (janela de 30s, 64.000 features).

6. **Linha de base MLP** (29/Out): Implementação de MLP profundo (3 camadas ocultas, ativações GELU, LayerNorm, Dropout) sobre a entrada achatada com perda de Huber. Estabilizou o desempenho em torno de 13,1-13,8 bpm de MAE, porém exigindo de 5 a 16 milhões de parâmetros dependendo do tamanho da janela.

7. **Formulações temporais com atenção** (21/Nov): Identificação do gargalo fundamental na abordagem de achatamento — destruição da estrutura temporal e explosão de parâmetros proporcional ao número de timesteps. Implementação de duas variantes temporais:
   - **Linear Temporal**: Projeção linear por subportadora com pesos compartilhados no tempo, seguida de pooling temporal ponderado por atenção. Fixo em 16.577 parâmetros independente do tamanho da janela.
   - **MLP Temporal**: Backbone MLP compartilhado aplicado independentemente por timestep, com mecanismo de atenção temporal aprendido para agregação ponderada. Fixo em 199.426 parâmetros — aproximadamente 100x menos que o MLP achatado equivalente.

## Dataset

Este projeto utiliza o dataset **eHealth CSI**, um dataset público coletado no Laboratório Midiacom (Universidade Federal Fluminense). O dataset compreende gravações CSI Wi-Fi de 86 participantes realizando diversas atividades, adquiridas usando:

- **Transmissor/Receptor**: Dispositivos Raspberry Pi com adaptadores Wi-Fi comerciais
- **Ground Truth**: Cinta torácica Polar H10 (referência de frequência cardíaca com precisão de ECG)
- **Resolução CSI**: 256 subportadoras ativas
- **Atividades**: 18 posturas e padrões de movimento distintos (sentado, em pé, deitado, caminhando, etc.)

O pipeline de pré-processamento aplica:
- Filtro passa-banda Butterworth de 3ª ordem (0,8-3,0 Hz, cobrindo 48-180 bpm)
- Suavização Savitzky-Golay (janela=15, ordem polinomial=3)
- Janelas deslizantes com passo de 0,5s em cinco tamanhos: 10s, 15s, 20s, 25s, 30s

Para acesso ao dataset e licenciamento, consulte [`docs/README_eHealth.md`](docs/README_eHealth.md).

## Resultados

### Tabela Resumo

| Modelo | Tratamento Temporal | Janela | Parâmetros | MAE de Teste (bpm) | RMSE de Teste (bpm) | R² de Teste |
|:---|:---|:---:|---:|---:|---:|---:|
| Wi-Cardio (literatura) | Bi-GRU + Atenção | ~20s | > 3.000.000 | **1,22** | **3,22** | **0,9642** |
| Pulse-Fi (literatura) | Bi-LSTM | ~20s | > 4.000.000 | **1,84** | **3,25** | **0,9635** |
| **MLP Temporal** | Projeção + Atenção | **20s** | **199.426** | **12,58** | **16,00** | **0,0839** |
| MLP Temporal | Projeção + Atenção | 10s | 199.426 | 13,22 | 16,61 | 0,0068 |
| MLP Temporal | Projeção + Atenção | 15s | 199.426 | 13,23 | 16,39 | 0,0385 |
| MLP Temporal | Projeção + Atenção | 25s | 199.426 | 14,10 | 17,35 | -0,0747 |
| MLP Temporal | Projeção + Atenção | 30s | 199.426 | 13,67 | 16,66 | 0,0109 |
| MLP (flatten) | Achatamento direto | 20s | > 10.000.000 | 13,71 | 16,42 | 0,0355 |
| Linear Temporal | Projeção + Pooling | 10s | 16.577 | 13,85 | 17,23 | -0,0696 |
| Linear (flatten) | OLS | 10s | 21.248 | 23,47 | 30,62 | -2,3772 |
| Linear (flatten) | OLS | 30s | 64.000 | 167,95 | 245,94 | -214,44 |

### Principais Conclusões

- **A formulação temporal é crítica**: O achatamento destrói a estrutura temporal e infla a contagem de parâmetros proporcionalmente ao número de timesteps. Modelos temporais com atenção alcançam acurácia comparável ou superior com ordens de magnitude menos parâmetros.
- **Janela de 20 segundos é ótima para o MLP Temporal**: Essa janela captura aproximadamente 20-30 ciclos cardíacos completos, evitando deriva de fase e acúmulo de ruído de baixa frequência proveniente da respiração e micro-movimentos posturais.
- **Piso de acurácia em torno de 12-14 bpm para modelos estáticos**: Sem propagação de estado recorrente (realimentação de estado oculto a cada timestep), modelos simples não conseguem rastrear a continuidade da forma de onda cardíaca periódica imersa em ruído, explicando a diferença de 10x em relação às arquiteturas Bi-GRU/Bi-LSTM.
- **Regressão linear colapsa catastroficamente quando achatada**: A maldição da dimensionalidade faz o erro crescer de 23 bpm para 168 bpm conforme a janela aumenta, mas a formulação temporal estabiliza-o em 13-17 bpm.

## Estrutura do Projeto

```
.
├── src/
│   ├── preprocess.py               # Pipeline de pré-processamento de sinais CSI
│   ├── preprocess_loop.py           # Pré-processamento em lote para todos os tamanhos de janela
│   ├── train_gru.py                 # Bi-GRU de referência (Wi-Cardio)
│   ├── train_lstm.py                # Bi-LSTM de referência (Pulse-Fi)
│   ├── train_lr.py                  # Regressão linear (flatten)
│   ├── train_lr_loop.py             # Executor em lote da regressão linear
│   ├── train_mlp.py                 # MLP (flatten)
│   ├── train_mlp_loop.py            # Executor em lote do MLP
│   ├── train_linear_temporal.py     # Linear temporal com atenção
│   ├── train_linear_temporal_loop.py
│   ├── train_mlp_temporal.py        # MLP temporal com atenção
│   ├── train_mlp_temporal_loop.py
│   ├── plot_results.py              # Visualização de resultados
│   └── plot_preprocessing.py        # Visualização do pré-processamento
├── data/                            # Janelas deslizantes pré-processadas (gitignored)
├── output/                          # Métricas e históricos de treinamento
│   ├── linear_{10,15,20,25,30}s/
│   ├── linear_temporal_{10..30}s/
│   ├── mlp_{10,15,20,25,30}s/
│   └── mlp_temporal_{10..30}s/
├── docs/                            # Documentação do dataset
├── notebooks/                       # Notebooks de validação no Colab
└── pyproject.toml                   # Metadados e dependências do projeto
```

## Configuração e Uso

### Pré-requisitos

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Dataset eHealth CSI (solicitar acesso em `csi@midiacom.uff.br`)

### Instalação

```bash
git clone git@github.com:ottojvn/projeto-topicos.git
cd projeto-topicos
uv sync
```

### Pré-processamento

```bash
# Tamanho de janela único
uv run python src/preprocess.py --window_sec 20 --step_sec 0.5

# Todos os tamanhos de janela (10s, 15s, 20s, 25s, 30s)
uv run python src/preprocess_loop.py
```

### Treinamento

```bash
# Modelo único, janela única
uv run python src/train_mlp_temporal.py --window 20

# Varredura completa em todos os tamanhos de janela
uv run python src/train_mlp_temporal_loop.py
```

Scripts de treinamento disponíveis: `train_lr.py`, `train_mlp.py`, `train_linear_temporal.py`, `train_mlp_temporal.py`, `train_gru.py`, `train_lstm.py` (cada um com variante `_loop.py` correspondente).

## Referências

1. **Wi-Cardio**: Heart Rate Monitoring Using Wi-Fi CSI Data and a GRU Architecture. SVR 2026.
2. **Pulse-Fi**: Contact-Free Pulse Rate Estimation Using Wi-Fi CSI. IEEE Sensors Journal, 2023.
3. **eHealth CSI**: A Wi-Fi CSI Dataset of Human Activities. IEEE, 2023. DOI: [10.1109/JSEN.2023.3289435](https://ieeexplore.ieee.org/abstract/document/10177905)

## Licença

Este projeto é disponibilizado para fins acadêmicos e de pesquisa. O dataset eHealth CSI está sujeito ao seu próprio [Acordo de Uso de Dados](docs/README_eHealth.md).
