# Autoformer 

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

Time series forecasting is a critical demand for real applications. Enlighted by the classic time series analysis and stochastic process theory, we propose the Autoformer as a general series forecasting model [[paper](https://arxiv.org/abs/2106.13008)]. **Autoformer goes beyond the Transformer family and achieves the series-wise connection for the first time.**


## Autoformer vs. Transformers

**1. Deep decomposition architecture**

We renovate the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

<p align="center">
<img src=".\utils\pic\Autoformer.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of Autoformer.
</p>

**2. Series-wise Auto-Correlation mechanism**

Inspired by the stochastic process theory, we design the Auto-Correlation mechanism, which can discover period-based dependencies and aggregate the information at the series level. This empowers the model with inherent log-linear complexity. This series-wise connection contrasts clearly from the previous self-attention family.

<p align="center">
<img src=".\utils\pic\Auto-Correlation.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Auto-Correlation mechansim.
</p>


### Get Started
1. Install Python >= 3.7
2. Install requirements.txt
3. Run Example.ipynb


### Special-designed implementation

- **Speedup Auto-Correlation:** We built the Auto-Correlation mechanism as a batch-normalization-style block to make it more memory-access friendly. See the [paper](https://arxiv.org/abs/2106.13008) for details.

- **Without the position embedding:** Since the series-wise connection will inherently keep the sequential information, Autoformer does not need the position embedding, which is different from Transformers.


## Available models

- [x] Autoformer
- [x] Informer
- [x] Transformer

## Versions
- Ver 1.2
  - Loss functions: MSE, MAE, MAPE, SMAPE
  - Optimizers: Adam, SGD, Adagrad
- Ver 1.1
  - Fix bugs
  - Store results in txt file
- Ver 1.0
  - Basic implemetation - Models: Transformer, Informer, AutoFormer

## Requirements

- tqdm==4.61.0
- torch==1.7.0
- scikit-learn==0.24.2
- pandas==1.1.3
- numpy==1.20.2
- matplotlib==3.4.2
