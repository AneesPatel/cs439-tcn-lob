# CS439 Final Project - TCN for Limit Order Book Mid-Price Prediction

Anees Patel | Rutgers University | Spring 2026

---

This is my final project for CS439. The basic idea is to predict whether the mid-price of a stock is going to go up, down, or stay flat in the next few timesteps, using a Temporal Convolutional Network (TCN). I compared it against a plain LSTM and some simpler baselines to see if the TCN actually does better.

Since I didn't have access to real order book data, I wrote a synthetic data generator that produces realistic-looking LOB snapshots with an actual learnable signal baked in (an AR(1) order imbalance process that drives price movements). The whole thing runs from a single Python file - no datasets to download, no setup beyond installing a few packages.

---

## How to run it

```bash
pip install torch scikit-learn numpy matplotlib seaborn shap
python tcn_lob.py
```

It'll take around 15–25 minutes on a normal laptop. When it finishes you'll have five plots saved in the same folder.

---

## What I found

Honestly the most interesting result was that the tiny TCN (receptive field of just 7 timesteps) performed just as well as the huge one (receptive field of 127). All the TCN variants and the LSTM basically tied around 0.57 macro-F1. That makes sense in hindsight - the signal in my synthetic data decays fast, so a bigger context window doesn't actually help.

Everything beat random guessing (0.16 F1) and logistic regression (0.50 F1) by a decent margin, which at least confirms the models are picking up on something real.

---

## Files

- `tcn_lob.py` - the whole pipeline in one script
- `main.tex` - the paper writeup (NeurIPS format)
- `run.log` - output from the run I used for the paper results
- `ablation_f1.png`, `confusion_matrix.png`, `pca_embeddings.png`, `shap_importance.png`, `training_curve.png` - figures from the paper
