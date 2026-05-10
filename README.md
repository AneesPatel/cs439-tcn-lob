# CS439 Final Project - TCN for Limit Order Book Mid-Price Prediction

---

This is my final project for CS439. The basic idea is to predict whether the mid-price of a stock is going to go up, down, or stay flat in the next few timesteps, using a Temporal Convolutional Network (TCN). I compared it against a plain LSTM and some simpler baselines to see if the TCN actually does better.

Since I didn't have access to real order book data, I wrote a synthetic data generator that produces realistic-looking LOB snapshots with an actual learnable signal baked in (an AR(1) order imbalance process that drives price movements). The whole thing runs from a single Python file, so no datasets to download, no setup beyond installing a few packages.



## How to run it

```bash
pip install torch scikit-learn numpy matplotlib seaborn shap
python tcn_lob.py
```

It will take a long time to run on a normal laptop. When it finishes you'll have five plots saved in the same folder.



## Files

- `tcn_lob.py` - the whole pipeline in one script
- `run.log` - output from the run I used for the paper results
- `requirements.txt` - pip packages needed to run it
- `ablation_f1.png`, `confusion_matrix.png`, `pca_embeddings.png`, `shap_importance.png`, `training_curve.png` - graphs/images from the paper
