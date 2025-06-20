import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score

# Load data
with open("bootstrap/kiba_fold0.pkl", 'rb') as f:
    data = pickle.load(f)

pred = np.array(data['pred'])
label = np.array(data['label'])

# MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Concordance Index
def CI(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 排列組合成所有可能的 pairwise index
    i, j = np.triu_indices(len(y_true), k=1)
    dy = y_true[i] - y_true[j]
    dp = y_pred[i] - y_pred[j]
    valid = dy != 0
    dy = dy[valid]
    dp = dp[valid]
    concordant = ((dy * dp) > 0).sum()
    ties = ((dp == 0) & (dy != 0)).sum()

    return (concordant + 0.5 * ties) / len(dy) if len(dy) > 0 else 0


# Rm^2 metric
def rm2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # R0² = fit without intercept (i.e., scaled prediction: k * y_pred)
    k = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    y_pred_k = k * y_pred
    ss_res_k = np.sum((y_true - y_pred_k) ** 2)
    r2_0 = 1 - ss_res_k / ss_tot
    rm2 = r2 * (1 - np.sqrt(np.abs(r2 ** 2 - r2_0 ** 2)))
    
    return rm2

# Bootstrap
B = 1000
n = len(pred)

mse_vals = []
ci_vals = []
rm2_vals = []

for _ in tqdm(range(B), desc="Bootstrapping"):
    idx = np.random.choice(n, n, replace=True)
    mse_vals.append(mse(label[idx], pred[idx]))
    ci_vals.append(CI(label[idx], pred[idx]))
    rm2_vals.append(rm2(label[idx], pred[idx]))

# Convert to arrays
mse_vals = np.array(mse_vals)
ci_vals = np.array(ci_vals)
rm2_vals = np.array(rm2_vals)

# Results
print(f"MSE  Mean: {np.mean(mse_vals):.4f}, SE: {np.std(mse_vals, ddof=1):.4f}")
print(f"CI   Mean: {np.mean(ci_vals):.4f}, SE: {np.std(ci_vals, ddof=1):.4f}")
print(f"Rm^2 Mean: {np.mean(rm2_vals):.4f}, SE: {np.std(rm2_vals, ddof=1):.4f}")
