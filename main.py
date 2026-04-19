import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os

# ==========================================
# [Setup] Create output directory
# ==========================================
output_dir = "oversampling_paradox_results"
if os.path.exists(output_dir): shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# --- 1. Core Model Definitions ---
def u_base(i, peak): return 1.0 - 0.25 * np.abs(i - peak)
def softmax(u, beta):
    ex = np.exp(beta * u)
    return ex / np.sum(ex)

def get_group_probs(v2, beta, peak):
    options = np.array([1, 2, 3, 4, 5])
    u_true = u_base(options, peak)
    u_sontaku = u_base(options, 4)
    return softmax((1 - v2) * u_true + v2 * u_sontaku, beta)

# --- 2. Validation J: The Oversampling Paradox ---
print("Running Validation J: Oversampling Paradox Simulation...")
v2_target = 0.5
beta = 5.0
N_total = 10000

# 1. 真のマイノリティ(10%)とマジョリティ(90%)を生成
n_minority = int(N_total * 0.10)
n_majority = N_total - n_minority

# 各グループの忖度後の分布
p_min = get_group_probs(v2_target, beta, 1)
p_maj = get_group_probs(v2_target, beta, 3)

# 各個人が選んだ回答をシミュレート
resp_min = np.random.choice([1,2,3,4,5], size=n_minority, p=p_min)
resp_maj = np.random.choice([1,2,3,4,5], size=n_majority, p=p_maj)

# 全データ結合
df = pd.DataFrame({
    'Group': ['Minority']*n_minority + ['Majority']*n_majority,
    'Response': np.concatenate([resp_min, resp_maj])
})

# オーバーサンプリング前の評価1の構成を確認
obs_rating_1 = df[df['Response'] == 1]
min_in_1 = len(obs_rating_1[obs_rating_1['Group'] == 'Minority'])
maj_in_1 = len(obs_rating_1[obs_rating_1['Group'] == 'Majority'])

# 3. 本来「評価1」であるべきマイノリティ(1000人)のうち、何人が「評価3や4」に逃げたか？
fugitives = len(df[(df['Group'] == 'Minority') & (df['Response'] != 1)])

# グラフ：オーバーサンプリングの「質の劣化」を可視化
plt.figure(figsize=(10, 6))
labels = ['True Minority in Rating 1', 'Majority Noise in Rating 1', 'Missing Signal (Fugitives)']
values = [min_in_1, maj_in_1, fugitives]
colors = ['green', 'red', 'lightgray']

plt.bar(labels, values, color=colors, edgecolor='black')
plt.title(f"Fig J: The Oversampling Paradox (v2={v2_target})", fontsize=14)
plt.ylabel("Number of Respondents")
plt.text(2, fugitives/2, f"These {fugitives} people\nare hidden in\nRating 3 & 4!", ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(output_dir, 'fig_J_oversampling_paradox.png'), dpi=300)
plt.close()

# CSV出力
df_summary = pd.DataFrame({
    'Metric': labels,
    'Count': values
})
df_summary.to_csv(os.path.join(output_dir, 'data_J_oversampling_summary.csv'), index=False)

shutil.make_archive("oversampling_paradox_archive", 'zip', output_dir)
print("✅ Done. 'oversampling_paradox_archive.zip' is ready.")
