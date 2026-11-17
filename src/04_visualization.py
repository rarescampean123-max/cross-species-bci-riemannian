import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_riemann


def bandpass_filter(signals, lowcut=0.02, highcut=0.25, fs=1.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signals, axis=-1)


def compute_spd_covariances(signals):
    return Covariances(estimator='lwf').fit_transform(signals)


def euclidean_alignment(cov_matrices):
    r_bar = np.mean(cov_matrices, axis=0)
    r_inv_sqrt = fractional_matrix_power(r_bar, -0.5)
    return np.einsum('ij,njk,kl->nil', r_inv_sqrt, cov_matrices, r_inv_sqrt)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_cache = os.path.join(base_dir, 'data', 'raw', 'real_caudate_sample.npz')
    metrics_path = os.path.join(base_dir, 'results', 'h1_validation_metrics.json')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)['real_fmri_sample']

    data = np.load(raw_cache)
    x_h, y_h, x_d, y_d = data['X_h'], data['y_h'], data['X_d'], data['y_d']

    cov_h_raw = compute_spd_covariances(bandpass_filter(x_h))
    cov_d_raw = compute_spd_covariances(bandpass_filter(x_d))
    cov_h_ea = euclidean_alignment(cov_h_raw)
    cov_d_ea = euclidean_alignment(cov_d_raw)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

    ts_raw = TangentSpace(metric='riemann').fit(np.concatenate([cov_h_raw, cov_d_raw]))
    feat_raw = ts_raw.transform(np.concatenate([cov_h_raw, cov_d_raw]))
    pca_raw = PCA(n_components=2).fit_transform(feat_raw)

    ts_ea = TangentSpace(metric='riemann')
    ts_ea.reference_ = np.eye(4)
    feat_ea = ts_ea.transform(np.concatenate([cov_h_ea, cov_d_ea]))
    pca_ea = PCA(n_components=2).fit_transform(feat_ea)

    n_h = len(y_h)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    axes[0].scatter(pca_raw[:n_h][y_h == 0, 0], pca_raw[:n_h][y_h == 0, 1], c='#1f77b4', marker='o', label='Om - Neutru', alpha=0.8)
    axes[0].scatter(pca_raw[:n_h][y_h == 1, 0], pca_raw[:n_h][y_h == 1, 1], c='#1f77b4', marker='^', s=70, label='Om - Recompensa', alpha=0.9)
    axes[0].scatter(pca_raw[n_h:][y_d == 0, 0], pca_raw[n_h:][y_d == 0, 1], c='#d62728', marker='o', label='Caine - Neutru', alpha=0.8)
    axes[0].scatter(pca_raw[n_h:][y_d == 1, 0], pca_raw[n_h:][y_d == 1, 1], c='#d62728', marker='^', s=70, label='Caine - Recompensa', alpha=0.9)
    axes[0].set_title(f"Inainte de Aliniere (RAW)\nDistanta inter-specie = {metrics['dist_inter_raw']}")
    axes[0].set_xlabel("PC1 Spatiu Tangent")
    axes[0].set_ylabel("PC2 Spatiu Tangent")
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].scatter(pca_ea[:n_h][y_h == 0, 0], pca_ea[:n_h][y_h == 0, 1], c='#1f77b4', marker='o', label='Om - Neutru', alpha=0.8)
    axes[1].scatter(pca_ea[:n_h][y_h == 1, 0], pca_ea[:n_h][y_h == 1, 1], c='#1f77b4', marker='^', s=70, label='Om - Recompensa', alpha=0.9)
    axes[1].scatter(pca_ea[n_h:][y_d == 0, 0], pca_ea[n_h:][y_d == 0, 1], c='#d62728', marker='o', label='Caine - Neutru', alpha=0.8)
    axes[1].scatter(pca_ea[n_h:][y_d == 1, 0], pca_ea[n_h:][y_d == 1, 1], c='#d62728', marker='^', s=70, label='Caine - Recompensa', alpha=0.9)
    axes[1].set_title(f"Dupa Aliniere Euclidiana in I (EA)\nDistanta inter-specie = {metrics['dist_inter_ea']} (-{metrics['distance_reduction_pct']}%)")
    axes[1].set_xlabel("PC1 Spatiu Tangent (centrat in I)")
    axes[1].set_ylabel("PC2 Spatiu Tangent (centrat in I)")
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig1_riemannian_alignment_pca.png'))
    plt.close()

    delta_h = mean_riemann(cov_h_ea[y_h == 1]) - mean_riemann(cov_h_ea[y_h == 0])
    delta_d = mean_riemann(cov_d_ea[y_d == 1]) - mean_riemann(cov_d_ea[y_d == 0])
    rois = ['Caudat Stg', 'Caudat Dr', 'NAcc Stg', 'NAcc Dr']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    vmax = max(np.abs(delta_h).max(), np.abs(delta_d).max())

    axes[0].imshow(delta_h, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[0].set_title("Om: Amprenta Recompensei (Delta C)")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(rois, rotation=25)
    axes[0].set_yticklabels(rois)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{delta_h[i, j]:.2f}", ha='center', va='center', fontsize=8)

    im1 = axes[1].imshow(delta_d, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Caine: Amprenta Recompensei (Delta C)\nCorelatie Om-Caine r = {metrics['topological_correlation_r']} (p = {metrics['permutation_p_value']})")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(rois, rotation=25)
    axes[1].set_yticklabels(rois)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{delta_d[i, j]:.2f}", ha='center', va='center', fontsize=8)

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label='Variatie Conectivitate (EA)')
    plt.savefig(os.path.join(results_dir, 'fig2_caudate_connectivity_h1.png'), bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    labels_bar = ['Inter-Specie\n(RAW)', 'Inter-Specie\n(EA in I)', 'Intra-Specie\n(Om Rew vs Neu)']
    values_bar = [metrics['dist_inter_raw'], metrics['dist_inter_ea'], metrics['dist_intra_human']]
    colors_bar = ['#d62728', '#2ca02c', '#1f77b4']
    bars = axes[0].bar(labels_bar, values_bar, color=colors_bar, width=0.55)
    axes[0].set_ylabel("Distanta Geodezica Riemanniana")
    axes[0].set_title(f"Criteriu Geometric H1: Raport = {metrics['conservation_index']} (< 1.0)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

    perm_dist = np.array(metrics['perm_null_distribution'])
    axes[1].hist(perm_dist, bins=25, color='#7f7f7f', alpha=0.75, edgecolor='black', label='Distributie Nula (1000 permutari)')
    axes[1].axvline(metrics['topological_correlation_r'], color='#d62728', linestyle='--', linewidth=2.2,
                    label=f"r observat = {metrics['topological_correlation_r']} (p = {metrics['permutation_p_value']})")
    axes[1].set_xlabel("Corelatia Pearson (r)")
    axes[1].set_ylabel("Frecventa")
    axes[1].set_title("Testul de Permutare pentru Conservarea H1")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig3_geodesic_distances_permutation.png'))
    plt.close()

    print("[SUCCES] Cele 3 figuri au fost generate in folderul 'results/'.")


if __name__ == "__main__":
    main()