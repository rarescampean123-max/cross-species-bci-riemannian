import numpy as np
from scipy.linalg import fractional_matrix_power
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def compute_spd_covariances(signals, estimator='lwf'):
    cov_estimator = Covariances(estimator=estimator)
    return cov_estimator.fit_transform(signals)


def euclidean_alignment(cov_matrices):
    r_bar = np.mean(cov_matrices, axis=0)
    r_inv_sqrt = fractional_matrix_power(r_bar, -0.5)
    aligned_covs = np.einsum('ij,njk,kl->nil', r_inv_sqrt, cov_matrices, r_inv_sqrt)
    return aligned_covs, r_bar


def extract_tangent_features(cov_matrices, reference_point=None):
    if reference_point is not None:
        ts = TangentSpace(metric='riemann')
        ts.reference_ = reference_point
        return ts.transform(cov_matrices)
    ts = TangentSpace(metric='riemann')
    return ts.fit_transform(cov_matrices)


if __name__ == "__main__":
    print("MATRICI DE COVARIANTA si ALINIERE EUCLIDIANA IN I")

    data = np.load("../data/preprocessed_roi_data.npz")
    X_human, y_human = data["X_human"], data["y_human"]
    X_canine, y_canine = data["X_canine"], data["y_canine"]

    cov_human_raw = compute_spd_covariances(X_human, estimator='lwf')
    cov_canine_raw = compute_spd_covariances(X_canine, estimator='lwf')
    print(f"[RAW] Forma matrici covarianta Om: {cov_human_raw.shape} | Caine: {cov_canine_raw.shape}")

    cov_human_ea, r_bar_human = euclidean_alignment(cov_human_raw)
    cov_canine_ea, r_bar_canine = euclidean_alignment(cov_canine_raw)

    mean_after_human = np.mean(cov_human_ea, axis=0)
    mean_after_canine = np.mean(cov_canine_ea, axis=0)
    identity_matrix = np.eye(cov_human_ea.shape[1])

    err_human = np.linalg.norm(mean_after_human - identity_matrix)
    err_canine = np.linalg.norm(mean_after_canine - identity_matrix)

    print(f"[EA] Distanta medie fata de Matricea Identitate I (Om): {err_human:.2e}")
    print(f"[EA] Distanta medie fata de Matricea Identitate I (Caine): {err_canine:.2e}")

    features_human_raw = extract_tangent_features(cov_human_raw)
    features_canine_raw = extract_tangent_features(cov_canine_raw, reference_point=np.mean(cov_human_raw, axis=0))

    features_human_ea = extract_tangent_features(cov_human_ea, reference_point=identity_matrix)
    features_canine_ea = extract_tangent_features(cov_canine_ea, reference_point=identity_matrix)

    print(f"[TANGENT SPACE] Vectori trasaturi extrasi -> Om: {features_human_ea.shape} | Caine: {features_canine_ea.shape}")

    np.savez("../data/riemannian_features.npz",
             cov_human_raw=cov_human_raw,
             cov_canine_raw=cov_canine_raw,
             cov_human_ea=cov_human_ea,
             cov_canine_ea=cov_canine_ea,
             features_human_raw=features_human_raw,
             features_canine_raw=features_canine_raw,
             features_human_ea=features_human_ea,
             features_canine_ea=features_canine_ea,
             y_human=y_human,
             y_canine=y_canine)

    print("[SUCCES] Matricile aliniate si vectorii din spatiul tangent au fost salvati in 'data/riemannian_features.npz'.")