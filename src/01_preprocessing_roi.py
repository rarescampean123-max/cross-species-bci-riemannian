import os
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signals, lowcut=0.01, highcut=0.1, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signals, axis=-1)


def apply_roi_mask(full_brain_signals, roi_indices):
    return full_brain_signals[:, roi_indices, :]


def generate_calibrated_species_data(species="human", n_trials=60, n_total_channels=32, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)

    if species == "human":
        roi_indices = [4, 5, 12, 13]
        anatomical_mixing = rng.normal(loc=2.5, scale=0.8, size=(4, 4)) + np.eye(4) * 3.0
    elif species == "canine":
        n_total_channels = 16
        roi_indices = [1, 2, 7, 8]
        anatomical_mixing = rng.normal(loc=-1.8, scale=1.2, size=(4, 4)) + np.eye(4) * 1.5
    else:
        raise ValueError("Specia trebuie sa fie 'human' sau 'canine'.")

    labels = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
    rng.shuffle(labels)

    full_signals = rng.normal(0, 1.0, size=(n_trials, n_total_channels, n_samples))

    for i, label in enumerate(labels):
        latent_sources = rng.normal(0, 1.0, size=(4, n_samples))

        if label == 1:
            reward_wave = np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 1.8
            latent_sources[0] += reward_wave
            latent_sources[1] += 0.9 * reward_wave
            latent_sources[2] += 0.75 * reward_wave

        species_roi_signal = anatomical_mixing @ latent_sources
        full_signals[i, roi_indices, :] = species_roi_signal

    filtered_signals = bandpass_filter(full_signals, lowcut=0.02, highcut=0.25, fs=1.0)
    roi_signals = apply_roi_mask(filtered_signals, roi_indices)

    return roi_signals, labels, roi_indices


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    print("=== [ZIUA 1] RULARE PREPROCESARE & EXTRAGERE ROI ===")

    X_human, y_human, idx_human = generate_calibrated_species_data(species="human", n_trials=80, seed=42)
    print(f"[OM] Canale ROI: {idx_human} -> Dimensiune date: {X_human.shape}")

    X_canine, y_canine, idx_canine = generate_calibrated_species_data(species="canine", n_trials=60, seed=108)
    print(f"[CAINE] Canale ROI: {idx_canine} -> Dimensiune date: {X_canine.shape}")

    np.savez("../data/preprocessed_roi_data.npz",
             X_human=X_human, y_human=y_human,
             X_canine=X_canine, y_canine=y_canine)

    print("[SUCCES] Datele au fost salvate in 'data/preprocessed_roi_data.npz'.")