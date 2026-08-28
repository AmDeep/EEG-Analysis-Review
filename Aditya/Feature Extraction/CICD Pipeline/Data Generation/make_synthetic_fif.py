import numpy as np
import mne

sfreq = 500.0
ch_names = ["Fz", "Cz", "CPz", "Pz", "C3", "C4"]
info = mne.create_info(ch_names, sfreq, ch_types="eeg")

n_epochs = 8
tmin = -0.3
n_times = int((0.6 - tmin) * sfreq) + 1
rng = np.random.default_rng(0)
data = rng.normal(0, 5e-6, size=(n_epochs, len(ch_names), n_times))  # volts

events = np.column_stack([
    np.arange(n_epochs) * n_times,
    np.zeros(n_epochs, dtype=int),
    np.ones(n_epochs, dtype=int),
])

epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, verbose=False)
epochs.save("sub-99_task-synthetic-epo.fif", overwrite=True)
print("wrote sub-99_task-synthetic-epo.fif")
