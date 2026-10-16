#!/usr/bin/env python3
"""Generate fake EEG/stimuli (no human or real pain data) for smoke testing."""
import argparse
from pathlib import Path
import mne
import numpy as np
import pandas as pd


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('demo_input'))
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    sfreq = 256
    names = ['F3','F4','Fz','FCz','C3','C4','Cz','Pz','O1','O2','VEOG']
    types = ['eeg']*10 + ['eog']
    n_seconds = 105
    t = np.arange(n_seconds*sfreq)/sfreq
    x = np.vstack([14e-6 * np.sin(2*np.pi*(9.5 + i*.05)*t + i*.3)
                   + 4e-6*np.sin(2*np.pi*22*t + i)
                   + rng.normal(0, 6e-6, t.shape) for i in range(10)])
    eye = rng.normal(0,4e-6,t.shape)
    # Inject a few ocular transients into both VEOG and anterior EEG.
    for onset in [11, 29, 45, 64, 88]:
        pulse = np.exp(-0.5*((t-onset)/.07)**2)*120e-6
        eye += pulse
        x[0] += pulse*.3
        x[1] += pulse*.3
    rows=[]
    for i, onset in enumerate(np.arange(5.0,99,5.5)):
        intensity = [2,4,6,8][i%4]
        pain = float(np.clip(1.2 + .8*intensity+rng.normal(0,.6), 0, 10))
        rows.append({'onset':onset,'duration':0,'trial_type':'laser',
                     'trial_id':f'trial-{i+1:04d}',
                     'stimulus_intensity': intensity,
                     'pain_rating': round(pain,2)})
        response = ((-1.0*np.exp(-.5*((t-onset-.25)/.055)**2)
                     + 1.5*np.exp(-.5*((t-onset-.42)/.07)**2))
                     *intensity*1.1e-6)
        x[6] += response
        x[4] += response*.55
        x[5] += response*.55
    # Deliberately contaminate one epoch so QC demonstrates a rejection.
    idx = int(55.3*sfreq)
    x[4,idx:idx+5] += 400e-6
    info=mne.create_info(names, sfreq, ch_types=types)
    raw=mne.io.RawArray(np.vstack([x,eye]), info, verbose='ERROR')
    raw.set_montage('standard_1020', on_missing='ignore', verbose='ERROR')
    raw.save(args.out/'synthetic_continuous_raw.fif', overwrite=True, verbose='ERROR')
    pd.DataFrame(rows).to_csv(args.out/'synthetic_events.tsv', sep='\t', index=False)
    print('SYNTHETIC ONLY — saved:',args.out)


if __name__ == '__main__':
    main()
