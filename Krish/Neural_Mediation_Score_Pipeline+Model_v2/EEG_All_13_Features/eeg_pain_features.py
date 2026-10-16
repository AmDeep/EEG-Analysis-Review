#!/usr/bin/env python3
"""Trial-level EEG baseline/response features for experimental pain studies.

Input: continuous EEG (.fif/.edf/.bdf/.vhdr/.set) and a stimulus event TSV/CSV
       with an onset column measured in seconds from recording start.
Output: accepted trial features.csv, all-trial qc.csv, run metadata.json,
        optional per-subject summary CSV.

Not a validated pain biomarker. See README.md before using real datasets.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal

LOGGER = logging.getLogger('eeg_pain')
BANDS = {'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
         'beta': (13.0, 30.0), 'gamma': (30.0, 45.0)}
# Fixed sensor groups/pairs: no nearest-electrode substitution.
REGIONS = {
    'sensorimotor': ['C3', 'C4'],
    'centro_midline': ['Cz'],
    'frontal_midline': ['Fz', 'FCz'],
    'gamma_central': ['C3', 'C4', 'Cz'],
    'all_eeg': [],
}
PAIRS = {'central_c3_c4': ('C3', 'C4'),
         'frontal_f3_f4': ('F3', 'F4'),
         'midline_fz_cz': ('Fz', 'Cz')}
EPS = 1e-12


@dataclass
class Config:
    pre_start: float = -2.0
    pre_end: float = -0.2
    post_start: float = 0.2
    post_end: float = 2.0
    highpass_hz: float = 0.3
    lowpass_hz: float = 45.0
    mains_hz: float = 60.0
    reject_ptp_uv: float = 150.0
    flat_ptp_uv: float = 0.5
    eog_reject_ptp_uv: float = 150.0
    min_eeg_channels: int = 4
    erp_baseline_start: float = -0.2
    erp_baseline_end: float = -0.02
    n2_start: float = 0.18
    n2_end: float = 0.35
    p2_start: float = 0.30
    p2_end: float = 0.60
    erp_lowpass_hz: float = 15.0
    n2_p2_channel: str = 'Cz'
    envelope_trim_s: float = 0.20
    use_ica: bool = True
    ica_seed: int = 17
    max_ica_eog_components: int = 4
    temporal_start: float = 0.15
    temporal_end: float = 0.75


def name_key(x: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', x).upper()


def label_column_names(event_columns: list[str]) -> list[str]:
    # Keep metadata, but ensure no collisions with generated feature names.
    blocked = {'onset', 'duration', 'sample'}
    return [x for x in event_columns if x not in blocked]


def load_raw(path: Path):
    ext = path.suffix.lower()
    if ext == '.fif':
        raw = mne.io.read_raw_fif(path, preload=True, verbose='ERROR')
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
    elif ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose='ERROR')
    elif ext == '.set':
        # Continuous EEGLAB recordings only. Pre-epoched .set files require
        # a different adapter and should not be treated as continuous.
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    else:
        raise ValueError(f'Unsupported EEG format: {ext}')
    if len(raw) == 0:
        raise ValueError('No samples in input recording')
    return raw


def eog_channel_detection(raw):
    """Only mark explicitly named ocular channels; never guess via ICA."""
    changes = {}
    for n, kind in zip(raw.ch_names, raw.get_channel_types()):
        k = name_key(n)
        if kind == 'eeg' and (k.startswith(('EOG', 'HEOG', 'VEOG')) or
                               k in {'HEO', 'VEO', 'LEOG', 'REOG'}):
            changes[n] = 'eog'
    if changes:
        raw.set_channel_types(changes, verbose='ERROR')
    return [n for n, kind in zip(raw.ch_names, raw.get_channel_types())
            if kind == 'eog']


def find_bad_eeg_channels(raw, manual_names):
    eeg_names = [n for n, kind in zip(raw.ch_names, raw.get_channel_types())
                 if kind == 'eeg']
    manual_map = {name_key(n): n for n in eeg_names}
    manual = [manual_map[name_key(n)] for n in manual_names
              if name_key(n) in manual_map]
    if len(eeg_names) < 4:
        raise ValueError('Need at least 4 EEG channels for this pipeline.')
    dat = raw.get_data(picks=eeg_names)  # volts
    # Flag flat/noisy channels using their standard deviation relative to peers.
    # Automated bad-channel estimates require manual review for a study.
    std_uv = np.std(dat, axis=1) * 1e6
    median_uv = float(np.median(std_uv[np.isfinite(std_uv)]))
    auto = []
    reasons = {}
    for name, std in zip(eeg_names, std_uv):
        why = []
        if not np.isfinite(std) or std < 0.15:
            why.append('near_flat_or_nonfinite')
        if median_uv > 0 and std > 8.0 * median_uv and std > 40:
            why.append('extreme_variance')
        if why:
            auto.append(name)
            reasons[name] = why
    bads = list(dict.fromkeys(list(raw.info['bads']) + manual + auto))
    raw.info['bads'] = bads
    report = {'auto_bads': auto, 'manual_bads': manual,
              'existing_bads': [n for n in bads if n not in manual and n not in auto],
              'bad_reasons': reasons, 'channel_std_uv': dict(zip(eeg_names, map(float, std_uv)))}
    return report


def preprocess(raw, cfg: Config, manual_bad_names):
    eog_names = eog_channel_detection(raw)
    qc = find_bad_eeg_channels(raw, manual_bad_names)
    good_eeg = [n for n, typ in zip(raw.ch_names, raw.get_channel_types())
                if typ == 'eeg' and n not in raw.info['bads']]
    if len(good_eeg) < cfg.min_eeg_channels:
        raise ValueError(f'Only {len(good_eeg)} good EEG channels remain.')
    nyquist = raw.info['sfreq'] / 2
    if nyquist <= cfg.lowpass_hz + 1:
        raise ValueError(f'Sampling rate {raw.info["sfreq"]} Hz too low for {cfg.lowpass_hz} Hz feature band.')
    # Notch before the final low-pass only when mains noise is within Nyquist.
    if 0 < cfg.mains_hz < nyquist - 1:
        raw.notch_filter(freqs=[cfg.mains_hz], picks='eeg', verbose='ERROR')
    raw.filter(cfg.highpass_hz, cfg.lowpass_hz, picks='eeg',
               fir_design='firwin', skip_by_annotation='edge', verbose='ERROR')
    qc['eog_channels'] = eog_names
    qc['ica_attempted'] = False
    qc['ica_removed'] = []
    if cfg.use_ica and eog_names and len(good_eeg) >= 6:
        # Fit ICA on a 1 Hz high-pass COPY, apply solution to 0.3 Hz original.
        try:
            ica_input = raw.copy().filter(l_freq=1.0, h_freq=None,
                                          picks='eeg', verbose='ERROR')
            ica = mne.preprocessing.ICA(n_components=0.99, method='fastica',
                                       random_state=cfg.ica_seed,
                                       max_iter=400, verbose='ERROR')
            ica.fit(ica_input, picks='eeg', decim=max(1, int(raw.info['sfreq'] / 200)),
                    reject={'eeg': 200e-6}, verbose='ERROR')
            eog_inds, _ = ica.find_bads_eog(ica_input, ch_name=eog_names[0],
                                            verbose='ERROR')
            ica.exclude = eog_inds[:cfg.max_ica_eog_components]
            if ica.exclude:
                ica.apply(raw, verbose='ERROR')
            qc['ica_attempted'] = True
            qc['ica_removed'] = list(map(int, ica.exclude))
        except Exception as exc:
            warnings.warn(f'ICA could not be completed: {exc}. Continuing without ICA.')
            qc['ica_error'] = str(exc)
    else:
        qc['ica_note'] = ('Skipped: EOG channel(s) and >=6 valid EEG channels required '
                          'for automatic ocular component identification.'
                          if cfg.use_ica else 'ICA disabled')
    # Average reference only after bad channels and ICA.
    raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')
    qc['good_eeg_channels'] = good_eeg
    qc['n_good_eeg'] = len(good_eeg)
    qc['sampling_hz'] = float(raw.info['sfreq'])
    return raw, qc


def read_events(path: Path, event_col: str, event_value: str | None):
    if path.suffix.lower() == '.tsv':
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
    if 'onset' not in df.columns:
        raise ValueError('events CSV/TSV must have onset (seconds from EEG recording start).')
    if event_value is not None:
        if event_col not in df.columns:
            raise ValueError(f'{event_col} not in events. Available: {df.columns.tolist()}')
        df = df[df[event_col].astype(str) == event_value].copy()
    if len(df) == 0:
        raise ValueError('No stimulus events remaining after filtering.')
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    if df['onset'].isna().any():
        raise ValueError('Every retained stimulus event must have a numeric onset.')
    df = df.sort_values('onset', kind='stable').reset_index(drop=True)
    if 'trial_id' not in df.columns:
        df['trial_id'] = [f'trial-{i+1:04d}' for i in range(len(df))]
    if df['trial_id'].duplicated().any():
        raise ValueError('trial_id values must be unique within a recording.')
    return df


def win(data, times, t0, t1):
    # End-exclusive to enforce nearly equal numbers of samples on both sides.
    mask = (times >= t0 - 1e-8) & (times < t1 - 1e-8)
    return data[..., mask]


def integrate_band(psd, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return float('nan')
    # Welch estimates are uV**2/Hz, integral -> uV**2.
    return float(np.trapezoid(psd[..., mask], x=freqs[mask], axis=-1).mean())


def psd_statistics(x_uv, sfreq):
    """Return region band log-powers, PAF and spectral entropy."""
    if x_uv.ndim == 1:
        x_uv = x_uv[None, :]
    if x_uv.shape[-1] < sfreq * 0.75:
        return None
    # No zero-padding; true spectral resolution determined by window length.
    freqs, pxx = signal.welch(x_uv, fs=sfreq, window='hann',
                              nperseg=x_uv.shape[-1], axis=-1,
                              detrend='constant')
    powers = {}
    for band, (lo, hi) in BANDS.items():
        p = integrate_band(pxx, freqs, lo, hi)
        powers[f'{band}_ln_power'] = float(np.log(max(p, EPS))) if np.isfinite(p) else float('nan')
    mask = (freqs >= 8) & (freqs <= 13)
    avg_psd = np.mean(pxx, axis=0)
    if np.count_nonzero(mask) >= 3:
        alpha_psd = avg_psd[mask]
        # Peak is only reported when a modestly identifiable local alpha maximum exists.
        quality = np.max(alpha_psd) / max(np.median(alpha_psd), EPS)
        paf = float(freqs[mask][np.argmax(alpha_psd)]) if quality >= 1.2 else float('nan')
    else:
        paf = float('nan')
    mask = (freqs >= 4) & (freqs <= 45)
    d = avg_psd[mask]
    distribution = d / max(float(d.sum()), EPS)
    spectral_entropy = float(-np.sum(distribution * np.log(distribution + EPS)) /
                             np.log(max(len(distribution), 2)))
    return powers, paf, spectral_entropy


def band_envelope_cv(x_uv, sfreq, lo=8, hi=13):
    if x_uv.shape[-1] < sfreq * 1.0:
        return float('nan')
    sos = signal.butter(4, (lo, hi), fs=sfreq, btype='bandpass', output='sos')
    y = signal.sosfiltfilt(sos, x_uv, axis=-1)
    env = np.abs(signal.hilbert(y, axis=-1))
    # Rows=channels. Mean across channel CVs.
    return float(np.mean(env.std(axis=-1) / np.maximum(env.mean(axis=-1), 1e-10)))


def pair_connectivity(x1, x2, sfreq, lo, hi):
    """Magnitude-squared coherence and phase-lag index, exploratory only."""
    if len(x1) < sfreq * 1.25:
        return (float('nan'), float('nan'))
    nseg = min(len(x1), int(round(1.0 * sfreq)))
    freqs, coh = signal.coherence(x1, x2, fs=sfreq, nperseg=nseg,
                                  noverlap=nseg // 2)
    mask = (freqs >= lo) & (freqs < hi)
    c = float(np.mean(coh[mask])) if np.any(mask) else float('nan')
    # Within-window PLI: use only as exploratory, with known short-window bias.
    sos = signal.butter(3, (lo, hi), fs=sfreq, btype='bandpass', output='sos')
    a = signal.hilbert(signal.sosfiltfilt(sos, x1))
    b = signal.hilbert(signal.sosfiltfilt(sos, x2))
    angles = np.angle(a) - np.angle(b)
    pli = float(np.abs(np.mean(np.sign(np.sin(angles)))))
    return c, pli


def valid_channels(ch_names):
    return {name_key(n): i for i, n in enumerate(ch_names)}


def regional(x, ch_map, names):
    found = [ch_map[name_key(ch)] for ch in names if name_key(ch) in ch_map]
    return x[found] if found else None


def load_neonatal_template(path: Path, channel: str) -> dict:
    """Load a supplied, experimentally justified infant noxious-evoked waveform.

    CSV columns: time_s, amplitude_uv. Units and montage must be verified by user.
    Never generates an adult-to-neonate surrogate or a template from test data.
    """
    tab = pd.read_csv(path)
    if not {'time_s', 'amplitude_uv'}.issubset(tab.columns):
        raise ValueError('Neonatal template requires columns time_s and amplitude_uv.')
    time = pd.to_numeric(tab['time_s'], errors='coerce').to_numpy(float)
    amplitude = pd.to_numeric(tab['amplitude_uv'], errors='coerce').to_numpy(float)
    if (len(time) < 20 or not np.isfinite(time).all() or
            not np.isfinite(amplitude).all() or np.any(np.diff(time) <= 0)):
        raise ValueError('Neonatal template requires >=20 finite, strictly increasing samples.')
    if time[0] > 0.05 or time[-1] < 0.70:
        raise ValueError('Neonatal template must cover at least 0.05 to 0.70 s after stimulus.')
    if np.std(amplitude) <= 1e-9:
        raise ValueError('Neonatal template has no waveform variation.')
    return {'time': time, 'waveform': amplitude, 'channel': channel}


def neonatal_template_metrics(epoch_uv, times, ch_map, template):
    """Waveform similarity over 0.05-0.70 s; template from independent study/training only."""
    if template is None or name_key(template['channel']) not in ch_map:
        return {'neonatal_noxious_template_corr': np.nan,
                'neonatal_noxious_template_projection': np.nan}
    mask = (times >= 0.05) & (times <= 0.70)
    x = epoch_uv[ch_map[name_key(template['channel'])]][mask]
    local_time = times[mask]
    ref = np.interp(local_time, template['time'], template['waveform'])
    x = x - np.mean(x)
    ref = ref - np.mean(ref)
    denom = np.linalg.norm(x) * np.linalg.norm(ref)
    correlation = float(np.dot(x, ref) / denom) if denom > 1e-10 else np.nan
    projection = float(np.dot(x, ref) / np.dot(ref, ref)) if np.dot(ref, ref) > 1e-10 else np.nan
    return {'neonatal_noxious_template_corr': correlation,
            'neonatal_noxious_template_projection': projection}


def row_features(epoch_uv, times, ch_names, sfreq, cfg, neonatal_template=None):
    """One event, returns flat numeric features with pre/post/delta prefixes."""
    ch_map = valid_channels(ch_names)
    pre = win(epoch_uv, times, cfg.pre_start, cfg.pre_end)
    post = win(epoch_uv, times, cfg.post_start, cfg.post_end)
    if abs(pre.shape[1] - post.shape[1]) > 1:
        raise ValueError('Pre and post windows must have equal samples.')
    result = {}
    groups = {
        'sensorimotor': ['C3', 'C4'],
        'frontal_midline': ['Fz', 'FCz'],
        'centro_midline': ['Cz'],
        'gamma_central': ['C3', 'C4', 'Cz'],
        'global': ch_names,
    }
    use_bands = {
        'sensorimotor': ['alpha', 'beta'],
        'frontal_midline': ['theta'],
        'centro_midline': ['theta'],
        'gamma_central': ['gamma'],
        'global': ['alpha', 'beta', 'theta', 'gamma'],
    }
    for region, chan_names in groups.items():
        # Unavailable canonical channels -> NaN; no data-dependent substitution.
        d = regional(epoch_uv, ch_map, chan_names)
        for period, (t0, t1) in {'pre': (cfg.pre_start, cfg.pre_end),
                                 'post': (cfg.post_start, cfg.post_end)}.items():
            segment = win(d, times, t0, t1) if d is not None else None
            stats = psd_statistics(segment, sfreq) if segment is not None else None
            for band in use_bands[region]:
                key = f'{region}_{band}_ln_power'
                result[f'{period}_{key}'] = (stats[0][f'{band}_ln_power']
                                              if stats is not None else np.nan)
            if region in ('sensorimotor', 'global'):
                result[f'{period}_{region}_peak_alpha_hz'] = stats[1] if stats else np.nan
                result[f'{period}_{region}_spectral_entropy'] = stats[2] if stats else np.nan
            if region == 'sensorimotor':
                # Alpha envelope variation over the 1.8-sec window is exploratory.
                result[f'{period}_sensorimotor_alpha_envelope_cv'] = (
                    band_envelope_cv(segment, sfreq) if segment is not None else np.nan)
    # Frontal alpha asymmetry: positive => relatively greater right F4 alpha power.
    # Defined as ln(P_F4) - ln(P_F3), not a left/right pain indicator.
    for period, (t0, t1) in {'pre': (cfg.pre_start, cfg.pre_end),
                             'post': (cfg.post_start, cfg.post_end)}.items():
        if 'F3' in ch_map and 'F4' in ch_map:
            x3 = win(epoch_uv[ch_map['F3']], times, t0, t1)
            x4 = win(epoch_uv[ch_map['F4']], times, t0, t1)
            alpha3 = psd_statistics(x3, sfreq)[0]['alpha_ln_power']
            alpha4 = psd_statistics(x4, sfreq)[0]['alpha_ln_power']
            result[f'{period}_frontal_alpha_asymmetry_ln_f4_over_f3'] = alpha4 - alpha3
        else:
            result[f'{period}_frontal_alpha_asymmetry_ln_f4_over_f3'] = np.nan
    # Fixed sensor-pair connectivity (not source-space connectivity).
    for pairname, (one, two) in PAIRS.items():
        for band, (lo, hi) in BANDS.items():
            for period, (t0, t1) in {'pre': (cfg.pre_start, cfg.pre_end),
                                     'post': (cfg.post_start, cfg.post_end)}.items():
                co_name = f'{period}_{pairname}_{band}_coherence'
                pl_name = f'{period}_{pairname}_{band}_pli'
                if name_key(one) not in ch_map or name_key(two) not in ch_map:
                    result[co_name] = np.nan
                    result[pl_name] = np.nan
                    continue
                x1 = win(epoch_uv[ch_map[name_key(one)]], times, t0, t1)
                x2 = win(epoch_uv[ch_map[name_key(two)]], times, t0, t1)
                result[co_name], result[pl_name] = pair_connectivity(x1, x2, sfreq, lo, hi)
    # Fill change features only when both measurements exist; no imputation.
    for key, pre_val in list(result.items()):
        if key.startswith('pre_'):
            ending = key[len('pre_'):]
            result['delta_' + ending] = result['post_' + ending] - pre_val
    # ERPs: low-pass to stabilize single-trial peaks; pre-stim mean correction.
    if name_key(cfg.n2_p2_channel) in ch_map:
        erp = epoch_uv[ch_map[name_key(cfg.n2_p2_channel)]]
        sos = signal.butter(4, cfg.erp_lowpass_hz, fs=sfreq, output='sos')
        smooth = signal.sosfiltfilt(sos, erp)
        baseline = win(smooth, times, cfg.erp_baseline_start, cfg.erp_baseline_end)
        smooth = smooth - float(np.mean(baseline))
        n2mask = (times >= cfg.n2_start) & (times < cfg.n2_end)
        p2mask = (times >= cfg.p2_start) & (times < cfg.p2_end)
        n2idx = np.where(n2mask)[0][np.argmin(smooth[n2mask])]
        p2idx = np.where(p2mask)[0][np.argmax(smooth[p2mask])]
        result['erp_cz_n2_amp_uv'] = float(smooth[n2idx])
        result['erp_cz_n2_latency_ms'] = float(times[n2idx] * 1000)
        result['erp_cz_p2_amp_uv'] = float(smooth[p2idx])
        result['erp_cz_p2_latency_ms'] = float(times[p2idx] * 1000)
        result['erp_cz_n2_to_p2_uv'] = float(smooth[p2idx] - smooth[n2idx])
        result['erp_cz_n2_mean_uv'] = float(np.mean(smooth[n2mask]))
        result['erp_cz_p2_mean_uv'] = float(np.mean(smooth[p2mask]))
        temporal_mask = (times >= cfg.temporal_start) & (times < cfg.temporal_end)
        # Single-trial response variation, NOT between-trial variability.
        result['erp_cz_response_temporal_sd_uv'] = float(np.std(smooth[temporal_mask]))
        result['erp_cz_response_temporal_rms_uv'] = float(np.sqrt(np.mean(smooth[temporal_mask]**2)))
        # Matched pre/post 600 ms temporal variability estimate, so the 11th
        # table category also has a baseline-adjusted model input.
        pre_temporal = (times >= -cfg.temporal_end) & (times < -cfg.temporal_start)
        result['pre_cz_response_temporal_sd_uv'] = float(np.std(smooth[pre_temporal]))
        result['post_cz_response_temporal_sd_uv'] = float(np.std(smooth[temporal_mask]))
        result['delta_cz_response_temporal_sd_uv'] = (
            result['post_cz_response_temporal_sd_uv'] - result['pre_cz_response_temporal_sd_uv'])
    else:
        for name in ('n2_amp_uv','n2_latency_ms','p2_amp_uv','p2_latency_ms',
                     'n2_to_p2_uv','n2_mean_uv','p2_mean_uv',
                     'response_temporal_sd_uv', 'response_temporal_rms_uv'):
            result['erp_cz_' + name] = np.nan
        for metric in ('pre_cz_response_temporal_sd_uv',
                       'post_cz_response_temporal_sd_uv',
                       'delta_cz_response_temporal_sd_uv'):
            result[metric] = np.nan
    result.update(neonatal_template_metrics(epoch_uv, times, ch_map, neonatal_template))
    return result


def process(raw_path: Path, events_path: Path, output_dir: Path, cfg: Config,
            event_col: str, event_value: str | None, manual_bad_names, subject,
            session, neonatal_template_path: Path | None = None,
            neonatal_cohort: bool = False, neonatal_template_channel: str = 'Cz'):
    if bool(neonatal_template_path) != bool(neonatal_cohort):
        raise ValueError('A neonatal template and --neonatal-cohort must be supplied together; never apply an infant template to adult EEG.')
    neonatal_template = (load_neonatal_template(neonatal_template_path, neonatal_template_channel)
                         if neonatal_cohort else None)
    output_dir.mkdir(exist_ok=True, parents=True)
    events_df = read_events(events_path, event_col, event_value)
    raw = load_raw(raw_path)
    LOGGER.info('Loaded %s: %.1f s, %.1f Hz, %d channels', raw_path.name,
                raw.times[-1], raw.info['sfreq'], len(raw.ch_names))
    raw, preprocessing_qc = preprocess(raw, cfg, manual_bad_names)
    # Only EEG channels, excluding bad channels; no EOG in spectral metrics.
    names = preprocessing_qc['good_eeg_channels']
    sfreq = float(raw.info['sfreq'])
    # MNE events are absolute sample indices (first_samp offset included).
    samples = raw.time_as_index(events_df['onset'].to_numpy(), use_rounding=True) + raw.first_samp
    mne_events = np.column_stack((samples, np.zeros(len(samples), dtype=int),
                                   np.ones(len(samples), dtype=int)))
    # Drop invalid events in/near boundaries or annotations; retain QC output.
    epoch_start = min(cfg.pre_start, cfg.erp_baseline_start)
    epoch_end = max(cfg.post_end, cfg.p2_end)
    epochs = mne.Epochs(raw, events=mne_events, event_id={'pain_stimulus': 1},
                        tmin=epoch_start, tmax=epoch_end, baseline=None,
                        picks=names, preload=True, reject=None,
                        reject_by_annotation=True, on_missing='raise',
                        event_repeated='drop', verbose='ERROR')
    ep_times = epochs.times
    dat = epochs.get_data(units='uV')
    # EOG is deliberately excluded from EEG feature estimates, but large ocular
    # excursions can reject entire trials when an actual EOG channel is recorded.
    eog_raw = (raw.get_data(picks=preprocessing_qc['eog_channels']) * 1e6
               if preprocessing_qc['eog_channels'] else None)
    # Obtain drop/rejection mapping from selection, not indexing raw events.
    epoch_map = {int(input_index): pos for pos, input_index in enumerate(epochs.selection)}
    accepted, qc_rows = [], []
    for i, event in events_df.iterrows():
        onset = float(event['onset'])
        record = {'subject': subject, 'session': session, 'trial_id': str(event['trial_id']),
                  'onset_s': onset, 'event_index': int(i),
                  'accepted': False, 'reject_reason': '',
                  'max_eeg_ptp_uv': np.nan, 'min_eeg_ptp_uv': np.nan,
                  'max_eog_ptp_uv': np.nan}
        if i not in epoch_map:
            reason = epochs.drop_log[i] if i < len(epochs.drop_log) else ('OUT_OF_BOUNDS',)
            record['reject_reason'] = '|'.join(reason) if reason else 'EPOCH_DROPPED'
            qc_rows.append(record)
            continue
        epoch_uv = dat[epoch_map[i]]
        ptp = np.ptp(epoch_uv, axis=-1)
        record['max_eeg_ptp_uv'] = float(np.max(ptp))
        record['min_eeg_ptp_uv'] = float(np.min(ptp))
        if eog_raw is not None:
            start_sample = int(epochs.events[epoch_map[i], 0] - raw.first_samp
                               + round(epoch_start * sfreq))
            ocular = eog_raw[:, start_sample:start_sample+epoch_uv.shape[1]]
            if ocular.shape[1] == epoch_uv.shape[1]:
                record['max_eog_ptp_uv'] = float(np.ptp(ocular, axis=-1).max())
        if not np.all(np.isfinite(epoch_uv)):
            record['reject_reason'] = 'NONFINITE_EEG'
        elif np.max(ptp) > cfg.reject_ptp_uv:
            record['reject_reason'] = 'AMPLITUDE_ARTIFACT'
        elif np.min(ptp) < cfg.flat_ptp_uv:
            record['reject_reason'] = 'FLAT_CHANNEL_IN_TRIAL'
        elif (np.isfinite(record['max_eog_ptp_uv']) and
              record['max_eog_ptp_uv'] > cfg.eog_reject_ptp_uv):
            record['reject_reason'] = 'OCULAR_EOG_ARTIFACT'
        if record['reject_reason']:
            qc_rows.append(record)
            continue
        f = row_features(epoch_uv, ep_times, names, sfreq, cfg, neonatal_template)
        row = {'subject': subject, 'session': session, 'cohort': 'neonatal' if neonatal_cohort else 'adult',
               'trial_id': str(event['trial_id']), 'onset_s': onset}
        for col in label_column_names(list(events_df.columns)):
            if col not in row and not col.startswith(('pre_', 'post_', 'delta_', 'erp_')):
                row[col] = event[col]
        row.update(f)
        accepted.append(row)
        record['accepted'] = True
        qc_rows.append(record)
    # Prior-trial variability uses only PREVIOUS accepted trials from this
    # recording. It never consults future responses or subjective pain labels.
    # First 3 trials have insufficient history and therefore retain NaN.
    recent = []
    for row in accepted:
        row['erp_cz_n2_to_p2_previous5_sd_uv'] = (float(np.std(recent[-5:], ddof=1))
            if len(recent) >= 3 else np.nan)
        amplitude = row.get('erp_cz_n2_to_p2_uv', np.nan)
        if np.isfinite(amplitude):
            recent.append(float(amplitude))
    qc_df = pd.DataFrame(qc_rows)
    qc_path = output_dir / 'trial_qc.csv'
    qc_df.to_csv(qc_path, index=False)
    feature_path = output_dir / 'eeg_trial_features.csv'
    if not accepted:
        # Still output a valid QC file and metadata, rather than fabricating data.
        LOGGER.warning('No epochs passed QC. Check event times, window limits, and thresholds.')
        pd.DataFrame(columns=['subject','session','trial_id','onset_s']).to_csv(feature_path, index=False)
        feature_df = pd.DataFrame()
    else:
        feature_df = pd.DataFrame(accepted)
        feature_df.to_csv(feature_path, index=False, float_format='%.7g')
    subject_summary = output_dir / 'subject_feature_variability.csv'
    # Distinct from features because per-subject aggregations can leak across CV folds.
    erp_cols = [c for c in feature_df.columns if c.startswith('erp_')]
    if len(feature_df) >= 2 and erp_cols:
        records = {'subject': subject, 'session': session, 'n_accepted': len(feature_df)}
        for c in erp_cols:
            records[c + '_across_trial_sd'] = float(feature_df[c].std(ddof=1))
        pd.DataFrame([records]).to_csv(subject_summary, index=False)
    else:
        pd.DataFrame(columns=['subject','session','n_accepted']).to_csv(subject_summary, index=False)
    metadata = {
        'raw_file': str(raw_path), 'events_file': str(events_path),
        'feature_file': str(feature_path),
        'config': asdict(cfg), 'preprocessing_qc': preprocessing_qc,
        'cohort': 'neonatal' if neonatal_cohort else 'adult_or_unspecified',
        'neonatal_template_path': str(neonatal_template_path) if neonatal_template_path else None,
        'neonatal_template_channel': neonatal_template_channel if neonatal_cohort else None,
        'n_stimulus_events': len(events_df), 'n_accepted': len(accepted),
        'n_rejected': len(events_df) - len(accepted),
        'missing_feature_columns': [c for c in feature_df.columns if
                                    (c.startswith(('pre_', 'post_', 'delta_', 'erp_'))
                                     and feature_df[c].isna().all())],
        'notes': [
            'EEG baseline power and post-stimulus power are observed features, not an additive decomposition.',
            'ERP features require event onset timing accuracy and should be validated with grand-average waveforms.',
            'Channel-space connectivity may reflect volume conduction; PLI can be biased on short epochs.',
            'PAF and envelope-CV on 1.8 s windows are exploratory and potentially unstable.',
            'Automated bad-channel/epoch rejection and ICA require visual QC on real data.',
            'Trial-wise labels are carried through but never used to calculate EEG features.',
            'Across-trial subject summary must not be fed to cross-subject ML without leakage-safe computation.',
            'Prior-five-trial SD looks backward only and is separate from single-trial temporal SD.',
            'Infant template correlation only calculated for explicitly neonatal cohorts using a user-supplied validated waveform.'
        ],
    }
    with (output_dir / 'run_metadata.json').open('w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2, allow_nan=False)
    LOGGER.info('Accepted %d/%d events; saved %s', len(accepted), len(events_df), feature_path)
    return feature_df, qc_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--raw', required=True, type=Path, help='Continuous EEG file (.fif/.edf/.bdf/.vhdr/.set).')
    p.add_argument('--events', required=True, type=Path, help='Event CSV/TSV: onset seconds, optional labels.')
    p.add_argument('--out', required=True, type=Path, help='Output directory.')
    p.add_argument('--subject', required=True, help='e.g., sub-001')
    p.add_argument('--session', default='ses-01')
    p.add_argument('--event-col', default='trial_type')
    p.add_argument('--event-value', default=None,
                   help='Select ONLY actual stimulus events, e.g., laser. Required if mixed event types.')
    p.add_argument('--bad-channels', default='', help='Comma-separated manually identified bad EEG channels.')
    p.add_argument('--mains', type=float, default=60, help='Local mains frequency (50 or 60 Hz); use 0 to skip.')
    p.add_argument('--reject-ptp-uv', type=float, default=150.0)
    p.add_argument('--reject-eog-ptp-uv', type=float, default=150.0)
    p.add_argument('--skip-ica', action='store_true', help='Do not fit ocular ICA.')
    p.add_argument('--neonatal-cohort', action='store_true', help='Explicitly identify a neonatal experiment; requires --neonatal-template.')
    p.add_argument('--neonatal-template', type=Path, default=None,
                   help='Scientifically validated external template CSV: time_s, amplitude_uv.')
    p.add_argument('--neonatal-template-channel', default='Cz', help='Electrode matched to the neonatal template.')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s')
    cfg = Config(mains_hz=a.mains, reject_ptp_uv=a.reject_ptp_uv,
                 eog_reject_ptp_uv=a.reject_eog_ptp_uv,
                 use_ica=not a.skip_ica)
    process(a.raw, a.events, a.out, cfg, a.event_col, a.event_value,
            [x.strip() for x in a.bad_channels.split(',') if x.strip()],
            a.subject, a.session, a.neonatal_template,
            a.neonatal_cohort, a.neonatal_template_channel)


if __name__ == '__main__':
    main()
