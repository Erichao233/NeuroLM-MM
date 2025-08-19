"""
by Wei-Bang Jiang (extended for ECG)
https://github.com/935963004/NeuroLM

ECG-only preprocessing for HMC dataset:
- keep channel: 'ECG'
- bandpass 0.1-75 Hz, notch 50 Hz, resample 200 Hz (match EEG pipeline)
- segment into 30s epochs using scoring file
- save as pkl with keys: X (C x T), ch_names (list), y (int)
"""

import mne
import numpy as np
import os
import pickle
import pandas as pd


symbols_hmc = {' Sleep stage W': 0, ' Sleep stage N1': 1, ' Sleep stage N2': 2, ' Sleep stage N3': 3,' Sleep stage R': 4}


def build_events(signals, times, event_df, fs: float = 200.0):
    num_events = len(event_df)
    num_chan, num_points = signals.shape
    features = []
    labels = []
    for _, row in event_df.iterrows():
        if row[' Duration'] != 30:
            continue
        start = np.where((times) >= row[' Recording onset'])[0][0]
        end = np.where((times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        if end - start != int(fs) * 30:
            # guard, skip abnormal epoch
            continue
        features.append(signals[:, start:end])
        labels.append(symbols_hmc[row[' Annotation']])
    if len(features) == 0:
        return np.zeros((0, num_chan, int(fs) * 30)), np.zeros((0, 1))
    features = np.stack(features, axis=0)
    labels = np.array(labels).reshape(-1, 1)
    return features, labels


def read_edf(filename: str):
    raw = mne.io.read_raw_edf(filename, preload=True)
    # retain ECG only if present
    if 'ECG' in raw.ch_names:
        raw.pick_channels(['ECG'])
    else:
        # some files may store ECG with different label; skip if not found
        raise ValueError('ECG channel not found')

    # preprocessing (match EEG pipeline)
    raw.filter(l_freq=0.1, h_freq=75.0)
    raw.notch_filter(50.0)
    raw.resample(200, n_jobs=5)

    _, times = raw[:]
    signals = raw.get_data(units='uV')
    label_file = filename[0:-4] + "_sleepscoring.txt"
    label_df = pd.read_csv(label_file)
    raw.close()
    return signals, times, label_df


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def process_split(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in files:
        print(f"\t{fname}")
        try:
            signals, times, label_df = read_edf(fname)
        except (ValueError, KeyError) as e:
            print(f"skip {fname}: {e}")
            continue
        features, labels = build_events(signals, times, label_df)
        for idx, (signal, label) in enumerate(zip(features, labels)):
            sample = {
                'X': signal,               # shape (1, 6000)
                'ch_names': ['ECG'],
                'y': int(label),
            }
            save_pickle(sample, os.path.join(out_dir, os.path.basename(fname).split('.')[0] + f"-ecg-{idx}.pkl"))


if __name__ == '__main__':
    # paths
    root = "/root/autodl-tmp/Datasets/HMC/hmc-sleep-staging/1.1/recordings"
    out_root = '/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC_ECG'
    train_out = os.path.join(out_root, 'train')
    eval_out = os.path.join(out_root, 'eval')
    test_out = os.path.join(out_root, 'test')

    # list edf files
    edf_files = []
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if len(fname) == 9 and fname.endswith('.edf'):
                edf_files.append(os.path.join(dirName, fname))
    edf_files.sort()

    train_files = edf_files[:100]
    eval_files = edf_files[100:125]
    test_files = edf_files[125:]

    print('processing train/eval/test for ECG...')
    process_split(train_files, train_out)
    process_split(eval_files, eval_out)
    process_split(test_files, test_out)


