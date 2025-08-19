"""
Prepare EOG (E1/E2) segments from HMC for 30s epochs, unified preprocessing with EEG pipeline:
- bandpass 0.1-75 Hz, notch 50 Hz, resample 200 Hz
- save pkl with keys: {'X': (C,T), 'ch_names': ['E1','E2'], 'y': int}
"""
import os
import mne
import numpy as np
import pandas as pd
import pickle

symbols_hmc = {' Sleep stage W': 0, ' Sleep stage N1': 1, ' Sleep stage N2': 2, ' Sleep stage N3': 3,' Sleep stage R': 4}

def read_edf(file_path: str):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    # pick EOG channels
    picks = []
    for ch in ['EOG E1-M2', 'EOG E2-M2']:
        if ch in raw.ch_names:
            picks.append(ch)
    if len(picks) == 0:
        raise ValueError('no EOG channels found')
    raw.pick_channels(picks)
    raw.filter(0.1, 75.0)
    raw.notch_filter(50.0)
    raw.resample(200, n_jobs=5)
    data = raw.get_data(units='uV')
    _, times = raw[:]
    label_file = file_path[:-4] + '_sleepscoring.txt'
    labels = pd.read_csv(label_file)
    raw.close()
    return data, times, labels, picks

def build_events(signals, times, labels, fs=200):
    feats = []
    ys = []
    for _, row in labels.iterrows():
        if row[' Duration'] != 30:
            continue
        start = np.where((times) >= row[' Recording onset'])[0][0]
        end = np.where((times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        if end - start != int(fs) * 30:
            continue
        feats.append(signals[:, start:end])
        ys.append(symbols_hmc[row[' Annotation']])
    if len(feats) == 0:
        return np.zeros((0, signals.shape[0], int(fs)*30)), np.zeros((0,1))
    feats = np.stack(feats, axis=0)
    ys = np.array(ys).reshape(-1,1)
    return feats, ys

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def process_split(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fp in files:
        try:
            sig, times, lab, picks = read_edf(fp)
        except Exception as e:
            print('skip', fp, e)
            continue
        feats, ys = build_events(sig, times, lab)
        for idx, (x, y) in enumerate(zip(feats, ys)):
            ch_names = []
            for name in picks:
                if 'EOG E1' in name:
                    ch_names.append('E1')
                elif 'EOG E2' in name:
                    ch_names.append('E2')
                else:
                    ch_names.append('EOG')
            sample = {'X': x, 'ch_names': ch_names, 'y': int(y)}
            save_pickle(sample, os.path.join(out_dir, os.path.basename(fp)[:-4] + f'-eog-{idx}.pkl'))

if __name__ == '__main__':
    root = "/root/autodl-tmp/Datasets/HMC/hmc-sleep-staging/1.1/recordings"
    out_root = '/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC_EOG'
    train_out = os.path.join(out_root, 'train')
    eval_out = os.path.join(out_root, 'eval')
    test_out = os.path.join(out_root, 'test')
    edf_files = []
    for d,_,fs in os.walk(root):
        for f in fs:
            if len(f)==9 and f.endswith('.edf'):
                edf_files.append(os.path.join(d,f))
    edf_files.sort()
    train_files = edf_files[:100]
    eval_files = edf_files[100:125]
    test_files = edf_files[125:]
    process_split(train_files, train_out)
    process_split(eval_files, eval_out)
    process_split(test_files, test_out)


