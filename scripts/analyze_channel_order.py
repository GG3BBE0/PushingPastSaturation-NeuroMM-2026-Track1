"""Check whether the 23 EEG channels are spatially ordered (adjacent = high corr)."""
import numpy as np
from pathlib import Path

root = Path('neuromm26_datasets/processed/features/eeg')
files = sorted(root.glob('*.npy'))
rng = np.random.default_rng(0)
sel = rng.choice(len(files), size=min(300, len(files)), replace=False)

corr_sum = np.zeros((23, 23)); n = 0
for i in sel:
    w = np.load(files[i]).astype(np.float32)
    eeg = w[:23]
    eeg = (eeg - eeg.mean(1, keepdims=True)) / (eeg.std(1, keepdims=True) + 1e-6)
    c = np.corrcoef(eeg)
    if not np.isnan(c).any():
        corr_sum += np.abs(c); n += 1
C = corr_sum / n
print(f'averaged |corr| over {n} samples')

adj = [C[i, i + 1] for i in range(22)]
offdiag = C[~np.eye(23, dtype=bool)]
pp = ' '.join(f'{a:.2f}' for a in adj)
print(f'adjacent |corr| mean={np.mean(adj):.3f}')
print(f'  per-pair (ch_i,ch_i+1): {pp}')
print(f'all off-diag |corr| mean={np.mean(offdiag):.3f}')
print(f'ratio adjacent/all = {np.mean(adj) / np.mean(offdiag):.2f}x')

nbr_is_max = 0
for i in range(23):
    row = C[i].copy(); row[i] = 0
    j = int(row.argmax())
    if abs(j - i) == 1:
        nbr_is_max += 1
print(f'channels whose strongest partner is an immediate neighbor: {nbr_is_max}/23')

print('\ncorr matrix (first 10x10):')
for i in range(10):
    print(' '.join(f'{C[i, j]:.2f}' for j in range(10)))
