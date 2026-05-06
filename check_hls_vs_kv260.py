"""
Compare y_pred_hls.npy vs y_pred_kv260.npy.
kv260 may have fewer samples — only the first N rows (kv260 length) are compared.

Usage:
    python check_hls_vs_kv260.py [--dir PATH]
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    default='hls4ml_output/hls4mlprj_exp_full',
    help='Directory containing y_pred_hls.npy and y_pred_kv260.npy',
)
args = parser.parse_args()

out_dir = Path(args.dir)
hls_path = out_dir / 'y_pred_hls.npy'
kv260_path = out_dir / 'y_pred_kv260.npy'

y_hls = np.load(hls_path)
y_kv260 = np.load(kv260_path)

n = len(y_kv260)
y_hls_sub = y_hls[:n]

print(f'y_pred_hls   shape : {y_hls.shape}')
print(f'y_pred_kv260 shape : {y_kv260.shape}')
print(f'Comparing first {n} samples\n')

diff = np.abs(y_hls_sub - y_kv260)

print(f'Max  |hls − kv260| : {diff.max():.6f}')
print(f'Mean |hls − kv260| : {diff.mean():.6f}')
print(f'Std  |hls − kv260| : {diff.std():.6f}')

tol = 1e-3
n_mismatch = int((diff > tol).any(axis=-1).sum())
print(f'\nSamples with any element > {tol} : {n_mismatch} / {n}')

if n_mismatch == 0:
    print('PASS — outputs match within tolerance.')
else:
    print('FAIL — outputs differ.')
    first_bad = int(np.argmax((diff > tol).any(axis=-1)))
    print(f'\nFirst mismatch at sample {first_bad}:')
    print(f'  hls  : {y_hls_sub[first_bad]}')
    print(f'  kv260: {y_kv260[first_bad]}')
    print(f'  diff : {diff[first_bad]}')
