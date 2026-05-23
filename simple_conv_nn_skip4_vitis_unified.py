"""
Skip-connection Conv NN — 4-part partition experiment (U-Net style).

  PART 1  inp → enc_conv1(skip1) → enc_conv2 → enc_pool1
  PART 2  enc_pool1 → enc_conv3(skip2) → enc_pool2 → bottleneck
  PART 3  [bottleneck, skip2] → dec_up1 → dec_conv1 → skip2_add
  PART 4  [skip2_add, skip1] → dec_up2 → dec_conv2 → skip1_add → gap → dense1 → dense_out

  part1 : in (8,8,1)              out [(4,4,16), (8,8,8)]
  part2 : in (4,4,16)             out [(2,2,16), (4,4,16)]
  part3 : in [(2,2,16), (4,4,16)] out (4,4,16)
  part4 : in [(4,4,16), (8,8,8)]  out (4,)

Lock dir: hls4ml_output/_exp_locked_skip4/
"""

import argparse
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.src.layers import GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Add, Conv2D, Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model

import hls4ml

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description='Skip-connection Conv NN hls4ml experiment (4-part)')
_parser.add_argument(
    '--mode',
    choices=['all', 'full', 'parts', 'part1', 'part2', 'part3', 'part4'],
    default='all',
    help=(
        'all   — full model + all 4 partitions + diag  (default)\n'
        'full  — full model only\n'
        'parts — part1–part4 only (no full model, no diag)\n'
        'part1…part4 — single partition'
    ),
)
_parser.add_argument(
    '--diag',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='HLS bisect diagnostic after full model (default: on)',
)
_ARGS = _parser.parse_args()
_RUN = {
    'all': {'full', 'part1', 'part2', 'part3', 'part4'},
    'parts': {'part1', 'part2', 'part3', 'part4'},
}.get(_ARGS.mode, {_ARGS.mode})

# ── USER CONFIG ───────────────────────────────────────────────────────────────
NUM_QUERIES = 20_000
MAX_QUERIES = 1_000_000
HLS_REUSE_FACTOR = 8
HLS_PRECISION = 'ap_fixed<16,6>'
HLS_OUT_PRECISION = 'ap_fixed<16,2>'
HLS_GAP_PRECISION = 'ap_fixed<32,16>'

# ── PATHS ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent / 'hls4ml_output2'
_LOCK = _BASE / '_exp_locked_skip4'
_WEIGHTS_FILE = _LOCK / 'full_skip4_weights.h5'
_INPUT_FILE = _LOCK / f'x_input_{MAX_QUERIES}.npy'

_DIRS = {
    'full': str(_BASE / 'hls4mlprj_skip4_full'),
    'part1': str(_BASE / 'hls4mlprj_skip4_part1'),
    'part2': str(_BASE / 'hls4mlprj_skip4_part2'),
    'part3': str(_BASE / 'hls4mlprj_skip4_part3'),
    'part4': str(_BASE / 'hls4mlprj_skip4_part4'),
}
_PROJECT_NAMES = {
    'full': 'my_proj_skip4_full',
    'part1': 'my_proj_skip4_p1',
    'part2': 'my_proj_skip4_p2',
    'part3': 'my_proj_skip4_p3',
    'part4': 'my_proj_skip4_p4',
}

# ── SETUP ─────────────────────────────────────────────────────────────────────
assert NUM_QUERIES <= MAX_QUERIES, f'NUM_QUERIES ({NUM_QUERIES}) > MAX_QUERIES ({MAX_QUERIES})'
_LOCK.mkdir(parents=True, exist_ok=True)
for _d in _DIRS.values():
    if os.path.exists(_d):
        shutil.rmtree(_d)
    os.makedirs(_d)

os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2023.2'
os.environ['XILINX_VIVADO'] = '/tools/Xilinx/Vivado/2023.2'
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

# ── TIMING ────────────────────────────────────────────────────────────────────
_tlogs: dict = {}


def _open_tlog(key: str) -> None:
    f = open(Path(_DIRS[key]) / 'timing_log.txt', 'w', buffering=1)
    _tlogs[key] = (f, f.name)
    _tlog_write(key, f'=== [{key}] timing log  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    _tlog_write(key, f'NUM_QUERIES={NUM_QUERIES}  MAX_QUERIES={MAX_QUERIES}')
    _tlog_write(key, f'weights_locked={_WEIGHTS_FILE.exists()}  input_locked={_INPUT_FILE.exists()}\n')


def _tlog_write(key: str, line: str) -> None:
    f, _ = _tlogs[key]
    f.write(line + '\n')
    os.fsync(f.fileno())


@contextmanager
def timed_step(key: str, name: str):
    _tlog_write(key, f'[START] {name}  ({time.strftime("%H:%M:%S")})')
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _tlog_write(key, f'[END]   {name}  elapsed={elapsed:.3f}s\n')
        print(f'  [{key}][timing] {name}: {elapsed:.3f}s')


for _key in _DIRS:
    _open_tlog(_key)


# ── HLS CONFIG HELPER (shared by run_model + diag) ────────────────────────────
def _make_hls_cfg(model) -> dict:
    """Build hls4ml config with per-layer precision rules."""
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg['Model']['Strategy'] = 'resource'
    cfg['Model']['ReuseFactor'] = HLS_REUSE_FACTOR
    cfg['Model']['Precision'] = HLS_PRECISION
    for ln in cfg.get('LayerName', {}):
        cfg['LayerName'][ln]['Precision'] = (
            HLS_GAP_PRECISION if any(p in ln.lower() for p in ('dense', 'gap')) else HLS_OUT_PRECISION
        )
    return cfg


_HLS_PARAMS = dict(
    backend='VitisUnified',
    io_type='io_stream',
    board='kv260',
    part='xck26-sfvc784-2LV-c',
    clock_period='10ns',
    input_type='float',
    output_type='float',
    axi_mode='axi_stream',
)


# ── WEIGHT LOCK HELPER ────────────────────────────────────────────────────────
def _init_weights(model) -> str:
    """Load weights from lock file, or train + save if missing."""
    if _WEIGHTS_FILE.exists():
        model.load_weights(str(_WEIGHTS_FILE))
        return f'Weights loaded ← {_WEIGHTS_FILE}'
    np.random.seed(0)
    X_tr = np.random.rand(500, 8, 8, 1).astype(np.float32)
    y_tr = np.eye(4)[np.random.randint(0, 4, 500)]
    print('[train] Training 20 epochs to break weight symmetry…')
    model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0)
    model.save_weights(str(_WEIGHTS_FILE))
    return f'Weights trained + saved → {_WEIGHTS_FILE}'


# ── INPUT POOL HELPER ─────────────────────────────────────────────────────────
def _load_input_pool() -> np.ndarray:
    """Load locked input pool, or generate + save if missing."""
    if _INPUT_FILE.exists():
        pool = np.load(str(_INPUT_FILE))
        print(f'[lock] Loaded {pool.shape[0]:,} samples ← {_INPUT_FILE}')
        _tlog_write('full', f'Input pool loaded ← {_INPUT_FILE}')
    else:
        np.random.seed(42)
        pool = np.random.rand(MAX_QUERIES, 8, 8, 1).astype(np.float32)
        np.save(str(_INPUT_FILE), pool)
        sz_mb = _INPUT_FILE.stat().st_size / 1e6
        print(f'[lock] Saved {MAX_QUERIES:,} samples → {_INPUT_FILE}  ({sz_mb:.1f} MB)')
        _tlog_write('full', f'Input pool saved → {_INPUT_FILE}  ({sz_mb:.1f} MB)')
    return pool


# ── SAVE PREDICTIONS HELPER ───────────────────────────────────────────────────
def _save_preds(key, output_dir, X_input, y_hls_r, y_keras) -> None:
    diff = float(np.max(np.abs(y_hls_r - y_keras)))
    n_show = min(10, len(y_hls_r))
    x_save = X_input[0] if isinstance(X_input, list) else X_input
    print(f'  [{key}] Max |HLS − Keras| = {diff:.6f}')
    _tlog_write(key, f'Max |HLS - Keras| = {diff:.6f}')
    np.save(os.path.join(output_dir, 'x_input.npy'), x_save)
    np.save(os.path.join(output_dir, 'y_pred_hls.npy'), y_hls_r)
    np.save(os.path.join(output_dir, 'y_pred_keras.npy'), y_keras)
    print(f'  [{key}] Results saved → {output_dir}')
    print(f'  [{key}] y_pred_keras (first {n_show}):\n{y_keras[:n_show]}')
    print(f'  [{key}] y_pred_hls   (first {n_show}):\n{y_hls_r[:n_show]}')


# ── BUILD KERAS MODELS ────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Building Keras models (skip-connection, 4-part)…')
print('=' * 60)

tf.random.set_seed(0)

with timed_step('full', '1. Keras model definition'):
    inp = Input(shape=(8, 8, 1), name='inp')

    # Part 1 — Encoder-A
    s1 = Conv2D(8, (3, 3), padding='same', activation='relu', name='enc_conv1')(inp)  # (8,8,8)  → skip1
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='enc_conv2')(s1)  # (8,8,16)
    p1_out = MaxPooling2D((2, 2), name='enc_pool1')(x)  # (4,4,16)

    # Part 2 — Encoder-B + Bottleneck
    s2 = Conv2D(16, (3, 3), padding='same', activation='relu', name='enc_conv3')(p1_out)  # (4,4,16) → skip2
    x = MaxPooling2D((2, 2), name='enc_pool2')(s2)  # (2,2,16)
    p2_out = Conv2D(16, (3, 3), padding='same', activation='relu', name='bottleneck')(x)  # (2,2,16)

    # Part 3 — Decoder-A
    y = UpSampling2D((2, 2), name='dec_up1')(p2_out)  # (4,4,16)
    y = Conv2D(16, (3, 3), padding='same', activation='relu', name='dec_conv1')(y)  # (4,4,16)
    p3_out = Add(name='skip2_add')([y, s2])  # (4,4,16) ← skip2

    # Part 4 — Decoder-B + Head
    y = UpSampling2D((2, 2), name='dec_up2')(p3_out)  # (8,8,16)
    y = Conv2D(8, (3, 3), padding='same', activation='relu', name='dec_conv2')(y)  # (8,8,8)
    y = Add(name='skip1_add')([y, s1])  # (8,8,8)  ← skip1
    y = GlobalAveragePooling2D(name='gap')(y)
    y = Dense(16, activation='relu', name='dense1')(y)
    full_out = Dense(4, activation=None, name='dense_out')(y)

    full_model = Model(inp, full_out, name='full_model')
    full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    msg = _init_weights(full_model)
    print(f'[lock] {msg}')
    _tlog_write('full', msg)
    full_model.summary()

# Part 1 sub-model: single input, two outputs (main + skip1 for part4)
part1_model = Model(inp, [p1_out, s1], name='part1')
part1_model.compile(optimizer='adam', loss='mse')

# Part 2 sub-model: rewire enc layers through a fresh input
_p2i = Input(shape=(4, 4, 16), name='p2_inp')
_s2 = full_model.get_layer('enc_conv3')(_p2i)
_bn = full_model.get_layer('bottleneck')(full_model.get_layer('enc_pool2')(_s2))
part2_model = Model(_p2i, [_bn, _s2], name='part2')
part2_model.compile(optimizer='adam', loss='mse')

# Part 3 sub-model: two inputs (bottleneck + skip2)
_p3m, _p3s2 = Input(shape=(2, 2, 16), name='p3_main_inp'), Input(shape=(4, 4, 16), name='p3_skip2_inp')
_y = full_model.get_layer('skip2_add')([full_model.get_layer('dec_conv1')(full_model.get_layer('dec_up1')(_p3m)), _p3s2])
part3_model = Model([_p3m, _p3s2], _y, name='part3')
part3_model.compile(optimizer='adam', loss='mse')

# Part 4 sub-model: two inputs (p3_out + skip1)
_p4m, _p4s1 = Input(shape=(4, 4, 16), name='p4_main_inp'), Input(shape=(8, 8, 8), name='p4_skip1_inp')
_y = full_model.get_layer('skip1_add')([full_model.get_layer('dec_conv2')(full_model.get_layer('dec_up2')(_p4m)), _p4s1])
_y = full_model.get_layer('dense_out')(full_model.get_layer('dense1')(full_model.get_layer('gap')(_y)))
part4_model = Model([_p4m, _p4s1], _y, name='part4')
part4_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('All 5 Keras models built — weights shared via same layer objects.')

# ── INPUT DATA ────────────────────────────────────────────────────────────────
X_pool = _load_input_pool()
X_full = X_pool[:NUM_QUERIES]

# Derive each partition's input from the preceding sub-model.
print('[derive] Computing partition inputs via Keras predict…')
_bs = min(NUM_QUERIES, 128)
[X_p2_in, X_skip1] = part1_model.predict(X_full, batch_size=_bs)  # p2 feed + skip1→part4
[X_p3_main, X_skip2] = part2_model.predict(X_p2_in, batch_size=_bs)  # p3 feed + skip2→part3
X_p4_main = part3_model.predict([X_p3_main, X_skip2], batch_size=_bs)  # p4 feed
print(
    f'[derive] p2_in={X_p2_in.shape}  skip1={X_skip1.shape}  '
    f'p3_main={X_p3_main.shape}  skip2={X_skip2.shape}  p4_main={X_p4_main.shape}'
)

# ── PER-PARTITION RUN CONFIG ──────────────────────────────────────────────────
# full_run=True  → predict + save results + build stub
# full_run=False → stop after compile (compile error caught)
_PART_CFG = {
    'full': dict(keras_model=full_model, X_input=X_full, input_flat=False, output_flat=False, full_run=True),
    'part1': dict(keras_model=part1_model, X_input=X_full, input_flat=False, output_flat=True, full_run=False),
    'part2': dict(keras_model=part2_model, X_input=X_p2_in, input_flat=True, output_flat=True, full_run=False),
    'part3': dict(keras_model=part3_model, X_input=[X_p3_main, X_skip2], input_flat=True, output_flat=True, full_run=False),
    'part4': dict(keras_model=part4_model, X_input=[X_p4_main, X_skip1], input_flat=True, output_flat=False, full_run=False),
}

# ── DIAGNOSTIC: layer-by-layer HLS bisect ────────────────────────────────────
_PROBE_LAYERS = [
    'enc_conv1',
    'enc_conv2',
    'enc_pool1',
    'enc_conv3',
    'enc_pool2',
    'bottleneck',
    'dec_up1',
    'dec_conv1',
    'skip2_add',
    'dec_up2',
    'dec_conv2',
    'skip1_add',
    'gap',
    'dense1',
    'dense_out',
]


def _diag_hls_bisect(full_keras_model, X_input, base_dir) -> None:
    """Probe each layer with csim to find where HLS signal collapses."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: HLS csim bisect — looking for signal collapse')
    print('=' * 60)
    inp_tensor = full_keras_model.input
    for layer_name in _PROBE_LAYERS:
        try:
            out_tensor = full_keras_model.get_layer(layer_name).output
        except ValueError:
            print(f'  [diag] "{layer_name}" not found — skipping')
            continue

        probe_model = Model(inp_tensor, out_tensor, name=f'probe_{layer_name}')
        y_keras = probe_model.predict(X_input, verbose=0)
        k_min, k_max, k_mean, k_std = (float(f(y_keras)) for f in (np.min, np.max, np.mean, np.std))
        probe_dir = str(Path(base_dir) / f'diag_{layer_name}')
        if os.path.exists(probe_dir):
            shutil.rmtree(probe_dir)

        try:
            hls_p = hls4ml.converters.convert_from_keras_model(
                probe_model,
                hls_config=_make_hls_cfg(probe_model),
                output_dir=probe_dir,
                input_flat=False,
                output_flat=False,
                package_as_xo=False,
                project_name=f'diag_{layer_name}',
                **_HLS_PARAMS,
            )
            hls_p.compile()
            y_hls = hls_p.predict(X_input)
            h_min, h_max, h_mean, h_std = (float(f(y_hls)) for f in (np.min, np.max, np.mean, np.std))
            uniform = h_std < 1e-6 and k_std > 1e-3
            mean_ok = abs(h_mean - k_mean) < 0.5 * (abs(k_mean) + 1e-6)
            status = 'COLLAPSED(uniform)' if uniform else ('OK' if mean_ok else 'COLLAPSED')
        except Exception as exc:
            h_min = h_max = h_mean = float('nan')
            status = f'ERROR({type(exc).__name__})'

        print(
            f'  [{layer_name:12s}]  '
            f'keras=[{k_min:+.4f}, {k_max:+.4f}] mean={k_mean:+.4f}  '
            f'hls=[{h_min:+.4f}, {h_max:+.4f}] mean={h_mean:+.4f}  {status}'
        )
    print('=' * 60 + '\n')


# ── RUN ONE PARTITION ─────────────────────────────────────────────────────────
def run_model(key, keras_model, X_input, input_flat, output_flat, full_run):
    """Convert → plot → compile → (predict + save if full_run)."""
    output_dir = _DIRS[key]
    x_shape = (
        '[' + ', '.join(str(a.shape) for a in X_input) + ']' if isinstance(X_input, (list, tuple)) else str(X_input.shape)
    )
    _tlog_write(key, f'input_flat={input_flat}  output_flat={output_flat}  full_run={full_run}  X.shape={x_shape}\n')

    with timed_step(key, '2. hls4ml config'):
        cfg = _make_hls_cfg(keras_model)

    with timed_step(key, '3. convert_from_keras_model'):
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=cfg,
            output_dir=output_dir,
            input_flat=input_flat,
            output_flat=output_flat,
            package_as_xo=full_run,
            project_name=_PROJECT_NAMES[key],
            **_HLS_PARAMS,
        )

    with timed_step(key, '4. plot_model'):
        hls4ml.utils.plot_model(
            hls_model,
            to_file=os.path.join(output_dir, 'hls4ml_model.png'),
            show_shapes=True,
            show_layer_names=True,
            show_precision=True,
        )
        print(f'  [{key}] model graph → {output_dir}/hls4ml_model.png')

    with timed_step(key, '5. compile (csim bridge)'):
        try:
            hls_model.compile()
            compile_ok = True
        except Exception as exc:
            compile_ok = False
            _tlog_write(key, f'[COMPILE ERROR] {type(exc).__name__}: {exc}')
            print(f'  [{key}] compile raised (caught): {type(exc).__name__}: {exc}')

    if not full_run:
        src = os.path.join(output_dir, 'hls_kernel_config_csim.cfg')
        dst = os.path.join(output_dir, 'hls_kernel_config.cfg')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  [{key}] copied hls_kernel_config_csim.cfg → hls_kernel_config.cfg')
        _tlog_write(key, f'=== [{key}] stopped after compile  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
        _tlogs[key][0].close()
        return hls_model

    # Full run — predict + save
    if compile_ok:
        with timed_step(key, '6. predict (csim)'):
            y_hls = hls_model.predict(X_input)
        with timed_step(key, '7. keras reference predict'):
            y_keras = keras_model.predict(X_input)
        _save_preds(key, output_dir, X_input, y_hls.reshape(y_keras.shape), y_keras)
    else:
        print(f'  [{key}] Skipping predict/save — compile did not succeed.')

    with timed_step(key, '8. build (synth + bitfile)'):
        pass  # hls_model.build(synth=True, bitfile=True, log_to_stdout=True)

    _tlog_write(key, f'=== [{key}] complete  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    _tlogs[key][0].close()
    return hls_model


# ── RUN SELECTED VARIANTS ─────────────────────────────────────────────────────
_LABELS = {
    'full': 'full model        (in=False, out=False)  [FULL RUN]',
    'part1': 'Part 1 Encoder-A  (in=False, out=True )  [STOP AFTER COMPILE]',
    'part2': 'Part 2 Enc-B+BN   (in=True,  out=True )  [STOP AFTER COMPILE]',
    'part3': 'Part 3 Decoder-A  (in=True,  out=True )  [STOP AFTER COMPILE]',
    'part4': 'Part 4 Decoder-B  (in=True,  out=False)  [STOP AFTER COMPILE]',
}

print(f'\nRunning mode: {_ARGS.mode}  → variants: {sorted(_RUN)}')
for _key in ['full', 'part1', 'part2', 'part3', 'part4']:
    if _key not in _RUN:
        continue
    print('\n' + '=' * 60)
    print(f'MODEL {_LABELS[_key]}')
    print('=' * 60)
    _cfg = _PART_CFG[_key]
    run_model(_key, _cfg['keras_model'], _cfg['X_input'], _cfg['input_flat'], _cfg['output_flat'], _cfg['full_run'])
    if _key == 'full' and _ARGS.diag:
        _diag_hls_bisect(full_model, X_full, str(_BASE / 'diag'))

print('\n' + '=' * 60)
print(f'Done.  Lock dir: {_LOCK}')
print('=' * 60)
