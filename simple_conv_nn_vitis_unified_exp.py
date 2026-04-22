"""
Three-variant Conv NN experiment: full, first-half, second-half.
All variants share weights via the same Keras layer objects.

Input pool of MAX_QUERIES samples is generated once and locked;
subsequent runs reuse the same data for reproducibility.

MODEL FLOW
----------
  full        : all steps through build (build is commented out)
  first_half  : convert → plot → compile (error caught) → stop
  second_half : convert → plot → compile (error caught) → stop

WEIGHT / INPUT LOCKING
-----------------------
  Weights and input pool are locked independently in _exp_locked/:
    full_weights.h5          — re-used as long as file exists
    x_input_{MAX_QUERIES}.npy — re-used if pool size hasn't changed

CONFIG
------
  Edit NUM_QUERIES and MAX_QUERIES in the USER CONFIG block below.
"""

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from keras.src.layers import GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model

import hls4ml

# ===========================================================================
# USER CONFIG
# ===========================================================================
NUM_QUERIES = 10  # samples to actually run this build (must be <= MAX_QUERIES)
MAX_QUERIES = 1_000_000  # total size of the locked input pool (~256 MB on disk)

# ===========================================================================
# Paths
# ===========================================================================
_BASE = Path(__file__).parent / 'hls4ml_output'
_LOCK = _BASE / '_exp_locked'

_WEIGHTS_FILE = _LOCK / 'full_weights.h5'
_INPUT_FILE = _LOCK / f'x_input_{MAX_QUERIES}.npy'

_DIRS = {
    'full': str(_BASE / 'hls4mlprj_exp_full'),
    'first_half': str(_BASE / 'hls4mlprj_exp_first_half'),
    'second_half': str(_BASE / 'hls4mlprj_exp_second_half'),
}

# ===========================================================================
# Setup: wipe HLS output dirs on every run; never wipe lock dir
# ===========================================================================
assert NUM_QUERIES <= MAX_QUERIES, f'NUM_QUERIES ({NUM_QUERIES}) > MAX_QUERIES ({MAX_QUERIES})'

_LOCK.mkdir(parents=True, exist_ok=True)
for _d in _DIRS.values():
    if os.path.exists(_d):
        shutil.rmtree(_d)
    os.makedirs(_d)

# ===========================================================================
# Timing utilities — one log file per model output dir
# ===========================================================================
_tlogs: dict = {}


def _open_tlog(key: str) -> None:
    log_path = Path(_DIRS[key]) / 'timing_log.txt'
    f = open(log_path, 'w', buffering=1)
    _tlogs[key] = (f, log_path)
    _tlog_write(key, f'=== [{key}] timing log  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    _tlog_write(key, f'NUM_QUERIES={NUM_QUERIES}  MAX_QUERIES={MAX_QUERIES}')
    _tlog_write(key, f'weights_locked={_WEIGHTS_FILE.exists()}  input_locked={_INPUT_FILE.exists()}')
    _tlog_write(key, '')


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

# ===========================================================================
# Xilinx tool-chain paths
# ===========================================================================
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2023.2'
os.environ['XILINX_VIVADO'] = '/tools/Xilinx/Vivado/2023.2'
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

# ===========================================================================
# 1. Build full Keras model with named layers
#    First-half and second-half sub-models are derived from the SAME layer
#    objects, so all three models share weights automatically.
# ===========================================================================
print('\n' + '=' * 60)
print('Building Keras models...')
print('=' * 60)

with timed_step('full', '1. Keras model definition'):
    inp = Input(shape=(8, 8, 1), name='inp')

    # -------- Encoder --------
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='enc_conv1')(inp)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', name='enc_conv2')(x)
    x = MaxPooling2D((2, 2), name='enc_pool1')(x)  # 8 → 4

    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='enc_conv3')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='enc_conv4')(x)
    x = MaxPooling2D((2, 2), name='enc_pool2')(x)  # 4 → 2

    # -------- Bottleneck  (split point between first and second half) --------
    bottleneck_out = Conv2D(64, (3, 3), padding='same', activation='relu', name='bottleneck')(x)  # (2, 2, 64)

    # -------- Decoder --------
    y = UpSampling2D((2, 2), name='dec_up1')(bottleneck_out)  # 2 → 4
    y = Conv2D(32, (3, 3), padding='same', activation='relu', name='dec_conv1')(y)
    y = UpSampling2D((2, 2), name='dec_up2')(y)  # 4 → 8
    y = Conv2D(16, (3, 3), padding='same', activation='relu', name='dec_conv2')(y)

    # -------- Head --------
    y = GlobalAveragePooling2D(name='gap')(y)
    y = Dense(64, activation='relu', name='dense1')(y)
    full_out = Dense(4, activation='softmax', name='dense_out')(y)

    full_model = Model(inp, full_out, name='full_model')
    full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if _WEIGHTS_FILE.exists():
        full_model.load_weights(str(_WEIGHTS_FILE))
        msg = f'Weights loaded ← {_WEIGHTS_FILE}'
    else:
        full_model.save_weights(str(_WEIGHTS_FILE))
        msg = f'Weights saved  → {_WEIGHTS_FILE}'
    print(f'[lock] {msg}')
    _tlog_write('full', msg)

    full_model.summary()

# First-half sub-model: input → bottleneck output (same layer objects → shared weights)
first_half_model = Model(inp, bottleneck_out, name='first_half')
first_half_model.compile(optimizer='adam', loss='mse')

# Second-half sub-model: re-wire decoder/head layers through a fresh input tensor.
# Calling an existing layer with a new tensor creates a new graph node but
# reuses the same weight tensors — standard Keras weight-sharing pattern.
sec_inp = Input(shape=(2, 2, 64), name='sec_inp')
z = full_model.get_layer('dec_up1')(sec_inp)
z = full_model.get_layer('dec_conv1')(z)
z = full_model.get_layer('dec_up2')(z)
z = full_model.get_layer('dec_conv2')(z)
z = full_model.get_layer('gap')(z)
z = full_model.get_layer('dense1')(z)
z = full_model.get_layer('dense_out')(z)
second_half_model = Model(sec_inp, z, name='second_half')
second_half_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('All three Keras models built — weights shared via same layer objects.')

# ===========================================================================
# 2. Input data
#    Full input pool is generated once and locked; slice to NUM_QUERIES each run.
#    Second-half input is always derived from the first-half Keras model so it
#    stays consistent with the locked full-model weights.
# ===========================================================================
if _INPUT_FILE.exists():
    print(f'[lock] Loading input pool ← {_INPUT_FILE}')
    X_pool = np.load(str(_INPUT_FILE))
    print(f'[lock] Loaded {X_pool.shape[0]:,} samples — using first {NUM_QUERIES:,}')
    _tlog_write('full', f'Input pool loaded ← {_INPUT_FILE}')
else:
    print(f'[lock] Generating {MAX_QUERIES:,} input samples (seed=42)…')
    np.random.seed(42)
    X_pool = np.random.rand(MAX_QUERIES, 8, 8, 1).astype(np.float32)
    np.save(str(_INPUT_FILE), X_pool)
    sz_mb = _INPUT_FILE.stat().st_size / 1e6
    print(f'[lock] Input pool saved  → {_INPUT_FILE}  ({sz_mb:.1f} MB)')
    _tlog_write('full', f'Input pool saved → {_INPUT_FILE}  ({sz_mb:.1f} MB)')

X_full = X_pool[:NUM_QUERIES]

print('[derive] Computing second-half input via first_half Keras predict…')
X_second_half = first_half_model.predict(X_full, batch_size=min(NUM_QUERIES, 128))
print(f'[derive] X_second_half.shape = {X_second_half.shape}')

# ===========================================================================
# Shared HLS conversion parameters
# ===========================================================================
_HLS_PARAMS = dict(
    backend='VitisUnified',
    io_type='io_stream',
    board='kv260',
    part='xck26-sfvc784-2LV-c',
    clock_period='10ns',
    input_type='float',
    output_type='float',
    axi_mode='axi_stream',
    project_name='my_project_2',
)


# ===========================================================================
# Helper: run one model through its designated hls4ml steps
# ===========================================================================
def run_model(key, keras_model, X_input, input_flat, output_flat, full_run):
    """
    full_run=True  — all steps; predict + save results; build commented out
    full_run=False — convert + plot + compile only; compile error is caught
    """
    output_dir = _DIRS[key]
    _tlog_write(
        key, f'input_flat={input_flat}  output_flat={output_flat}  full_run={full_run}  X_input.shape={X_input.shape}\n'
    )

    # ---- step 2: hls4ml config ----
    with timed_step(key, '2. hls4ml config'):
        cfg = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
        cfg['Model']['Strategy'] = 'resource'
        cfg['Model']['ReuseFactor'] = 8
        cfg['Model']['Precision'] = 'ap_fixed<4,2>'

    # ---- step 3: convert ----
    with timed_step(key, '3. convert_from_keras_model'):
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=cfg,
            output_dir=output_dir,
            input_flat=input_flat,
            output_flat=output_flat,
            **_HLS_PARAMS,
        )

    # ---- step 4: plot ----
    with timed_step(key, '4. plot_model'):
        hls4ml.utils.plot_model(
            hls_model,
            to_file=os.path.join(output_dir, 'hls4ml_model.png'),
            show_shapes=True,
            show_layer_names=True,
            show_precision=True,
        )
        print(f'  [{key}] model graph → {output_dir}/hls4ml_model.png')

    # ---- step 5: compile (error caught for wrapper-only models) ----
    with timed_step(key, '5. compile (csim bridge)'):
        try:
            hls_model.compile()
            compile_ok = True
        except Exception as exc:
            compile_ok = False
            _tlog_write(key, f'[COMPILE ERROR] {type(exc).__name__}: {exc}')
            print(f'  [{key}] compile raised (caught): {type(exc).__name__}: {exc}')

    if not full_run:
        _tlog_write(key, f'=== [{key}] stopped after compile  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
        _tlogs[key][0].close()
        return hls_model

    # ---- steps below: full run only ----
    if compile_ok:
        with timed_step(key, '6. predict (csim)'):
            y_hls = hls_model.predict(X_input)

        with timed_step(key, '7. keras reference predict'):
            y_keras = keras_model.predict(X_input)

        y_hls_r = y_hls.reshape(y_keras.shape)
        diff = float(np.max(np.abs(y_hls_r - y_keras)))
        print(f'  [{key}] Max |HLS − Keras| = {diff:.6f}')
        _tlog_write(key, f'Max |HLS - Keras| = {diff:.6f}')

        np.save(os.path.join(output_dir, 'x_input.npy'), X_input)
        np.save(os.path.join(output_dir, 'y_pred_hls.npy'), y_hls_r)
        np.save(os.path.join(output_dir, 'y_pred_keras.npy'), y_keras)
        print(f'  [{key}] Results saved to {output_dir}')
    else:
        print(f'  [{key}] Skipping predict/save — compile did not succeed.')

    with timed_step(key, '8. build (synth + bitfile)'):
        pass  # hls_model.build(synth=True, bitfile=True, log_to_stdout=True)

    _tlog_write(key, f'=== [{key}] complete  {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    _tlogs[key][0].close()
    return hls_model


# ===========================================================================
# Run all three models
# ===========================================================================
print('\n' + '=' * 60)
print('MODEL 1/3: full  (input_flat=False, output_flat=False)  [FULL RUN]')
print('=' * 60)
run_model('full', full_model, X_full, input_flat=False, output_flat=False, full_run=True)

print('\n' + '=' * 60)
print('MODEL 2/3: first_half  (input_flat=False, output_flat=True)  [STOP AFTER COMPILE]')
print('=' * 60)
run_model('first_half', first_half_model, X_full, input_flat=False, output_flat=True, full_run=False)

print('\n' + '=' * 60)
print('MODEL 3/3: second_half  (input_flat=True, output_flat=False)  [STOP AFTER COMPILE]')
print('=' * 60)
run_model('second_half', second_half_model, X_second_half, input_flat=True, output_flat=False, full_run=False)

print('\n' + '=' * 60)
print(f'Done.  Lock dir: {_LOCK}')
print('=' * 60)
