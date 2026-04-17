"""
First-half model: Encoder + Bottleneck
Input:  (8, 8, 1)
Output: (2, 2, 64)  — bottleneck feature map fed into the second-half model
"""

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D
from tensorflow.keras.models import Model

import hls4ml

# ---------------------------------------------------------------------------
# Output directory (defined early so the timing log can live inside it)
# ---------------------------------------------------------------------------
output_dir = str(Path(__file__).parent / 'hls4ml_output' / 'hls4mlprj_first_half_vitis_unified')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# ---------------------------------------------------------------------------
# Crash-safe timing logger
# ---------------------------------------------------------------------------
_TIMING_LOG = Path(output_dir) / 'timing_log.txt'
_tlog = open(_TIMING_LOG, 'w', buffering=1)  # line-buffered → each line flushed immediately


def _tlog_write(line: str) -> None:
    """Write a line and fsync so it survives a crash."""
    _tlog.write(line + '\n')
    os.fsync(_tlog.fileno())


_tlog_write(f'=== Timing log started at {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
_tlog_write(f'Log file: {_TIMING_LOG}')
_tlog_write('')


@contextmanager
def timed_step(name: str):
    """Context manager that times a block and appends the result to the log."""
    _tlog_write(f'[START] {name}  ({time.strftime("%H:%M:%S")})')
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _tlog_write(f'[END]   {name}  elapsed={elapsed:.3f}s')
        _tlog_write('')
        print(f'[timing] {name}: {elapsed:.3f}s')


# ---------------------------------------------------------------------------
# Xilinx tool-chain paths
# ---------------------------------------------------------------------------
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2023.2'
os.environ['XILINX_VIVADO'] = '/tools/Xilinx/Vivado/2023.2'
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

# ---------------------------------------------------------------------------
# Model definition — encoder + bottleneck
# Input  (8, 8, 1)  →  Output (2, 2, 64)
# ---------------------------------------------------------------------------
with timed_step('1. Keras model definition (first half)'):
    inputs = Input(shape=(8, 8, 1))

    # -------- Encoder --------
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)  # 8 -> 4

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)  # 4 -> 2

    # Bottleneck
    outputs = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # (2, 2, 64)

    model = Model(inputs, outputs, name='first_half')
    model.compile(optimizer='adam', loss='mse')
    model.summary()

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
np.random.seed(42)
X_input = np.random.rand(BATCH_SIZE, 8, 8, 1).astype(np.float32)

# ---------------------------------------------------------------------------
# hls4ml config
# ---------------------------------------------------------------------------
with timed_step('2. hls4ml config'):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'resource'
    config['Model']['ReuseFactor'] = 8
    config['Model']['Precision'] = 'ap_fixed<4,2>'

# ---------------------------------------------------------------------------
# VitisUnified conversion
# ---------------------------------------------------------------------------
with timed_step('3. convert_from_keras_model'):
    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend='VitisUnified',
        io_type='io_stream',
        board='kv260',
        part='xck26-sfvc784-2LV-c',
        clock_period='10ns',
        input_type='float',
        output_type='float',
        axi_mode='axi_stream',
    )

# ---------------------------------------------------------------------------
# Model visualization
# ---------------------------------------------------------------------------
with timed_step('4. plot_model'):
    hls4ml.utils.plot_model(
        vitis_unified_model,
        to_file=os.path.join(output_dir, 'hls4ml_model.png'),
        show_shapes=True,
        show_layer_names=True,
        show_precision=True,
    )
    print(f'Model graph saved to {output_dir}/hls4ml_model.png')

# ---------------------------------------------------------------------------
# C-simulation
# ---------------------------------------------------------------------------
with timed_step('5. vitis_unified_model.compile (csim bridge)'):
    print('\n--- Compiling HLS model (csim bridge) ---')
    vitis_unified_model.compile()

with timed_step('6. vitis_unified_model.predict (csim)'):
    print('\n--- Running prediction via csim bridge ---')
    y_hls = vitis_unified_model.predict(X_input)

with timed_step('7. keras model.predict (reference)'):
    print('\n--- Keras reference prediction ---')
    y_keras = model.predict(X_input)

y_hls = y_hls.reshape(y_keras.shape)  # HLS bridge flattens outputs; restore spatial dims
print('\nMax absolute difference (HLS vs Keras):', np.max(np.abs(y_hls - y_keras)))

# Save results — y_hls is the bottleneck input for the second-half model
np.save(os.path.join(output_dir, 'x_input.npy'), X_input)
np.save(os.path.join(output_dir, 'y_pred_hls.npy'), y_hls)
np.save(os.path.join(output_dir, 'y_pred_keras.npy'), y_keras)
print(f'\nSaved inputs/outputs to {output_dir}')

# ---------------------------------------------------------------------------
# HLS synthesis + bitfile generation
# ---------------------------------------------------------------------------
with timed_step('8. vitis_unified_model.build (synth + bitfile)'):
    print('\n--- Building bitfile (synthesis + implementation) ---')
    vitis_unified_model.build(synth=True, bitfile=True, log_to_stdout=True)

_tlog_write(f'=== All steps complete at {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
_tlog.close()
print(f'\nTiming log written to {_TIMING_LOG}')
