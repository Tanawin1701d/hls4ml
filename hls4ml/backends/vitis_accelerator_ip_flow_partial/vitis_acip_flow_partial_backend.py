import os

from hls4ml.backends import VitisAcceleratorIPFlowBackend, VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report


class VitisAcceleratorIPFlowPartialBackend(VitisAcceleratorIPFlowBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisAcceleratorIPFlowPartial')
        self._register_layer_attributes()
        self._register_flows()


    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        log_to_stdout=True
    ):
        ##### it builds and return vivado reports
        return super().build(
            model,
            reset=reset,
            csim=csim,
            synth=synth,
            cosim=cosim,
            validation=validation,
            export=export,
            vsynth=vsynth,
            fifo_opt=fifo_opt,
            log_to_stdout=log_to_stdout,
        )

    def create_initial_config(
        self,
        board='pynq-z2',
        part=None,
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        interface='axi_stream',
        driver='python',
        input_type='float',
        output_type='float',
    ):
        return super().create_initial_config(
        board             = board,
        part              = part,
        clock_period      = clock_period,
        clock_uncertainty = clock_uncertainty,
        io_type           = io_type,
        interface         = interface,
        driver            = driver,
        input_type        = input_type,
        output_type       = output_type
        )


    def _register_flows(self):
        vitis_ip = 'vitis:ip'
        writer_passes = ['make_stamp', 'vitisacceleratoripflowpartial:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)
        self._default_flow = vitis_ip