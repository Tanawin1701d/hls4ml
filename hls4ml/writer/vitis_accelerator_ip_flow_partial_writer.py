import os
from shutil import copyfile

from hls4ml.writer.vitis_accelerator_ip_flow_writer import VitisWriter

class VitisAcceleratorIPFlowPartialWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.vitis_accelerator_ip_flow_partial_config = None

    def write_axi_wrapper_io(self, inps, outs):
        inputList = []
        outputList = []
        for inp in inps:
            inputList.append(f'hls::stream<dma_data_packet> &par_in_{inp.name}')
        for out in outs:
            outputList.append(f'hls::stream<dma_data_packet> &par_out_{out.name}')

        if len(inputList) == 0 or len(outputList) == 0:
            raise Exception("No input or output stream found")
        newline = ",\n ".join(inputList) + ",\n\n ///output\n " + ", ".join(outputList) + "\n"
        return newline


    def write_axi_wrapper(self, model):
        '''
            We we want to have multi io system
        '''
        inp_axi_t, out_axi_t, inps, outs = self.vitis_accelerator_ip_flow_partial_config.get_corrected_types()
        indent = '    '


        ######################
        # myproject_axi.h
        ######################
        filedir = os.path.dirname(os.path.abspath(__file__))
        f       = open(os.path.join(filedir, '../templates/vitis_accelerator_ip_flow_partial/myproject_axi.h'))
        fout    = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}_axi.h', 'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif '// hls-fpga-machine-learning insert include' in line:
                newline = f'#include "{model.config.get_project_name()}.h"\n'
                newline += '#include "ap_axi_sdata.h"\n'
            elif 'myproject' in line:
                newline =  line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert definitions' in line:

                ##### make input
                newline = ''
                inputSizeStr = "{ " + ", ".join([str(inp.size()) for inp in inps]) +  " }"
                newline += f'static const unsigned N_IN  [{len(inps)}] = {inputSizeStr};\n'

                ##### make output
                outputSizeStr = "{ " + ", ".join([str(out.size()) for out in outs]) +  " }"
                newline += f'static const unsigned N_OUT [{len(outs)}] = {outputSizeStr};\n'
                if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
                    newline += 'typedef hls::axis<float, 0, 0, 0> dma_data_packet;\n'
                else:
                    newline += f'typedef {inp_axi_t} input_axi_t;\n'
                    newline += f'typedef {out_axi_t} output_axi_t;\n'
            elif '// hls-fpga-machine-learning insert multi-io' in line:
                newline = ''
                if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
                    newline += self.write_axi_wrapper_io(inps, outs)
                else:
                    raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream")

            else:
                newline = line

            #### TODO add stream

            fout.write(newline)
        f.close()
        fout.close()

        ######################
        # myproject_axi.cpp
        ######################
        f     = open(os.path.join(filedir, '../templates/vitis_accelerator_ip_flow_partial/myproject_axi.cpp'))
        fout  = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}_axi.cpp', 'w')

        io_type = model.config.get_config_value("IOType")

        for line in f.readlines():
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert include' in line:
                newline = f'#include "{model.config.get_project_name()}_axi.h"\n'
            elif '// hls-fpga-machine-learning insert multiIo' in line:
                newline = ''
                if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
                    newline += self.write_axi_wrapper_io(inps, outs)
                else:
                    raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream")
            elif '// hls-fpga-machine-learning insert local vars' in line:
                pass
            elif '// hls-fpga-machine-learning insert call' in line:
                pass
            elif '// hls-fpga-machine-learning insert interface' in line:
                pass
            elif '// hls-fpga-machine-learning insert enqueue' in line:
                pass
            elif '// hls-fpga-machine-learning insert dequeue' in line:
                pass
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_board_script(self, model):
        print("[partial reconfig] we are not supporting write_board_script this yet")

    def write_driver(self, model):
        print("[partial reconfig] we are not supporting write_driver this yet")

    def write_wrapper_test(self, model):
        print("[partial reconfig] we are not supporting write_wrapper_test this yet")

    def modify_build_script(self, model):
        '''
        Modify the build_prj.tcl and build_lib.sh scripts to add the extra wrapper files and set the top function
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        oldfile = f'{model.config.get_output_dir()}/build_prj.tcl'
        newfile = f'{model.config.get_output_dir()}/build_prj_axi.tcl'
        f = open(oldfile)
        fout = open(newfile, 'w')

        for line in f.readlines():
            if 'set_top' in line:
                newline = line[:-1] + '_axi\n'  # remove the newline from the line end and append _axi for the new top
                newline += f'add_files firmware/{model.config.get_project_name()}_axi.cpp -cflags "-std=c++0x"\n'
            elif f'{model.config.get_project_name()}_cosim' in line:
                newline = line.replace(
                    f'{model.config.get_project_name()}_cosim',
                    f'{model.config.get_project_name()}_axi_cosim',
                )
            elif '${project_name}.tcl' in line:
                newline = line.replace('${project_name}.tcl', '${project_name}_axi.tcl')
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir, '../templates/vitis_accelerator_ip_flow/build_lib.sh'))
        fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_new_tar(self, model):
        super().write_tar(model)

    def write_hls(self, model, is_multigraph=False):

        from hls4ml.backends import VitisACIPFlowParConfig

        self.vitis_accelerator_ip_flow_partial_config = VitisACIPFlowParConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )
        super().write_hls(model, is_multigraph=is_multigraph)
        if not is_multigraph:
            self.write_board_script(model)
            self.write_driver(model)
            self.write_wrapper_test(model)
            self.write_axi_wrapper(model)
            self.modify_build_script(model)
            self.write_new_tar(model)
