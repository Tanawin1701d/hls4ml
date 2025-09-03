import os

from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta
from hls4ml.writer.vitis_unified_writer.wrap_gen import VitisUnified_WrapperGen


class VitisUnifiedPartial_WrapperGen(VitisUnified_WrapperGen):

    @classmethod
    def gen_io_str(self, mg, indent, inp_gmem_t, out_gmem_t, inps, outs, meta=None):
        inputStreamList = []
        outputStreamList = []

        for inp_idx, inp in enumerate(inps):
            inp_type = mg.get_dma_type_name()
            if (meta.vitis_unified_config.is_free_interim_input()) and (meta.vitis_unified_config.get_graph_idx() != 0):
                inp_type = mg.get_axi_wrapper_type(inp)
            inputStreamList.append(f"{indent} hls::stream<{inp_type}>& {mg.get_io_port_name(inp, True, inp_idx)}")

        for out_idx, out in enumerate(outs):
            out_type = mg.get_dma_type_name()
            if (meta.vitis_unified_config.is_free_interim_output()) and (
                meta.vitis_unified_config.get_graph_idx() != (meta.vitis_unified_config.get_amt_graph() - 1)
            ):
                out_type = mg.get_axi_wrapper_type(out)
            outputStreamList.append(f"{indent} hls::stream<{out_type}>& {mg.get_io_port_name(out, False, out_idx)}")

        return ", \n".join(inputStreamList) + ",\n" + ", \n".join(outputStreamList)

    @classmethod
    def write_wrapper(self, meta: VitisUnifiedWriterMeta, model, mg):

        inp_axis_t, out_axis_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
        indent = '      '

        # start write myproject_axi.cpp

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified_partial/myproject_axi.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_wrapper_file_name(model)}.cpp', 'w')

        for line in fin.readlines():

            # if "MY_PROJECT_AXI_INC" in line:
            #     line = line.replace("MY_PROJECT_AXI_INC", mg.get_main_wrapper_file_name(model))
            if "MY_PROJECT_TOP_FUNC" in line:
                line = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model))
            elif "WRAPPER_FILE_NAME" in line:
                line = line.replace("WRAPPER_FILE_NAME", mg.get_wrapper_file_name(model))
            elif "// hls-fpga-machine-learning insert multi-io" in line:
                line = self.gen_io_str(mg, indent, inp_axis_t, out_axis_t, inps, outs, meta) + "\n"
            elif "// hls-fpga-machine-learning insert interface" in line:
                for inp_idx, inp in enumerate(inps):
                    line += f"{indent} #pragma HLS INTERFACE axis port={mg.get_io_port_name(inp, True, inp_idx)}\n"
                for out_idx, out in enumerate(outs):
                    line += f"{indent} #pragma HLS INTERFACE axis port={mg.get_io_port_name(out, False, out_idx)}\n"
            elif "// hls-fpga-machine-learning insert local vars" in line:
                # declare stream variable
                for inp_idx, inp in enumerate(inps):
                    line += f"{indent} static hls::stream<{inp.type.name}> {mg.get_local_stream_name(inp, True, inp_idx)};\n"
                for out_idx, out in enumerate(outs):
                    line += (
                        f"{indent} static hls::stream<{out.type.name}> {mg.get_local_stream_name(out, False, out_idx)};\n"
                    )
                # declare stream size
                for inp_idx, inp in enumerate(inps):
                    line += (
                        f"#pragma HLS STREAM variable={mg.get_local_stream_name(inp, True, inp_idx)} depth={inp.pragma[1]}\n"
                    )
                for out_idx, out in enumerate(outs):
                    line += (
                        f"#pragma HLS STREAM variable={mg.get_local_stream_name(out, False, out_idx)} "
                        f"depth={out.pragma[1]}\n"
                    )

            elif "// hls-fpga-machine-learning insert isLast vars" in line:
                for inp_idx in range(len(inps)):
                    line += f"bool {mg.get_is_last_var(inp_idx)} = false;\n"

            elif "// hls-fpga-machine-learning insert enqueue" in line:
                for inp_idx, inp in enumerate(inps):
                    if meta.vitis_unified_config.is_free_interim_input():
                        line += mg.get_enqueue_func_stream2rstream(inp, inp_idx)
                    else:
                        line += mg.get_enqueue_func_atom2stream(inp, inp_idx)
                    line += "\n"

            elif "// hls-fpga-machine-learning insert call" in line:
                poolList = []
                for inp_idx, inp in enumerate(inps):
                    poolList.append(f"{mg.get_local_stream_name(inp, True, inp_idx)}")
                for out_idx, out in enumerate(outs):
                    poolList.append(f"{mg.get_local_stream_name(out, False, out_idx)}")
                joinedIo = ", \n".join(poolList)
                line += f"{indent} {mg.get_top_model_name(model)}({joinedIo});\n"

            elif "// hls-fpga-machine-learning insert dequeue" in line:
                for out_idx, out in enumerate(outs):
                    if meta.vitis_unified_config.is_free_interim_output():
                        line += mg.get_dequeue_func_rstream2stream(out, out_idx, mg.get_all_last_logic(len(inps))) + "\n"
                    else:
                        line += (
                            mg.get_dequeue_func_rstream2atom(out, out_idx, mg.get_all_last_logic(len(inps)), out_axis_t)
                            + "\n"
                        )

            fout.write(line)
        fin.close()
        fout.close()

        # start write myproject_axi.h

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified_partial/myproject_axi.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_wrapper_file_name(model)}.h', 'w')

        for line in fin.readlines():

            newline = line
            if "FILENAME" in line:
                newline = line.replace("FILENAME", mg.get_wrapper_file_name(model).upper())
            if "MY_PROJECT_TOP_FUNC" in line:
                newline = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model))
            elif "MY_PROJECT_AXI_INC" in line:
                newline = line.replace("MY_PROJECT_AXI_INC", mg.get_main_file_name(model))
            elif "// hls-fpga-machine-learning insert definitions" in line:
                # make input
                newline = ''
                inputSizeStr = "{ " + ", ".join([str(inp.size()) for inp in inps]) + " }"
                newline += f'constexpr unsigned {mg.get_input_size_arr_name(model)}  [{len(inps)}] = {inputSizeStr};\n'

                # make output
                outputSizeStr = "{ " + ", ".join([str(out.size()) for out in outs]) + " }"
                newline += f'constexpr unsigned {mg.get_output_size_arr_name(model)} [{len(outs)}] = {outputSizeStr};\n'
                newline += f'typedef hls::axis<{inp_axis_t}, 0, 0, 0, AXIS_ENABLE_LAST> dma_data_packet;\n'
                # incase the io is interim input
                if meta.vitis_unified_config.is_free_interim_input():
                    for inp in inps:
                        newline += mg.get_axi_wrapper_dec(inp) + "\n"
                # incase the io is interim output
                if meta.vitis_unified_config.is_free_interim_output():
                    for out in outs:
                        newline += mg.get_axi_wrapper_dec(out) + "\n"
            elif "// vitis-unified-wrapper-io" in line:
                newline = self.gen_io_str(mg, indent, inp_axis_t, out_axis_t, inps, outs, meta) + "\n"
            fout.write(newline)

        fin.close()
        fout.close()
