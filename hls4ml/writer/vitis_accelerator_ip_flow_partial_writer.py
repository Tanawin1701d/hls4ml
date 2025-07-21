import os
from shutil import copyfile

from hls4ml.writer.vitis_accelerator_ip_flow_writer import VitisWriter

class VitisAcceleratorIPFlowPartialWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.vitis_accelerator_ip_flow_partial_config = None

    #######################################################
    ## naming of variable function helper #################
    #######################################################

    def getDmaTypeName(self):
        return "dma_data_packet"

    def getWrapperPortName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}"

    def getTopModelName(self, model):
        return f"{model.config.get_project_name()}_axi"

    ########################################################
    ## axi_wrapper.h & axi_wrapper.cpp  function helper ####
    ########################################################
    ##### variable
    def getWrapperPortNameLocal(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_local"

    def getWrapperTmpName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_tmp"

    def getWrapperIsLastCnt(self, idx):
        return f"isLastCnt_{str(idx)}"
    ##### io
    def write_axi_wrapper_io(self, inps, outs):
        inputList = []
        outputList = []
        for inp in inps:
            inputList.append(f'hls::stream<dma_data_packet>& {self.getWrapperPortName(inp, True)}')
        for out in outs:
            outputList.append(f'hls::stream<dma_data_packet> & {self.getWrapperPortName(out, False)}')

        if len(inputList) == 0 or len(outputList) == 0:
            raise Exception("No input or output stream found")
        newline = "/////// inputs\n" +  ",\n ".join(inputList) + ",\n\n ///outputs\n " + ", ".join(outputList) + "\n"
        return newline
    ##### content in axi_wrapper.cpp
    def write_axi_wrapper_interface(self, model, inps, outs):
        if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
            newline = ''
            indent = "      "
            for inp in inps:
                portname = self.getWrapperPortName(inp, True)
                newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
            for out in outs:
                portname = self.getWrapperPortName(out, False)
                newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
            if model.config.get_config_value("IOType") == 'io_stream':
                    newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                    newline += indent + '#pragma HLS DATAFLOW\n'
            return newline
        else:
            raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream @ interface retriever")

    def write_axi_local_vars(self, model, inps, outs):

        ####### build local stream variable

        newline = '///// wrinting local stream vars /////\n'
        if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
            indent = "      "
            ##### loop to build local stream to send data into the system
            newline += '///////// build input vars ///////////\n'
            for idx, inp in enumerate(inps):
                newline += f"    bool {self.getWrapperIsLastCnt(idx)} = false;\n"
                portname = self.getWrapperPortNameLocal(inp, True)
                newline += indent + f'hls::stream<{inp.type.name}> {portname}("{portname}");\n'
            newline += '///////// build output vars ///////////\n'
            for out in outs:
                portname = self.getWrapperPortNameLocal(out, False)
                newline += indent + f'hls::stream<{out.type.name}> {portname}("{portname}");\n'

        ####### set stream DEPTH

            newline += '///////// set the stream depth ///////////\n'
            ##### loop to set depth

            for inpIdx, inp in enumerate(inps):
                portname = self.getWrapperPortNameLocal(inp, True)
                newline += indent + f'#pragma HLS STREAM variable={portname} depth={inps[inpIdx].pragma[1]}\n'
            for outIdx, out in enumerate(outs):
                portname = self.getWrapperPortNameLocal(out, False)
                newline += indent + f'#pragma HLS STREAM variable={portname} depth={model.get_output_variables()[outIdx].pragma[1]}\n'

        else:
            raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream @ local vars")


        return newline

    def write_axi_wrapper_each_enqueue(self, model, inps, idx):

        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// enqueue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'dma_data_packet {self.getWrapperTmpName(inps[idx], True)};\n'
            ### newline += indent + f'{inps[idx].type.name}\n'
            newline += indent + 'for(unsigned i = 0; i < N_IN[' +str(idx) +']/' + inps[idx].type.name + '::size; ++i){\n'
            newline += indent + indent + inps[idx].type.name + ' ctype;\n'
            newline += indent + indent + 'for(unsigned j = 0; j < '+ inps[idx].type.name + '::size; ++j){\n'
            if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
                newline += indent + indent + indent + self.getWrapperPortName(inps[idx], True) + f'.read({self.getWrapperTmpName(inps[0], True)});\n'
                newline += indent + indent + indent + "ctype[j] = " + self.getWrapperTmpName(inps[idx], True) + ".data;\n"
                newline += indent + indent + indent + self.getWrapperIsLastCnt(idx) + " = " + self.getWrapperTmpName(inps[idx], True) + ".last;\n"
            else:
                raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream @ each enqueue")

            newline += indent + indent + '}\n'
            newline += indent + indent + self.getWrapperPortNameLocal(inps[idx], True) + ".write(ctype);\n"
            newline += indent + '}\n'
            newline += indent + self.getWrapperTmpName(inps[idx], True) + ".last = 0;\n"

        else:
            raise Exception("vitis_accelerator_ip_flow_partial supports only io_stream @ each enqueue")

        return newline

    def write_axi_wrapper_dequeue(self, model, inputs, outs, idx, out_axi_t):

        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// dequeue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'dma_data_packet {self.getWrapperTmpName(outs[idx], False)} = {self.getWrapperTmpName(inputs[0], True)};\n'
            ####### the tmp must copy from input to prevent dma get stuck
            newline += indent + 'for(unsigned i = 0; i < N_OUT[' +str(idx) +']/' + outs[idx].type.name + '::size; ++i){\n'
            newline += indent + indent + outs[idx].type.name + ' ctype = ' + self.getWrapperPortNameLocal(outs[idx], False) + '.read();\n'
            newline += indent + indent + 'for(unsigned j = 0; j < ' + outs[idx].type.name + '::size; ++j){\n'
            if self.vitis_accelerator_ip_flow_partial_config.get_interface() == 'axi_stream':
                newline += indent + indent + indent + self.getWrapperTmpName(outs[idx], False) + f'.data = ({out_axi_t}) (ctype[j]);\n'
                poolLastCondition = " & ".join([self.getWrapperIsLastCnt(condIdx) for condIdx  in range(len(inputs))])
                newline += indent + indent + indent + f"if({poolLastCondition}){{\n"
                newline += indent + indent + indent + indent + self.getWrapperTmpName(outs[idx], False) + f".last = (((i+1)*(j+1))==N_OUT[{str(idx)}]);\n"
                newline += indent + indent + indent + "}\n"
                newline += indent + indent + indent + self.getWrapperPortName(outs[idx], False) + f'.write({self.getWrapperTmpName(outs[idx], False)});\n'
                newline += indent + indent + "}\n"
                newline += indent + "}\n"
                newline += indent + self.getWrapperTmpName(outs[idx], False) + ".last = 0;\n"
            else:
                raise Exception("vitis_accelerator_ip_flow_partial supports only axi_stream @ each dequeue")
        else:
            raise Exception("vitis_accelerator_ip_flow_partial supports only io_stream @ each dequeue")

        return newline

    def write_axi_wrapper_insert_call(self, model, inps, outs):
        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = indent + f'{model.config.get_project_name()}' + "("
        inputList = []
        outputList = []
        for inp in inps:
            inputList.append(self.getWrapperPortNameLocal(inp, True))
        for out in outs:
            outputList.append(self.getWrapperPortNameLocal(out, False))
        newline += ", ".join(inputList) + ", " + ", ".join(outputList) + ");\n"
        return newline

    ##### main function

    def write_axi_wrapper(self, model):
        '''
            We we want to have multi io system
        '''
        inp_axi_t, out_axi_t, inps, outs = self.vitis_accelerator_ip_flow_partial_config.get_corrected_types()
        indent = '    '

        print("------------------------------- input write wrapper is -------------------------")
        print([inp.name for inp in inps])
        print(model.inputs)
        print("------------------------------- output write wrapper is -------------------------")
        print([out.name for out in outs])
        print(model.outputs)
        print("-----------------------------------------------------------------------------------")

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
            elif '// hls-fpga-machine-learning insert interface' in line:
                newline = self.write_axi_wrapper_interface(model, inps, outs)
            elif '// hls-fpga-machine-learning insert local vars' in line:
                newline = self.write_axi_local_vars(model, inps, outs)
            elif '// hls-fpga-machine-learning insert enqueue' in line:
                newline = ''
                for idx, inp in enumerate(inps):
                    newline += self.write_axi_wrapper_each_enqueue(model, inps, idx) + '\n'
            elif '// hls-fpga-machine-learning insert call' in line:
                newline = '////// call the main variable\n'
                newline += self.write_axi_wrapper_insert_call(model, inps, outs)
            elif '// hls-fpga-machine-learning insert dequeue' in line:
                newline = ''
                for idx, out in enumerate(outs):
                    newline += self.write_axi_wrapper_dequeue(model, inps, outs, idx, out_axi_t)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    ########################################################
    ## write test script  function helper    ###############
    ########################################################

    def write_wrapper_test(self, model):

        oldfile = f"{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp"
        newfile = f"{model.config.get_output_dir()}/{model.config.get_project_name()}_test_wrapper.cpp"

        filedir = os.path.dirname(os.path.abspath(__file__))
        f    = open(os.path.join(filedir, '../templates/vitis_accelerator_ip_flow_partial/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs  = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        fout.write("//// generated by partial backend\n")

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            #Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inputIdx, inp in enumerate(model_inputs):
                    newline += '      ' + f"hls::stream<{self.getDmaTypeName()}> {self.getWrapperPortName(inp, True)}\n"
                    newline += '      nnet::copy_data<float, {destype}, {offset}, N_IN[{inputIdx}]>(in, {inputPortName});\n'.format(
                        destype = self.getDmaTypeName(), offset = offset, inputIdx = str(inputIdx), inputPortName = self.getWrapperPortName(inp, True)
                    )
                    #newline += '      ' + inp.definition_cpp() + ';\n'
                    # newline += '      nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                    #     inp.type.name, offset, inp.size_cpp(), inp.name
                    # )
                    offset += inp.size()
                for out in model_outputs:
                    newline += '      ' + f"hls::stream<{self.getDmaTypeName}> {self.getWrapperPortName(out, False)}\n"
                    #newline += '      ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars  = ','.join([self.getWrapperPortName(inp, True) for inp in model_inputs])
                output_vars = ','.join([self.getWrapperPortName(out, False) for out in model_outputs])
                bram_vars   = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{self.getTopModelName(model)}({all_vars});\n'

                newline += top_level
            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for outIdx, out in enumerate(model_outputs):
                    #newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + f'for(int i = 0; i < N_OUT[{outIdx}]; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                if tb_stream != 'stdout':
                    for outIdx, out in enumerate(model_outputs):
                        # newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                        #     out.type.name, out.size_cpp(), out.name
                        # )  # TODO enable this
                        newline += indent + 'nnet::print_result<{actualType}, {dmaType}, N_OUT[{arrSize}]>({portName}, fout);\n'.format(
                            actualType = out.type.name, dmaType = self.getDmaTypeName, arrSize = outIdx, portName = self.getWrapperPortName(out, False)
                        )  # TODO enable this
            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inpIdx, inp in enumerate(model_inputs):
                    # newline += indent + inp.definition_cpp() + ';\n'
                    # newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                    newline += "        " + f"hls::stream<{self.getDmaTypeName()}> {self.getWrapperPortName(inp, True)}\n"
                    newline += "        " + (f'nnet::fill_zero<{inp.type.name}, {self.getDmaTypeName()},N_INPUT[{str(inpIdx)}]>'
                                             f'({self.getWrapperPortName(inp,True)});\n')
                for out in model_outputs:
                    #newline += indent + out.definition_cpp() + ';\n'
                    newline += "        " + f"hls::stream<{self.getDmaTypeName()}> {self.getWrapperPortName(out, False)}\n"

            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                keep_output = str(tb_stream != 'stdout').lower()  # We keep output if we need to write it to file too.
                if tb_stream != 'file':
                    for outIdx, out in enumerate(model_outputs):
                        #     newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, {});\n'.format(
                        #         out.type.name, out.size_cpp(), out.name, keep_output
                        #     )
                        newline += (indent + 'nnet::print_result<{actualType}, {dmaType}, N_OUT[{arrIdx}]>({portName}, std::cout, true);\n'
                                    .format( actualType = out.type.name,
                                             dmaType = self.getDmaTypeName,
                                             arrIdx = outIdx,
                                             portName = self.getWrapperPortName(out, False) ))

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line

            fout.write(newline)
        f.close()
        fout.close()



    ########################################################
    ## write test script  function helper    ###############
    ########################################################

    def write_board_script(self, model):
        print("[partial reconfig] we are not supporting write_board_script this yet")

    def write_driver(self, model):
        print("[partial reconfig] we are not supporting write_driver this yet")

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

        f = open(os.path.join(filedir, '../templates/vitis_accelerator_ip_flow_partial/build_lib.sh'))
        fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_build_script_multigraph(self, model):
        """Write the build script (build_lib.sh) for stitched multigraph project
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        out = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')
        out.close()

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



