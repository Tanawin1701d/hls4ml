// hls-fpga-machine-learning insert include

void load_input(hls::stream<dma_data_packet> &axi_input_stream, hls::stream<INPUT_LAYER_TYPE> &model_input_stream,
                int batch_size) {
load_input_loop:
    // send the data to the stream
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N_IN_0 / INPUT_LAYER_TYPE::size; ++chunk_idx) {
            INPUT_LAYER_TYPE input_chunk;
            for (unsigned elem_idx = 0; elem_idx < INPUT_LAYER_TYPE::size; elem_idx++) {
                dma_data_packet axi_packet;
                axi_input_stream.read(axi_packet);
                input_chunk[elem_idx] = axi_packet.data;
            }
            model_input_stream.write(input_chunk);
        }
    }
}

template <unsigned N, typename AXI_T, typename MODEL_T>
void load_input_flat(hls::stream<AXI_T> &axi_input_stream, hls::stream<MODEL_T> &model_input_stream, int batch_size) {
load_input_flat_loop:
    // send the data to the stream
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N / MODEL_T::size; ++chunk_idx) {
            AXI_T axi_packet;
            axi_input_stream.read(axi_packet);
            MODEL_T input_chunk;
            input_chunk = axi_packet.data;
            model_input_stream.write(input_chunk);
        }
    }
}

void store_result(hls::stream<OUTPUT_LAYER_TYPE> &model_output_stream, hls::stream<dma_data_packet> &axi_output_stream,
                  int batch_size) {
store_result_loop:
    // send back the data
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N_OUT_0 / OUTPUT_LAYER_TYPE::size; ++chunk_idx) {
            OUTPUT_LAYER_TYPE output_chunk = model_output_stream.read();
            for (unsigned elem_idx = 0; elem_idx < OUTPUT_LAYER_TYPE::size; elem_idx++) {
                dma_data_packet axi_packet;
                axi_packet.keep = -1;
                axi_packet.data = (OUTPUT_GMEM_TYPE)(output_chunk[elem_idx]);
                axi_packet.last = (q == (batch_size - 1)) && (((chunk_idx + 1) * (elem_idx + 1)) == N_OUT_0);
                axi_output_stream.write(axi_packet);
            }
        }
    }
}

template <unsigned N, typename AXI_T, typename MODEL_T>
void store_result_flat(hls::stream<MODEL_T> &model_output_stream, hls::stream<AXI_T> &axi_output_stream, int batch_size) {
store_result_flat_loop:
    // send back the data
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N / MODEL_T::size; ++chunk_idx) {
            MODEL_T output_chunk = model_output_stream.read();
            AXI_T axi_packet;
            axi_packet.data = output_chunk;
            axi_packet.last = (q == (batch_size - 1)) && ((chunk_idx + 1) == (N / MODEL_T::size));
            axi_output_stream.write(axi_packet);
        }
    }
}

void compute( // hls-fpga-machine-learning insert stream parameter,
    int batch_size) {
    for (int q = 0; q < batch_size; q++) {
        MY_PROJECT(// hls-fpga-machine-learning insert compute-streams);
    }
}

// hls-fpga-machine-learning insert top-func-decl
// hls-fpga-machine-learning insert interface

// hls-fpga-machine-learning insert stream decl

#pragma HLS DATAFLOW

// hls-fpga-machine-learning insert load-calls
    compute(// hls-fpga-machine-learning insert compute-call-args);
    // hls-fpga-machine-learning insert store-calls
    }
