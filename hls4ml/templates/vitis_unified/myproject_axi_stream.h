#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include <iostream>
// hls-fpga-machine-learning insert include

// hls-fpga-machine-learning insert definitions

void MY_PROJECT_TOP_FUNC(hls::stream<MY_DMA_PACKET_TYPE_INPUT> &axi_input_stream,
                         hls::stream<MY_DMA_PACKET_TYPE_OUTPUT> &axi_output_stream, int batch_size);
#endif
