class MgsConMeta:
    # it describes the connection between kernel io and idx of magic streamer
    def __init__(self, io_idx, tensor):
        self.io_idx = io_idx  # the io Idx of the kernel
        self.mgs_idx = -1  # magic streamer idx    -1 means DMA
        self.mgs_wrap_width = 32  # connection width
        self.mgs_row_idx_width = 10  # magic streamer idx size
        # it is not specify that it is input or output


class MagicBufferMeta:
    # it specifies the specification of the magic streamer
    def __init__(self, data_width, row_idx_width, mgs_idx):
        self.data_width = data_width
        self.row_idx_width = row_idx_width
        self.mgs_idx = mgs_idx

    def upgrade_mgs_to_support(self, mgs_con_meta: MgsConMeta):
        # it will upgrade the buffer size of the magic streamer incase it has been reused
        # and the system
        if self.row_idx_width < mgs_con_meta.mgs_row_idx_width:
            self.row_idx_width = mgs_con_meta.mgs_row_idx_width

    def is_data_width_match(self, check_width):
        return self.data_width == check_width

    # the shared class between subgraph and host graph


class MgsConGraph:
    # in suppose to be pool of connection for each sub-graph

    def __init__(self, gid, input_node_links, amt_graph, mgs_model):  # gid = graph id
        self.gid = gid  # graph id
        self.input_cons = []  # input port to magic streamer
        self.output_cons = []  # output port to magic streamer
        # the index of the input_cons and output_cons supposed to be the index of the kernel's io port as well
        self.input_node_links = input_node_links  # the link meta data
        self.amt_graph = amt_graph
        self.mgs_model = mgs_model  # the main magic streamer pool

    def start_convert_graph(self, graph):

        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP

        # THIS IS VERY CRITICAL
        # because current magic streamer don't support input and output at the same time
        # if we do add input connection first the MgsModel class will notice some magic streamer is free
        # and it will assign to the new output at the same time
        # we must ensure that magic streamer to store the output at the magic streamer that did not use as the input

        out_var = graph.get_output_variables()
        for out_idx, out in enumerate(out_var):
            mgs_con_meta = MgsConMeta(out_idx, out)
            self.add_output_con(mgs_con_meta)

        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP
        # DO NOT SWAP INPUT LOOP BEFORE OUTPUT LOOP

        # do the input
        in_var = graph.get_input_variables()
        for in_idx, inp in enumerate(in_var):
            mgs_con_meta = MgsConMeta(in_idx, inp)
            self.add_input_con(mgs_con_meta)

    def is_last_graph(self):
        return (self.gid + 1) == self.amt_graph

    def add_input_con(self, mgs_con_meta):

        src_gid, src_out_idx = self.input_node_links[self.gid][mgs_con_meta.io_idx]
        # src graph idx
        # src_out_idx is the output Idx of the KERNEL not magic streamer idx

        if src_gid == -1:
            pass  # src_mgs_idx = -1  # it means dma
        else:
            # get the magic streamer idx corresponding that contain the input for this port
            src_mgs_con_meta = self.mgs_model.get_mgs_idx(src_gid, src_out_idx)
            mgs_con_meta.mgs_idx = src_mgs_con_meta
            # tell the master that magic streamer is free because it use the input already
            self.mgs_model.move_buffer_to_free_list(src_mgs_con_meta)

        self.input_cons.append(mgs_con_meta)

    def add_output_con(self, mgs_con_meta):

        if self.is_last_graph():
            self.output_cons.append(mgs_con_meta)
            return

        # we check the available magic stream at the time which current reconfigurable module is operating
        stream_buffer_idx = self.mgs_model.get_existing_possible_mgs_buffer(mgs_con_meta)

        if stream_buffer_idx is None:
            stream_buffer_idx = self.mgs_model.allocate_mgs_buffer(mgs_con_meta)

        # upgrade the magic stream buffer to match size of it is lower
        self.mgs_model.upgrade_mgs_to_support(mgs_con_meta, stream_buffer_idx)

        # tel the master  that magicstreamer is occupied
        self.mgs_model.move_buffer_to_using_list(stream_buffer_idx)
        mgs_con_meta.mgs_idx = stream_buffer_idx

        self.output_cons.append(mgs_con_meta)


class MgsModel:
    def __init__(self, multigraph):
        self.multigraph = multigraph
        self.con_graphs = []
        self.mgs_buffer_meta = []  # index of the system supposed to be magic streamer and its port id

        self.mgs_buffer_holding = []
        self.mgs_buffer_empty = []

    def start_convert_model(self):
        for gid, sub_graph in enumerate(self.multigraph.graphs):
            # initialize the MgsConGraph
            input_node_link = self.multigraph.input_node_links
            amt_graph = len(self.multigraph.graphs)
            mgs_con_graph = MgsConGraph(gid, input_node_link, amt_graph, self)
            mgs_con_graph.start_convert_graph(sub_graph)
            # start fill the metadata
            self.add_mgs_con_graph(mgs_con_graph)

        print("finish_convert")

    def add_mgs_con_graph(self, mgs_con_graph):
        self.con_graphs.append(mgs_con_graph)

    # start get the data

    def get_io_idx_for_all_mgs_buffer_with_dma(self, gid, is_input):
        related_con_graph = self.con_graphs[gid]
        io_con_meta_list = related_con_graph.input_cons if is_input else related_con_graph.output_cons

        result_list = [None for _ in range(len(self.mgs_buffer_meta))]

        dma_port = []

        for io_idx, io_con_meta in enumerate(io_con_meta_list):
            mgs_idx = io_con_meta.mgs_idx
            if mgs_idx == -1:
                dma_port.append(io_idx)
                continue
            result_list[mgs_idx] = io_idx

        # calculate with dma
        if len(dma_port) > 1:
            print("[warning] the system detect io port that should be used by dma is more than one")
        result_list_with_dma = [None if len(dma_port) == 0 else dma_port[0]]
        result_list_with_dma.extend(result_list)

        return result_list_with_dma

    # for outsider used
    def get_mgs_idx_src(self, gid, inputIdx):
        if gid not in range(0, self.multigraph.amt_graph):
            raise Exception(
                "get_mgs_idx_src: gid {gid} is out of bound. The amount of graph is {amt_graph}".format(
                    gid=gid, amt_graph=self.multigraph.amt_graph
                )
            )
        if inputIdx not in range(0, len(self.con_graphs[gid].input_cons)):
            raise Exception(
                "get_mgs_idx_src: inputIdx {inputIdx} is out of bound. The amount of input is {input_num}".format(
                    inputIdx=inputIdx, input_num=len(self.con_graphs[gid].input_cons)
                )
            )
        return self.con_graphs[gid].input_cons[inputIdx].mgs_idx

    # for outsider use
    def get_mgs_idx_dst(self, gid, outputIdx):
        if gid not in range(0, self.multigraph.amt_graph):
            raise Exception(
                "get_mgs_idx_dst: gid {gid} is out of bound. The amount of graph is {amt_graph}".format(
                    gid=gid, amt_graph=self.multigraph.amt_graph
                )
            )
        if outputIdx not in range(0, len(self.con_graphs[gid].output_cons)):
            raise Exception(
                "get_mgs_idx_dst: outputIdx {outputIdx} is out of bound. The amount of output is {output_num}".format(
                    outputIdx=outputIdx, output_num=len(self.con_graphs[gid].output_cons)
                )
            )
        return self.con_graphs[gid].output_cons[outputIdx].mgs_idx

    # magic streamer buffer

    # used by the vitis uniifed partial backend writer and MgsConGraph
    def get_mgs_idx(self, gid, outputIdx):
        # find the magic streamer index at subgraph gid at output port outputIdx of kernel
        if gid < 0:
            return -1
        mgs_con_meta = self.con_graphs[gid].output_cons[outputIdx]
        return mgs_con_meta.mgs_idx

    def upgrade_mgs_to_support(self, mgs_con_meta, mgsIdx):
        if mgsIdx >= len(self.mgs_buffer_meta) or mgsIdx < 0:
            raise Exception("upgrade magic streamer with Idx {mgsIdx} is out of bound.")
        self.mgs_buffer_meta[mgsIdx].upgrade_mgs_to_support(mgs_con_meta)

    def allocate_mgs_buffer(self, mgs_con_meta):
        newStreamBuffer = MagicBufferMeta(
            mgs_con_meta.mgs_wrap_width, mgs_con_meta.mgs_row_idx_width, len(self.mgs_buffer_meta)
        )
        mgs_idx = len(self.mgs_buffer_meta)
        self.mgs_buffer_meta.append(newStreamBuffer)
        return mgs_idx

    def get_existing_possible_mgs_buffer(self, mgs_con_meta):
        # filter the match buffer from exis
        matched_buffer = list(
            filter(lambda mgs: mgs.is_data_width_match(mgs_con_meta.mgs_wrap_width), self.mgs_buffer_empty)
        )

        highest_possible_buffer = sorted(matched_buffer, key=lambda x: x.row_idx_width, reverse=True)

        return None if len(highest_possible_buffer) == 0 else highest_possible_buffer[0].mgs_idx

    def move_buffer_to_using_list(self, mgs_idx):
        # delete from free list first
        self.mgs_buffer_empty = list(filter(lambda x: x.mgs_idx != mgs_idx, self.mgs_buffer_empty))
        # add to holding list
        self.mgs_buffer_holding.append(self.mgs_buffer_meta[mgs_idx])

    def move_buffer_to_free_list(self, mgs_idx):
        # delete from holding list first
        self.mgs_buffer_holding = list(filter(lambda x: x.mgs_idx != mgs_idx, self.mgs_buffer_holding))
        # add to free list
        self.mgs_buffer_empty.append(self.mgs_buffer_meta[mgs_idx])
