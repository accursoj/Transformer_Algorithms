
darts_search_5 = Genotype(normal=[('sep_conv_3', 0), ('conv_3', 1), ('dil_conv_1', 0), ('dil_conv_3', 2), ('sep_conv_1', 3), ('sep_conv_1', 0), ('skip_connect', 0), ('dense', 1)], normal_concat=range(4, 6))

darts_search_6 = Genotype(normal=[('conv_1', 1), ('dil_conv_1', 0), ('dil_conv_1', 0), ('dense', 2), ('encoder_att', 3), ('conv_1', 2), ('encoder_att', 0), ('encoder_att', 3)], normal_concat=range(4, 6))

darts_search_7 = Genotype(normal=[('conv_1', 1), ('conv_3', 0), ('dil_conv_1', 0), ('encoder_att', 1), ('dil_conv_3', 0), ('conv_1', 3), ('dil_conv_3', 3), ('sep_conv_1', 0)], normal_concat=range(4, 6))

darts_search_8 = Genotype(normal=[('dense', 1), ('encoder_att', 0), ('dil_conv_1', 0), ('dense', 1), ('dil_conv_3', 0), ('dil_conv_3', 3), ('sep_conv_1', 0), ('decoder_att', 1)], normal_concat=range(4, 6))

darts_search_9 = Genotype(normal=[('dense', 1), ('conv_3', 0), ('dil_conv_1', 0), ('dense', 2), ('sep_conv_1', 0), ('dense', 2), ('sep_conv_1', 0), ('dil_conv_3', 3)], normal_concat=range(4, 6))
