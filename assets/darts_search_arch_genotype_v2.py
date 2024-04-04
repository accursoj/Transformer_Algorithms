
darts_search_5 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3', 3), ('skip_connect', 0), ('dil_conv_3', 4), ('skip_connect', 0)], normal_concat=range(4, 6))

darts_search_6 = Genotype(normal=[('skip_connect', 1), ('encoder_att', 0), ('sep_conv_3', 0), ('skip_connect', 1), ('dil_conv_3', 3), ('encoder_att', 2), ('skip_connect', 1), ('dense', 0)], normal_concat=range(4, 6))

darts_search_7 = Genotype(normal=[('dense', 1), ('dense', 0), ('sep_conv_3', 0), ('dense', 1), ('dil_conv_3', 3), ('skip_connect', 0), ('encoder_att', 2), ('encoder_att', 3)], normal_concat=range(4, 6))

darts_search_8 = Genotype(normal=[('sep_conv_1', 1), ('conv_1', 0), ('skip_connect', 2), ('dil_conv_3', 0), ('dil_conv_3', 3), ('decoder_att', 0), ('encoder_att', 1), ('encoder_att', 2)], normal_concat=range(4, 6))

darts_search_9 = Genotype(normal=[('conv_1', 0), ('conv_1', 1), ('sep_conv_3', 0), ('conv_1', 1), ('dil_conv_3', 3), ('decoder_att', 0), ('encoder_att', 2), ('encoder_att', 1)], normal_concat=range(4, 6))
