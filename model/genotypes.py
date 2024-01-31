from collections import namedtuple

Genotype = namedtuple('Genotype', 'v_att v_ret')

arch = Genotype(v_att=[[('conv_1x1', 0.90878224), ('conv_7x7', 0.05903241), ('dil_7x7', 0.032185297), ('conv_3x3', 2.1e-44), ('conv_5x5', 2.1e-44), ('conv_1x3', 2.1e-44), ('conv_3x1', 2.1e-44), ('conv_1x5', 2.1e-44), ('conv_5x1', 2.1e-44), ('dil_3x3', 2.1e-44), ('dil_5x5', 2.1e-44)], [('dil_3x3', 0.8650955), ('conv_1x3', 0.07472725), ('conv_1x1', 0.060177255), ('conv_3x3', 2.7e-44), ('conv_5x5', 2.7e-44), ('conv_7x7', 2.7e-44), ('conv_3x1', 2.7e-44), ('conv_1x5', 2.7e-44), ('conv_5x1', 2.7e-44), ('dil_5x5', 2.7e-44), ('dil_7x7', 2.7e-44)], [('conv_7x7', 0.9621563), ('conv_1x5', 0.019885864), ('conv_3x1', 0.017957794), ('conv_1x1', 1.3e-44), ('conv_3x3', 1.3e-44), ('conv_5x5', 1.3e-44), ('conv_1x3', 1.3e-44), ('conv_5x1', 1.3e-44), ('dil_3x3', 1.3e-44), ('dil_5x5', 1.3e-44), ('dil_7x7', 1.3e-44)], [('conv_5x5', 0.51434284), ('dil_7x7', 0.44820768), ('conv_7x7', 0.037449475), ('conv_1x1', 1.3e-44), ('conv_3x3', 1.3e-44), ('conv_1x3', 1.3e-44), ('conv_3x1', 1.3e-44), ('conv_1x5', 1.3e-44), ('conv_5x1', 1.3e-44), ('dil_3x3', 1.3e-44), ('dil_5x5', 1.3e-44)], [('dil_7x7', 0.9560288), ('conv_3x3', 0.02392026), ('conv_7x7', 0.020050876), ('conv_1x1', 1.4e-44), ('conv_5x5', 1.4e-44), ('conv_1x3', 1.4e-44), ('conv_3x1', 1.4e-44), ('conv_1x5', 1.4e-44), ('conv_5x1', 1.4e-44), ('dil_3x3', 1.4e-44), ('dil_5x5', 1.4e-44)], [('conv_5x5', 0.5080779), ('dil_7x7', 0.47940484), ('conv_1x3', 0.012517204), ('conv_1x1', 8e-45), ('conv_3x3', 8e-45), ('conv_7x7', 8e-45), ('conv_3x1', 8e-45), ('conv_1x5', 8e-45), ('conv_5x1', 8e-45), ('dil_3x3', 8e-45), ('dil_5x5', 8e-45)], [('dil_5x5', 0.9155723), ('dil_7x7', 0.04468688), ('conv_1x3', 0.039740868), ('conv_1x1', 2.1e-44), ('conv_3x3', 2.1e-44), ('conv_5x5', 2.1e-44), ('conv_7x7', 2.1e-44), ('conv_3x1', 2.1e-44), ('conv_1x5', 2.1e-44), ('conv_5x1', 2.1e-44), ('dil_3x3', 2.1e-44)], [('conv_7x7', 0.6196366), ('conv_5x1', 0.36296332), ('conv_1x5', 0.017400118), ('conv_1x1', 1e-44), ('conv_3x3', 1e-44), ('conv_5x5', 1e-44), ('conv_1x3', 1e-44), ('conv_3x1', 1e-44), ('dil_3x3', 1e-44), ('dil_5x5', 1e-44), ('dil_7x7', 1e-44)], [('conv_5x5', 0.9176833), ('dil_5x5', 0.04673757), ('conv_1x3', 0.035579205), ('conv_1x1', 1.8e-44), ('conv_3x3', 1.8e-44), ('conv_7x7', 1.8e-44), ('conv_3x1', 1.8e-44), ('conv_1x5', 1.8e-44), ('conv_5x1', 1.8e-44), ('dil_3x3', 1.8e-44), ('dil_7x7', 1.8e-44)], [('conv_5x1', 0.92496157), ('dil_7x7', 0.042640142), ('dil_5x5', 0.03239823), ('conv_1x1', 1.8e-44), ('conv_3x3', 1.8e-44), ('conv_5x5', 1.8e-44), ('conv_7x7', 1.8e-44), ('conv_1x3', 1.8e-44), ('conv_3x1', 1.8e-44), ('conv_1x5', 1.8e-44), ('dil_3x3', 1.8e-44)]], v_ret=[[('conv_5x5', 0.58136183), ('dil_7x7', 0.3916563), ('conv_7x7', 0.02698191), ('conv_1x1', 1.1e-44), ('conv_3x3', 1.1e-44), ('conv_1x3', 1.1e-44), ('conv_3x1', 1.1e-44), ('conv_1x5', 1.1e-44), ('conv_5x1', 1.1e-44), ('dil_3x3', 1.1e-44), ('dil_5x5', 1.1e-44)], [('conv_5x5', 0.57436407), ('conv_7x7', 0.39823005), ('conv_3x3', 0.02740587), ('conv_1x1', 1.1e-44), ('conv_1x3', 1.1e-44), ('conv_3x1', 1.1e-44), ('conv_1x5', 1.1e-44), ('conv_5x1', 1.1e-44), ('dil_3x3', 1.1e-44), ('dil_5x5', 1.1e-44), ('dil_7x7', 1.1e-44)], [('conv_7x7', 0.5828956), ('conv_5x5', 0.4042664), ('conv_3x3', 0.012838086), ('conv_1x1', 8e-45), ('conv_1x3', 8e-45), ('conv_3x1', 8e-45), ('conv_1x5', 8e-45), ('conv_5x1', 8e-45), ('dil_3x3', 8e-45), ('dil_5x5', 8e-45), ('dil_7x7', 8e-45)], [('conv_7x7', 0.51584774), ('conv_5x5', 0.4588264), ('conv_3x3', 0.025325846), ('conv_1x1', 1e-44), ('conv_1x3', 1e-44), ('conv_3x1', 1e-44), ('conv_1x5', 1e-44), ('conv_5x1', 1e-44), ('dil_3x3', 1e-44), ('dil_5x5', 1e-44), ('dil_7x7', 1e-44)], [('conv_7x7', 0.54078174), ('conv_3x1', 0.44098654), ('conv_1x3', 0.018231727), ('conv_1x1', 8e-45), ('conv_3x3', 8e-45), ('conv_5x5', 8e-45), ('conv_1x5', 8e-45), ('conv_5x1', 8e-45), ('dil_3x3', 8e-45), ('dil_5x5', 8e-45), ('dil_7x7', 8e-45)]])

