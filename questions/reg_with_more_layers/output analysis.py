import pandas as pd
import numpy as np
# from source import view_and_print_output
import ggplot as gg


df = pd.DataFrame()
for num_layers, num_nodes in [(3, 50), (4, 50), (5, 50), (2, 50)]:
    file_coarse = '../../data/coarse_lambda_dropout_' + str(num_layers) + '_' + str(num_nodes) + '.txt'
    newdata = pd.read_csv(file_coarse)
    newdata = newdata.sort_values(by='validation error', ascending=True)
    newdata['lambda'] = np.log10(newdata['lambda'])
    newdata['index'] = (np.arange(len(newdata), dtype='float')/len(newdata))**3
    newdata['num_layers'] = num_layers
    newdata['num_nodes'] = num_nodes
    df = df.append(newdata)
print(df.sort_values(by='validation error', ascending=False).head(20))
p = gg.ggplot(gg.aes(x='lambda', y='dropout prob', color='index'), data=df) + \
        gg.geom_point() + \
        gg.xlab('lambda') + \
        gg.ylab('dropout prob') + \
        gg.scale_x_continuous(limits=(-5, 2)) + \
        gg.facet_wrap("num_layers")
print(p)
