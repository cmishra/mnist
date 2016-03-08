import pandas as pd
import numpy as np
# from source import view_and_print_output
import ggplot as gg


df = pd.DataFrame()
for num_layers, num_nodes in [(2, 50), (2, 100), (2, 150), (2, 200), (4, 50), (4, 100), (4, 150), (4, 200)]:
    file_coarse = '../../data/coarse_lambda_dropout_' + str(num_layers) + '_' + str(num_nodes) + '.txt'
    newdata = pd.read_csv(file_coarse)
    newdata = newdata.sort_values(by='validation error', ascending=True)
    newdata['lambda'] = np.log10(newdata['lambda'])
    newdata['index'] = (np.arange(len(newdata), dtype='float')/len(newdata))**3
    newdata['config'] = str(num_layers * 100 + num_nodes) +  ' ' +  str(num_layers) + ' ' + str(num_nodes)
    df = df.append(newdata)
print(df.sort_values(by='validation error', ascending=False).head(20))
p = gg.ggplot(gg.aes(x='lambda', y='dropout prob', color='index'), data=df) + \
        gg.geom_point() + \
        gg.xlab('lambda') + \
        gg.ylab('dropout prob') + \
        gg.scale_x_continuous(limits=(-5, 2)) + \
        gg.facet_wrap('config')
print(p)

# Conclusion: ignore dropout