import pandas as pd
import numpy as np
# from source import view_and_print_output
import ggplot as gg


file_coarse = 'data/coarse_lambda_dropout_' + str(2) + '_' + str(50) + '.txt'
newdata = pd.read_csv(file_coarse)
newdata['lambda'] = np.log10(newdata['lambda'])
print(newdata.sort_values(by='validation error', ascending=False))

file_coarse = 'data/fine_lambda_dropout_' + str(2) + '_' + str(50) + '_15.txt'
newdata = pd.read_csv(file_coarse)
newdata['lambda'] = np.log10(newdata['lambda'])
newdata = newdata.sort_values(by='validation error', ascending=False)
newdata['index'] = (1-np.arange(len(newdata))/len(newdata)) **3
print(newdata.sort_values(by='validation error', ascending=False))
p = gg.ggplot(gg.aes(x='lambda', y='dropout prob', color='index'), data=newdata) + \
        gg.geom_point() + \
        gg.xlab('lambda') + \
        gg.ylab('dropout prob') + \
        gg.scale_x_continuous(limits=(-5, 2))
print(p)
