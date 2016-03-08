import pandas as pd
import numpy as np
import ggplot as gg


def assess_output(filepath, modelname):
    df = pd.read_csv(filepath)
    nb_epoch = np.amax(df['itr'])
    df['fold'] = np.repeat(np.arange(1, 11), nb_epoch)
    df['fold'] = df['fold'].astype('object')
    df = df.groupby(['itr']).median()
    df['model'] = np.repeat(np.array([modelname]), nb_epoch)
    df['model'] = df['model'].astype('object')
    return df

filepath_baseline = '../data/assess_baseline.txt'
filepath_w_reg = '../data/assess_baseline_with_reg.txt'

df = assess_output('../data/assess_baseline.txt', 'baseline')
df = df.append(assess_output('../data/assess_baseline_with_reg.txt', 'L2=10^-4'))
df['itr'] = df.index
print(df)
print(df.dtypes)
p = gg.ggplot(gg.aes(x='itr', y='val_acc', color='model'), data=df) + \
    gg.geom_point()
print(p)


# p = gg.ggplot(gg.aes(x='itr', y='acc', color='index'), data=df) + \
#         gg.geom_point() + \
#         gg.xlab('lambda') + \
#         gg.ylab('dropout prob') + \
#         gg.scale_x_continuous(limits=(-5, 2)) + \
#         gg.facet_wrap("num_layers")
# print(p)
