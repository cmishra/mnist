import pandas as pd
import numpy as np
import ggplot as gg


filepath_baseline = '../data/assess_baseline.txt'
filepath_w_reg = '../data/assess_baseline_with_reg.txt'

to_analyze = filepath_baseline

df = pd.read_csv(to_analyze)
nb_epoch = np.amax(df['itr'])
df['fold'] = np.repeat(np.arange(1, 11), nb_epoch)
df['fold'] = df['fold'].astype('object')

p = gg.ggplot(gg.aes(x='itr', y='val_acc', color='fold'), df) + \
    gg.geom_point()
print(p)