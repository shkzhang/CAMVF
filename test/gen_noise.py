import numpy as np
import pandas as pd

classes = [136,102]
noise_size = 256
noise = np.random.rand(sum(classes),noise_size)
df = pd.DataFrame(noise, columns=[f'dimension_{i}' for i in range(noise_size)])

df['class'] = [0]*(classes[0]) + [1]*(classes[1])

df.to_csv('noise_data_with_class.csv', index=False)