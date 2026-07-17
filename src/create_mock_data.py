import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
num_engines = 5
data = []
for engine in range(1, num_engines + 1):
    max_cycle = np.random.randint(150, 250)
    for cycle in range(1, max_cycle + 1):
        row = [engine, cycle] + list(np.random.rand(3)) + list(np.random.rand(21))
        data.append(row)

df = pd.DataFrame(data)
df.to_csv('data/train_FD001.txt', sep=' ', index=False, header=False)
print("Mock train_FD001.txt created in data/ folder for testing.")
