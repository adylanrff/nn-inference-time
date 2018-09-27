import pandas as pd
import numpy as np

import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

FILEPATH = 'data/data.csv'
df = pd.read_csv(FILEPATH, header=None)
df.head()

input_data = np.array(df)

models = []
for depth in range(0,32):
    model = Sequential()
    model.add(Dense(units=352, input_dim = 352))
    for i in range(depth):
        model.add(Dense(units=352))
    model.add(Dense(units=1))
    
    models.append(model)


inference_times = []
inference_avg_times = []
for model in models:
    start = time.time()
    x  = model.predict(input_data)
    inference_time = (time.time()-start)
    inference_avg_time = inference_time / len(input_data)
    inference_times.append(inference_time)
    inference_avg_times.append(inference_avg_time)


INFERENCE_FIGUREPATH = "figure/inference_32_1.png"
INFERENCE_AVG_FIGUREPATH = "figure/inference_32_1_avg.png"

plt.title("Inference Time")
plt.xlabel("Depth")
plt.ylabel("seconds")
plt.scatter([i for i in range(1,33)],inference_times)
plt.plot([i for i in range(1,33)],inference_times)
# plt.hist(inference_times)
plt.savefig(FIGUREPATH)
plt.show()


print(inference_avg_times)
plt.title("Inference Time Average (100k entries)")
plt.xlabel("Depth")
plt.ylabel("seconds")
plt.ylim(0,0.0002)
plt.scatter([i for i in range(1,33)],inference_avg_times)
plt.plot([i for i in range(1,33)],inference_avg_times)
# plt.hist(inference_times)
plt.savefig(FIGUREPATH)
plt.show()
