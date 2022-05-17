import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

import pandas as pd

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# TODO make it so it doesn't crash on emtpy rows, like in the dataset
# dataset = tf.data.experimental.make_csv_dataset("./c4-100.csv", batch_size=32, column_names=["game", "moveid", "player", "column", "winner"], label_name="winner", num_epochs=1)

# # For demonstration, iterate over the batches yielded by the dataset.
# for data in dataset:
#    print(data)
#    break

# print("done")

data = pd.read_csv(
    "./c4-100.csv",
    names=["game", "moveid", "player", "column", "winner"])

data = np.array(data)

# print(data)

normalData = data.copy()
normalizer = Normalization(axis=-1)
normalizer.adapt(normalData)

normalized_data = normalizer(normalData)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
