import tensorflow as tf
import numpy as np

arr_range_list = np.arange(0, 100).astype(np.float32)
shape = arr_range_list.shape
dataset = tf.data.Dataset.from_tensor_slices(arr_range_list)
dataset_iter = dataset.shuffle(shape[0]).batch(10)

def model(xs):
    return tf.multiply(xs, 0.1)

for it in dataset_iter:
    logits = model(it)
    print logits