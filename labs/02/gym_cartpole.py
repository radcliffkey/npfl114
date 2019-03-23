#!/usr/bin/env python3

# our team IDs:
# adf6ddd7-4724-11e9-b0fd-00505601122b
# bd9460fd-444e-11e9-b0fd-00505601122b

import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load the data
observations, labels = [], []
with open("gym_cartpole-data.txt", "r") as data:
    for line in data:
        columns = line.rstrip("\n").split()
        observations.append([float(column) for column in columns[0:-1]])
        labels.append(int(columns[-1]))
observations, labels = np.array(observations), np.array(labels)

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=observations.shape[0] / args.batch_size,
    decay_rate=0.96
)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(4,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir)
model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])

model.save("gym_cartpole_model.h5", include_optimizer=False)
