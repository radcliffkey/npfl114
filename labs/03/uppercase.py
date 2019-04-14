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

from uppercase_data import UppercaseData


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=uppercase_data.train.size / args.batch_size,
    decay_rate=0.8
)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(uppercase_data.train.alphabet), output_dim=128, input_length=2 * args.window + 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)


canHaveCase = [c.upper() != c.lower() for c in uppercase_data.train.text]
train_x = uppercase_data.train.data['windows'][canHaveCase]
train_y = uppercase_data.train.data['labels'][canHaveCase]

canHaveCaseDev = [c.upper() != c.lower() for c in uppercase_data.dev.text]
dev_x = uppercase_data.dev.data['windows'][canHaveCaseDev]
dev_y = uppercase_data.dev.data['labels'][canHaveCaseDev]

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir)
model.fit(train_x, train_y,
    validation_data=(dev_x,dev_y),
    batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])


with open("uppercase_pred_dev.txt", "w", encoding="utf-8") as out_file:
    test_pred = model.predict(uppercase_data.dev.data['windows'])
    pred_text = ''.join(c.upper() if test_pred[i] > 0.5 else c for i, c in enumerate(uppercase_data.dev.text.lower()))
    print(pred_text, file=out_file)

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    test_pred = model.predict(uppercase_data.test.data['windows'])
    pred_text = ''.join(c.upper() if test_pred[i] > 0.5 else c for i, c in enumerate(uppercase_data.test.text.lower()))
    print(pred_text, file=out_file)
