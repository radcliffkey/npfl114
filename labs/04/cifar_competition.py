#!/usr/bin/env python3

# our team IDs:
# adf6ddd7-4724-11e9-b0fd-00505601122b
# bd9460fd-444e-11e9-b0fd-00505601122b
import sys
from typing import List

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, layers_specifications: List[str]):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.

        # TODO: After creating the model, call `self.compile` with appropriate arguments.

        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        hidden = inputs

        for spec in layers_specifications:
            if spec.startswith('CB'):
                params = spec.split(sep='-')
                filters = int(params[1])
                kernel_size = int(params[2])
                stride = int(params[3])
                padding = params[4]
                hidden = tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=stride,
                    padding=padding, use_bias=False)(hidden)

                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.layers.Activation(activation=tf.nn.relu)(hidden)

            elif spec.startswith('C'):
                params = spec.split(sep='-')
                filters = int(params[1])
                kernel_size = int(params[2])
                stride = int(params[3])
                padding = params[4]

                hidden = tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=stride,
                    padding=padding, activation=tf.nn.relu)(hidden)

            elif spec.startswith('M'):
                params = spec.split(sep='-')
                kernel_size = int(params[1])
                stride = int(params[2])
                hidden = tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride)(hidden)

            elif spec.startswith('R'):
                skip = hidden

                for layer in spec.replace('R-[', '').replace(']', '').split(','):
                    if layer.startswith('CB'):
                        params = layer.split(sep='-')
                        filters = int(params[1])
                        kernel_size = int(params[2])
                        stride = int(params[3])
                        padding = params[4]

                        hidden = tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size,
                            strides=stride, padding=padding, use_bias=False)(hidden)

                        hidden = tf.keras.layers.BatchNormalization()(hidden)
                        hidden = tf.keras.layers.Activation(activation=tf.nn.relu)(hidden)

                    elif layer.startswith('C'):
                        params = layer.split(sep='-')
                        filters = int(params[1])
                        kernel_size = int(params[2])
                        stride = int(params[3])
                        padding = params[4]

                        hidden = tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size,
                            strides=stride,
                            padding=padding, activation=tf.nn.relu)(hidden)

                hidden = tf.keras.layers.add([hidden, skip])

            elif spec.startswith('F'):
                hidden = tf.keras.layers.Flatten()(hidden)

            elif spec.startswith('G'):
                hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)

            elif spec.startswith('D'):
                params = spec.split(sep='-')
                size = params[1]
                hidden = tf.keras.layers.Dense(size, activation=tf.nn.relu)(hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=(len(cifar.train.data["images"])/args.batch_size) * args.epochs,
            decay_rate=0.1
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def train_generator(self, cifar, generator, args):
        self.fit_generator(
            # cifar.train.data["images"], cifar.train.data["labels"],
            # batch_size=args.batch_size,
            generator.flow(cifar.train.data["images"], cifar.train.data["labels"], batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, cifar, args):
        test_logs = self.evaluate(cifar.dev.data["images"], cifar.dev.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn",
        # default='CB-16-5-2-same,M-3-2,F,D-100',
        # default='C-16-3-1-same,C-16-3-1-same,M-2-2,CB-32-3-1-valid,M-2-2,O-0.5,D-100,O-0.5,D-100',
        # default='CB-16-5-2-same,M-3-2,F,O-0.5,D-100,O-0.5,D-100',
        # todo:
        default='C-16-3-2-same,C-32-3-1-same,M-2-2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,G',
        # default='C-32-5-2-same,C-32-5-2-same,M-2-2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,A,O-0.5,D-512,O-0.5,D-512',

        # Miso:
        #default='C-32-3-1-same,C-32-3-1-same,M-2-2,C-32-3-1-same,M-2-2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,CB-256-3-1-same,CB-256-3-1-same,M-2-2,A,DO-0.3'
        type=str,
        help="CNN architecture.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))

    # Load data
    cifar = CIFAR10()

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # rescale=1. / 255,
        rescale=None,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    generator.fit(cifar.train.data["images"])

    layers_specifications = re.split(r',\s*(?![^\[\]]*\])', args.cnn)

    # Create the network and train
    network = Network(layers_specifications)
    # network.train(cifar, args)
    network.train_generator(cifar, generator, args)

    # Compute dev set accuracy and print it
    accuracy = network.test(cifar, args)
    print("\nACCURACY = {:.2f}".format(100 * accuracy), file=sys.stderr)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
