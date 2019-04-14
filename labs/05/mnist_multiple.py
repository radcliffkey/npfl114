#!/usr/bin/env python3

# our team IDs:
# adf6ddd7-4724-11e9-b0fd-00505601122b
# bd9460fd-444e-11e9-b0fd-00505601122b

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.
        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation (the goal is to predict if the first digit is larger than the second)
        #
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.

        shared_digit_input = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        hidden = tf.keras.layers.Conv2D(10, 3, 2, "valid")(shared_digit_input)
        #hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Conv2D(20, 3, 2, "valid")(hidden)
        #hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Flatten()(hidden)
        shared_digit_output = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
        #hidden = tf.keras.layers.Dropout(args.dropout)(hidden)

        digit_model = tf.keras.Model(inputs=shared_digit_input, outputs=shared_digit_output)

        shared_digit_cls = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        digit1_input = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        digit2_input = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        digit1_repr = digit_model(digit1_input)
        digit2_repr = digit_model(digit2_input)

        digit1_cls = shared_digit_cls(digit1_repr)
        digit2_cls = shared_digit_cls(digit2_repr)

        diff_repr = tf.keras.layers.concatenate([digit1_repr, digit2_repr])
        diff_hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(diff_repr)
        diff_output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(diff_hidden)

        self.model = tf.keras.Model(inputs=[digit1_input, digit2_input], outputs=[digit1_cls, digit2_cls, diff_output])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.BinaryCrossentropy()],
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_dig"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy_gt")
            ],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None


    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                # yield the suitable modified inputs and targets using batches[0:2]
                yield ([batches[0]["images"], batches[1]["images"]], [batches[0]["labels"], batches[1]["labels"], batches[0]["labels"] > batches[1]["labels"]])
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # Train for one epoch using `model.train_on_batch` for each batch.
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(batch[0], batch[1])

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        total_cnt = 0
        accuracy_sum = 0.0
        indirect_accuracy_sum = 0.0

        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            digit1_pred, digit2_pred, is_gt_pred = self.model.predict_on_batch(inputs)
            batch_size = is_gt_pred.shape[0]

            digit1_pred = digit1_pred.argmax(1)
            digit2_pred = digit2_pred.argmax(1)

            is_gt_pred = is_gt_pred.flatten() >= 0.5
            is_gt_indirect_pred = digit1_pred > digit2_pred

            accuracy_sum += np.sum(is_gt_pred == targets[2])
            indirect_accuracy_sum += np.sum(is_gt_indirect_pred == targets[2])
            total_cnt += batch_size

        return accuracy_sum / total_cnt, indirect_accuracy_sum / total_cnt


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
