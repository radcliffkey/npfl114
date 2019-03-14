#!/usr/bin/env python3
#
# bd9460fd-444e-11e9-b0fd-00505601122b
#
import numpy as np

from collections import Counter

if __name__ == "__main__":

    data_dist = Counter()
    model_dist = Counter()

    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip('\n')
            data_dist[line] += 1

    data_total_cnt = sum(data_dist.values())

    data_dist = {k: v / data_total_cnt for k, v in data_dist.items()}

    # Load model distribution, each line `word \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip('\n')
            k, v = line.split('\t')
            model_dist[k] += float(v)

    model_total_cnt = sum(model_dist.values())
    model_dist = {k: v / model_total_cnt for k, v in model_dist.items()}

    all_keys = data_dist.keys() | model_dist.keys()

    data_dist_arr = np.array([data_dist.get(k, 0.0) for k in all_keys])
    model_dist_arr = np.array([model_dist.get(k, 0.0) for k in all_keys])

    entropy = -np.sum(data_dist_arr * np.log(data_dist_arr, where=data_dist_arr != 0.0))
    print("{:.2f}".format(entropy))

    cross_entropy = -np.sum(data_dist_arr * np.log(model_dist_arr))
    print("{:.2f}".format(cross_entropy))

    kl_div = cross_entropy - entropy
    print("{:.2f}".format(kl_div))
