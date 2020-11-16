import numpy as np
from collections import defaultdict, Counter


def mrr_micro(y_true, y_pred, k):

    # Initialize list for storing reciprocal ranks
    rr_list = list()

    # Loop over labels and predictions
    for true, pred in zip(y_true, y_pred):

        # Take top k predictions
        cand_list = pred[:k]

        if true in cand_list:

            # Compute rank if true label in top k predictions
            r = cand_list.index(true) + 1

            # Add reciprocal rank to list
            rr_list.append(1 / r)

        else:

            # Otherwise add 0 to list
            rr_list.append(0)

    # Compute mean reciprocal rank
    return np.mean(rr_list)


def mrr_macro(y_true, y_pred, k):

    # Initialize dictionary for storing reciprocal ranks
    rr_dict = defaultdict(list)

    # Loop over labels and predictions
    for true, pred in zip(y_true, y_pred):

        # Take top k predictions
        cand_list = pred[:k]

        if true in cand_list:

            # Compute rank if true label in top k predictions
            r = cand_list.index(true) + 1

            # Add reciprocal rank to list
            rr_dict[true].append(1 / r)

        else:

            # Otherwise add 0 to list
            rr_dict[true].append(0)

    # Compute mean reciprocal rank per affix
    mrr_dict = {a: np.mean(rr_list) for a, rr_list in rr_dict.items()}

    return mrr_dict
