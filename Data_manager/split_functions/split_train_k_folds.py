#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Mauro Orazio Drago
"""

import numpy as np
import scipy.sparse as sps
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

def split_train_k_folds(URM_train, k=10):
    """
    Split the URM into k folds for cross-validation.
    :param URM_all: Full user-item interaction matrix (sparse matrix).
    :param k: Number of folds for cross-validation.
    :return: List of (train, validation) pairs for each fold.
    """
    assert k > 1, "Number of folds must be greater than 1."
    
    # Get the number of users and items
    num_users, num_items = URM_train.shape

    # Shuffle the interaction indices randomly
    URM_train = sps.coo_matrix(URM_train)
    indices_for_sampling = np.arange(0, URM_train.nnz, dtype=np.int32)
    np.random.shuffle(indices_for_sampling)

    # Split the interactions into k folds
    fold_size = len(indices_for_sampling) // k
    fold_indices = [indices_for_sampling[i * fold_size: (i + 1) * fold_size] for i in range(k)]

    folds = []

    for fold_idx in range(k):
        # Determine validation set (the current fold)
        validation_indices = fold_indices[fold_idx]
        # The remaining indices will be part of the training set
        train_indices = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])

        # Create the training set
        URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
        URM_train_builder.add_data_lists(URM_train.row[train_indices], URM_train.col[train_indices], URM_train.data[train_indices])
        URM_train_split = URM_train_builder.get_SparseMatrix()
        URM_train_split = sps.csr_matrix(URM_train_split)

        # Create the validation set
        URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
        URM_validation_builder.add_data_lists(URM_train.row[validation_indices], URM_train.col[validation_indices], URM_train.data[validation_indices])
        URM_validation_split = URM_validation_builder.get_SparseMatrix()
        URM_validation_split = sps.csr_matrix(URM_validation_split)

        user_no_item_train_split = np.sum(np.ediff1d(URM_train_split.indptr) == 0)
        user_no_item_validation_split = np.sum(np.ediff1d(URM_validation_split.indptr) == 0)

        if user_no_item_train_split != 0 or user_no_item_train_split != 0:
            print("FOLD N. {}".format(fold_idx))
            if user_no_item_train_split != 0:
                print("    Warning: {} ({:.2f} %) of {} users have no train items".format(user_no_item_train_split, user_no_item_train_split/num_users*100, num_users))
            if user_no_item_validation_split != 0:
                print("    Warning: {} ({:.2f} %) of {} users have no sampled items".format(user_no_item_validation_split, user_no_item_validation_split/num_users*100, num_users))

        # Append the train-validation pair for this fold
        folds.append((URM_train_split, URM_validation_split))

    return folds
