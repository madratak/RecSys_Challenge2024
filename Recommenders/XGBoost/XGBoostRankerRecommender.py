#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mauro Orazio Drago
"""

import numpy as np
import scipy.sparse as sps
from Recommenders.BaseRecommender import BaseRecommender
from xgboost import XGBRanker

class XGBoostRankerRecommender(BaseRecommender):
    """XGBoost Ranker Recommender"""

    RECOMMENDER_NAME = "XGBoostRankerRecommender"

    def __init__(self, training_dataframe, verbose=True):
        super(XGBoostRankerRecommender, self).__init__(URM_train=None, verbose=verbose)
        self.training_dataframe = training_dataframe
        self.recommendations_dataframe = None
        self.model = None

    def set_recommendations_dataframe(self, recommendations_dataframe):
        """
        Update the training dataframe.

        Parameters:
        - new_training_dataframe: New dataframe to be used for training.
        """
        self.recommendations_dataframe = recommendations_dataframe

    def fit(self, X_train, y_train, groups, n_estimators=50, learning_rate=1e-1, reg_alpha=1e-1, 
            reg_lambda=1e-1, max_depth=5, max_leaves=0, grow_policy="depthwise", 
            objective="pairwise", booster="gbtree", random_seed=None):
        """
        Train the XGBoost Ranker model.

        Parameters:
        - X_train: DataFrame of features for training.
        - y_train: Labels indicating relevance.
        - groups: Array of group sizes by user.
        - n_estimators: Number of boosting rounds.
        - learning_rate: Boosting learning rate.
        - reg_alpha: L1 regularization term.
        - reg_lambda: L2 regularization term.
        - max_depth: Maximum tree depth.
        - max_leaves: Maximum number of leaves per tree.
        - grow_policy: Growth policy for trees.
        - objective: Ranking objective function.
        - booster: Type of booster to use.
        - random_seed: Random seed for reproducibility.
        """

        # Initialize and train the XGBRanker
        self.model = XGBRanker(
            objective='rank:{}'.format(objective),
            n_estimators=int(n_estimators),
            random_state=random_seed,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_depth=int(max_depth),
            max_leaves=int(max_leaves),
            grow_policy=grow_policy,
            verbosity=0,  # Adjust verbosity if needed
            booster=booster,
            enable_categorical=True,
            tree_method="hist"  # Supported tree methods are `gpu_hist`, `approx`, and `hist`.
        )

        self.model.fit(
            X_train,
            y_train,
            group=groups,
            verbose=True
        )

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Compute item scores for given users.

        Parameters:
        - user_id_array: Array containing the user indices.
        - items_to_compute: Array containing items to compute scores for. If None, compute for all items.

        Returns:
        - Array of scores with shape (len(user_id_array), n_items).
        """
        scores = []
        for user_id in user_id_array:
            X_user = self.recommendations_dataframe[self.recommendations_dataframe['UserID'] == user_id]

            # Filter by items_to_compute if provided
            if items_to_compute is not None:
                user_data = user_data[user_data['ItemID'].isin(items_to_compute)]

            # Predict scores
            user_scores = self.model.predict(X_user)
            scores.append(user_scores)

        return np.array(scores)
