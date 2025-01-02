"""
@author: Mauro Orazio Drago
"""

import numpy as np
from Recommenders.BaseRecommender import BaseRecommender
from xgboost import XGBRanker

class XGBoostRankerRecommender(BaseRecommender):
    """XGBoost Ranker Recommender"""

    RECOMMENDER_NAME = "XGBoostRankerRecommender"

    def __init__(self, URM_train, X_train, y_train, evaluation_dataframe, verbose=True):
        super(XGBoostRankerRecommender, self).__init__(URM_train)

        self.X_train = X_train
        self.y_train = y_train
        self.evaluation_dataframe = evaluation_dataframe
        self.model = None
        self.verbose = verbose

    def fit(self, groups, n_estimators=50, learning_rate=0.1, reg_alpha=0.1, 
            reg_lambda=0.1, max_depth=5, max_leaves=0, grow_policy="depthwise", 
            objective="pairwise", booster="gbtree", random_seed=None, tree_method="hist"):
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
        - tree_method: Tree construction algorithm (e.g., 'hist', 'approx', 'gpu_hist').
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
            verbosity=0, # if 2 self.verbose else 0,
            booster=booster,
            enable_categorical=True,
            tree_method=tree_method
        )

        self.model.fit(
            self.X_train,
            self.y_train,
            group=groups,
            verbose=self.verbose
        )

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Compute item scores for the given users.
    
        Parameters:
        - user_id_array: Array of user IDs for whom to compute scores.
        - items_to_compute: Array of item IDs to score. If None, scores all items.
    
        Returns:
        - scores_matrix: A 2D NumPy array of shape (len(user_id_array), n_items),
          where n_items corresponds to the total number of items in the dataset.
          Unscored items will have a value of -np.inf.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call the fit method first.")
    
        # Initialize score matrix
        scores_matrix = np.full((len(user_id_array), self.n_items), -np.inf, dtype=np.float32)
    
        # Ensure item IDs are mapped to indices
        item_index_map = {item_id: idx for idx, item_id in enumerate(range(self.n_items))}
    
        for user_index, user_id in enumerate(user_id_array):
            # Filter user-specific data
            X_user = self.evaluation_dataframe[self.evaluation_dataframe['UserID'] == user_id]
    
            # If scoring only a subset of items
            if items_to_compute is not None:
                user_features = user_features[user_features['ItemID'].isin(items_to_compute)]
    
            # Predict scores
            scores = self.model.predict(X_user)
    
            # Map scores back to the full item space
            item_ids = user_features['ItemID'].values
            item_indices = [item_index_map[item_id] for item_id in item_ids]
            scores_matrix[user_index, item_indices] = scores
    
        return scores_matrix