"""
@author: Mauro Orazio Drago
"""

from xgboost import XGBRanker

class XGBoostRankerRecommender:
    """XGBoost Ranker Recommender"""

    RECOMMENDER_NAME = "XGBoostRankerRecommender"

    def __init__(self, URM_train, training_dataframe, verbose=False):
        self.URM_train = URM_train
        self.training_dataframe = training_dataframe
        self.recommendations_dataframe = None
        self.model = None
        self.verbose = verbose

    def set_recommendations_dataframe(self, recommendations_dataframe):
        """
        Update the recommendations dataframe.

        Parameters:
        - recommendations_dataframe: New dataframe to be used for recommendations.
        """
        self.recommendations_dataframe = recommendations_dataframe

    def fit(self, X_train, y_train, groups, n_estimators=50, learning_rate=0.1, reg_alpha=0.1, 
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
            verbosity=2 if self.verbose else 0,
            booster=booster,
            enable_categorical=True,
            tree_method=tree_method
        )

        self.model.fit(
            X_train,
            y_train,
            group=groups,
            verbose=self.verbose
        )

    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True):
        """
        Generate recommendations for the given users.

        Parameters:
        - user_ids: List or array of user IDs.
        - cutoff: Number of top recommendations to return.
        - remove_seen_flag: If True, remove items the user has already interacted with.

        Returns:
        - A dictionary with user IDs as keys and recommended item IDs as values.
        """
        recommendations = {}

        for user_id in user_ids:
            X_user = self.recommendations_dataframe[self.recommendations_dataframe['UserID'] == user_id]

            # Predict scores
            user_scores = self.model.predict(X_user)
            X_user['score'] = user_scores

            # Sort by score
            X_user_sorted = X_user.sort_values(by='score', ascending=False)

            # Remove seen items if flag is set
            if remove_seen_flag:
                seen_items = self.training_dataframe[self.training_dataframe['UserID'] == user_id]['ItemID']
                X_user_sorted = X_user_sorted[~X_user_sorted['ItemID'].isin(seen_items)]

            # Get top items
            recommendations[user_id] = X_user_sorted['ItemID'].head(cutoff).tolist()

        return recommendations