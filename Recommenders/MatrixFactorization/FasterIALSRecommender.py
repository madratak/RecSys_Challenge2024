"""
@author: Mauro Orazio Drago
"""

import implicit

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

class FasterIALSRecommender(BaseMatrixFactorizationRecommender):
    """FasterIALSRecommender: Implicit Alternating Least Squares Recommender with faster training"""

    RECOMMENDER_NAME = "IALSRecommender"

    def __init__(self, URM_train, verbose=True):
        super(FasterIALSRecommender, self).__init__(URM_train, verbose=verbose)
        self.als_model = None

def fit(self, factors=64, regularization=0.05, iterations=15, alpha=2.0):
        
        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha
        )

        self.als_model.fit(self.URM_train)

        self.USER_factors = self.als_model.user_factors
        self.ITEM_factors = self.als_model.item_factors