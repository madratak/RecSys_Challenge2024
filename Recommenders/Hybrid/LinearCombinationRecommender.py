#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mauro Orazio Drago
"""

import numpy as np
import time

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class LinearCombinationRecommender(BaseRecommender, BaseItemSimilarityMatrixRecommender):
    """
    LinearCombinationRecommender combines scores from multiple models using weighted sums, 
    enabling flexible integration of different recommenders.
    """

    RECOMMENDER_NAME = "LinearCombinationRecommender"

    def __init__(self, URM_train, recommenders: list[BaseRecommender]):
        super(LinearCombinationRecommender, self).__init__(URM_train)

        self.recommenders = recommenders
        self.weights = None
        
        
    def fit(self, weights: list[float]):
        
        start_time = time.time()
        self.weights = np.array(weights)
        total_time = time.time() - start_time
        self._print("Fit completed in {:.2f} seconds.").format(total_time)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
    
        scores = np.zeros((len(user_id_array), self.URM_train.shape[1]))
        
        for recommender, weight in zip(self.recommenders, self.weights):
            scores += weight * recommender._compute_item_score(user_id_array, items_to_compute)
        
        return scores