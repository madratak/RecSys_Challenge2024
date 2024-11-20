#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mauro Orazio Drago
"""

import numpy as np
import time
from numpy import linalg as LA

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class LinearCombinationRecommender(BaseItemSimilarityMatrixRecommender):
    """
    LinearCombinationRecommender combines scores from multiple models using weighted sums, 
    enabling flexible integration of different recommenders.
    """

    RECOMMENDER_NAME = "LinearCombinationRecommender"

    def __init__(self, URM_train, recommenders: list[BaseRecommender]):
        super(LinearCombinationRecommender, self).__init__(URM_train)

        self.recommenders = recommenders
        
        
    def fit(self, weights: list[float], norm):
        
        start_time = time.time()
        self.weights = np.array(weights)
        self.norm = norm
        total_time = time.time() - start_time
        self._print(f"Fit completed in {total_time:.2f} seconds.")


    def _compute_item_score(self, user_id_array, items_to_compute=None):
    
        scores = np.zeros((len(user_id_array), self.URM_train.shape[1]))
        
        for recommender, weight in zip(self.recommenders, self.weights):
            
            item_scores = recommender._compute_item_score(user_id_array, items_to_compute)
            norm_item_scores = LA.norm(item_scores, self.norm, axis=1, keepdims=True)            
            scores += weight * (item_scores / norm_item_scores)
        
        return scores