"""
@author: Mauro Orazio Drago
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

import json
import os
import time

def fit_recommenders(metric, URM_train, ICM_all, recommenders, GH_PATH):
    """
    Fit recommenders with the best parameters for a specified evaluation metric.

    Parameters:
    - metric: Metric used for parameter optimization (e.g., "MAP" or "Recall").
    - URM_train: User-Item interaction matrix used for training.
    - ICM_all: Item-Content matrix (if needed by specific recommenders).
    - recommenders: Dictionary mapping recommender names to their classes.
    - GH_PATH: Base path to load best parameter files.

    Returns:
    - features_recommenders: Dictionary of trained recommenders.
    """
    paths_to_best_params = {
        "RP3beta": "GraphBased",
        "P3alpha": "GraphBased",
        "ItemKNNCF": "KNN",
        "ItemKNNCBF": "KNN",
        "UserKNNCF": "KNN",
        "FasterIALS": "MatrixFactorization",
        "NMF": "MatrixFactorization",
        "PureSVDItem": "MatrixFactorization/PureSVDRecommender",
        "ScaledPureSVD": "MatrixFactorization/PureSVDRecommender",
        "MultVAE": "Neural",
        "SLIMElasticNet": "SLIM",
        "SLIM_BPR": "SLIM",
    }
    
    fitted_recommenders = {}
    
    for recommender_name, recommender_class in recommenders.items():
        start_time = time.time()
        
        print(f"{recommender_name} Model - TRAINING with its best parameters.")
        
        try:
            # Initialize recommender
            recommender = recommender_class(URM_train)
        except Exception as e:
            recommender = recommender_class(URM_train, ICM_all)
        
        # Load best parameters
        param_file_path = os.path.join(
            GH_PATH, paths_to_best_params[recommender_name], 
            f"{recommender_name}Recommender", f"Optimizing{metric}", 
            f"best_params_{recommender_name}_{metric}.json"
        )
        
        try:
            with open(param_file_path, 'r') as best_params_json:
                best_params = json.load(best_params_json)
        except FileNotFoundError:
            print(f"Error: Parameter file not found for {recommender_name} at {param_file_path}. Skipping.")
            continue
        
        # Train recommender with best parameters
        recommender.fit(**best_params)
        
        # Save the trained recommender
        fitted_recommenders[recommender_name] = recommender
        
        elapsed_time = time.time() - start_time
        print(f"Training of {recommender_name} completed in {elapsed_time:.2f} seconds.\n")
    
    return fitted_recommenders


def create_XGBoost_dataframe(URM, candidate_generator_recommenders, features_recommenders, ICM, reference_URM=None, cutoff=50, categorical=False):
    """
    Create a DataFrame for a recommendation system pipeline, including additional feature generation from recommenders.

    Parameters:
    - URM: Sparse matrix for user-item interactions (training or testing set).
    - reference_URM: Sparse matrix for user-item interactions (reference set for validation or testing, optional).
    - candidate_generator_recommenders: Dictionary of recommenders generating candidate recommendations.
    - features_recommenders: Dictionary of recommenders providing additional features.
    - ICM: Sparse matrix of item-content features.
    - cutoff: Number of recommendations to consider from each recommender.
    - config: Dictionary containing configuration, e.g., for categorical encoding.

    Returns:
    - interaction_dataframe: DataFrame of features.
    - groups: Array indicating group sizes by user.
    """
    n_users, n_items = URM.shape

    # Initialize DataFrame to store recommendations
    interaction_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    interaction_dataframe.index.name = 'UserID'

    for user_id in tqdm(range(n_users)):
        recommendations = np.array([])

        # Generate candidate recommendations from candidate generators
        for candidate_recommender in candidate_generator_recommenders.values():
            candidate_recommendations = candidate_recommender.recommend(user_id, cutoff=cutoff)
            recommendations = np.union1d(recommendations, candidate_recommendations)

        interaction_dataframe.loc[user_id, "ItemID"] = recommendations

    # Expand the recommendations into rows
    interaction_dataframe = interaction_dataframe.explode("ItemID")

    if reference_URM is not None:
        # Map reference data to user-item pairs for generating labels
        reference_URM_coo = sps.coo_matrix(reference_URM)
        correct_recommendations = pd.DataFrame({"UserID": reference_URM_coo.row,
                                                "ItemID": reference_URM_coo.col})

        # Merge to identify correct recommendations
        interaction_dataframe = pd.merge(interaction_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left', indicator='Exist')
        interaction_dataframe["Label"] = interaction_dataframe["Exist"] == "both"
        interaction_dataframe.drop(columns=['Exist'], inplace=True)

        # Set UserID as the index for efficient updates
        interaction_dataframe = interaction_dataframe.set_index('UserID')

    # Add recommendation scores from feature recommenders
    for user_id in tqdm(range(n_users)):
        for rec_label, rec_instance in features_recommenders.items():
            item_list = interaction_dataframe.loc[user_id, "ItemID"].values.tolist()

            all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)

            interaction_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

    # Reset index for further processing
    interaction_dataframe = interaction_dataframe.reset_index()
    interaction_dataframe = interaction_dataframe.rename(columns={"index": "UserID"})

    # Add item popularity feature
    item_popularity = np.ediff1d(sps.csc_matrix(URM).indptr)
    interaction_dataframe['item_popularity'] = item_popularity[interaction_dataframe["ItemID"].values.astype(int)]

    # Add user profile length feature
    user_popularity = np.ediff1d(sps.csr_matrix(URM).indptr)
    interaction_dataframe['user_profile_len'] = user_popularity[interaction_dataframe["UserID"].values.astype(int)]

    # Add content-based features from ICM
    features_df = pd.DataFrame.sparse.from_spmatrix(ICM)
    interaction_dataframe = interaction_dataframe.set_index('ItemID').join(features_df, how='inner')
    interaction_dataframe = interaction_dataframe.reset_index()
    interaction_dataframe = interaction_dataframe.rename(columns={"index": "ItemID"})

    # Clean and sort data
    interaction_dataframe = interaction_dataframe.sort_values("UserID").reset_index()
    interaction_dataframe.drop(columns=['index'], inplace=True)

    # Group size for each user
    groups = interaction_dataframe.groupby("UserID").size().values

    # Encode categorical features if specified in the config
    if categorical:
        interaction_dataframe["UserID"] = interaction_dataframe["UserID"].astype("category")
        interaction_dataframe["ItemID"] = interaction_dataframe["ItemID"].astype("category")

    return interaction_dataframe, groups
