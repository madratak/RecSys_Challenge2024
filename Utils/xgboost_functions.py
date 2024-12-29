"""
@author: Mauro Orazio Drago
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

import zipfile
import os
import string

import json
import os
import time

# from Utils.notebookFunctions import upload_file
from Utils.seconds_to_biggest_unit import *

import os
import zipfile

def put_dataset_zipped_into_local_repo(input_directory, output_zip_file):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_zip_file)
    os.makedirs(output_dir, exist_ok=True)  # Creates missing directories if necessary
    
    # Step 1: Find the _model_state directory
    model_state_directory = os.path.join(input_directory, '_model_state')

    # Ensure _model_state exists before proceeding
    if not os.path.exists(model_state_directory):
        raise FileNotFoundError(f"The '_model_state' directory was not found in {input_directory}")

    # Step 2: Create the inner zip file for the _model_state directory
    inner_zip_path = os.path.join(output_dir, '_model_state.zip')
    with zipfile.ZipFile(inner_zip_path, 'w', zipfile.ZIP_DEFLATED) as inner_zip:
        for root, dirs, files in os.walk(model_state_directory):
            for file in files:
                file_path = os.path.join(root, file)
                inner_zip.write(file_path, os.path.relpath(file_path, model_state_directory))
    
    # Step 3: Create the outer zip file and add the rest of the Model directory excluding _model_state
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as outer_zip:
        for root, dirs, files in os.walk(input_directory):
            # Skip the _model_state directory when adding files
            if '_model_state' in dirs:
                dirs.remove('_model_state')  # This ensures _model_state is not added directly to outer zip
            for file in files:
                file_path = os.path.join(root, file)
                outer_zip.write(file_path, os.path.relpath(file_path, input_directory))
        
        # Add the inner zip (_model_state.zip) into the outer zip
        outer_zip.write(inner_zip_path, arcname='_model_state.zip')
    
    # Clean up the inner zip file after embedding it in the outer zip
    os.remove(inner_zip_path)

def fit_recommenders(metric, phase, URM_train, ICM_all, recommenders, GH_PATH, type_recommenders, repo):
    """
    Fit recommenders with the best parameters for a specified evaluation metric and training phase.

    Parameters:
    - metric: Metric used for parameter optimization (e.g., "MAP" or "Recall").
    - phase: Current training phase (e.g., "Train", "TrainValidation", "TrainValidationTest").
    - URM_train: User-Item interaction matrix used for training.
    - ICM_all: Item-Content matrix (if needed by specific recommenders).
    - recommenders: Dictionary mapping recommender names to their classes.
    - GH_PATH: Base path to load best parameter files.

    Returns:
    - fitted_recommenders: Dictionary of trained recommenders.
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

    phases = ["Train", "TrainVal", "TrainValTest"]

    types = {
        "cg": "CandidateGenerator",
        "f": "Feature"
    }
    
    if phase not in phases:
        raise ValueError(f"Invalid phase: '{phase}'. Must be one of {phases}.")

    if type_recommenders not in types:
        raise ValueError(f"Invalid type: '{type_recommenders}'. Must be one of {types}.")

    fitted_recommenders = {}

    for recommender_name, recommender_class in recommenders.items():
        start_time = time.time()

        print(f"{recommender_name} Model - TRAINING with its best parameters.")

        try:
            # Initialize recommender
            recommender = recommender_class(URM_train)
        except Exception:
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

        # Check if the model is already saved
        dataset_path = f"/kaggle/input/best-{recommender_name.lower()}-{metric.lower()}-{phase.lower()}-tuned"

        if os.path.exists(dataset_path):

            saved_model_file_path = os.path.join(
                GH_PATH, "XGBoost", types[type_recommenders], phase, 
                f"best_{recommender_name}_{metric}_{phase}_tuned.zip"
            )

            put_dataset_zipped_into_local_repo(dataset_path, saved_model_file_path)
            
            print(f"Model for {recommender_name} already exists. Loading the saved model.")
            recommender.load_model(folder_path=os.path.dirname(saved_model_file_path), 
                                file_name=os.path.basename(saved_model_file_path).replace('.zip', ''))
            fitted_recommenders[recommender_name] = recommender
            print("\n")
            continue

        # Train recommender with best parameters
        recommender.fit(**best_params)

        # Save the trained recommender
        fitted_recommenders[recommender_name] = recommender

        elapsed_time = time.time() - start_time
        print(f"Training of {recommender_name} completed in {seconds_to_biggest_unit(elapsed_time)} seconds.\n")

        # Save the trained model locally
        # recommender.save_model(folder_path='/kaggle/working/', file_name=f"best_{recommender_name}_{metric}_{phase}_tuned")

        # zip_file_path = f"/kaggle/working/best_{recommender_name}_{metric}_{phase}_tuned.zip"

        # # 50MB limitation management for GitHub pushes
        # try:
        #     upload_file(
        #         zip_file_path,  
        #         saved_model_file_path, 
        #         f"{recommender_name} recommender tuned with best parameters for {phase} (from Kaggle notebook)",
        #         repo
        #     )
        # except Exception as e:
        #     print(f"Error while uploading {zip_file_path} to GitHub: {e}")

    return fitted_recommenders


def create_XGBoost_dataframe(URM, candidate_generator_recommenders, features_recommenders, ICM, reference_URM=None, cutoff=50, categorical=False, contents=False):
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
            item_list = np.array(item_list, dtype=int)

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
    if contents:
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
