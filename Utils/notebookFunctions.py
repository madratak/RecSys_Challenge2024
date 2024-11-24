#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/03/2019

@author: Mauro Orazio Drago
"""
from github import Github, Auth
import pandas as pd

def get_repo_from_github(token):
    """
    Gets the repository 'RECsys_Challenge2024' from GitHub.

    Parameters:
    - token: authentication key for GitHub.
    """
    # Authenticate using a personal access token
    auth_token = Auth.Token(token)
    github_client = Github(auth=auth_token)

    # Define the repository name you want to find
    target_repo_name = 'RECsys_Challenge2024'
    repo = None

    # Search for the repository in the user's repositories
    try:
        for repository in github_client.get_user().get_repos():
            if repository.name == target_repo_name:
                repo = repository
                print(f"Repository '{target_repo_name}' found.")
                break
        if repo is None:
            print(f"Repository '{target_repo_name}' not found.")
    except Exception as e:
        print("An error occurred while accessing the repositories:", e)

    return repo


def upload_file(filepath_kaggle, filepath_github, commit_message, repo=None):
    """
    Uploads a file from Kaggle to GitHub, updating it if it already exists in the repository,
    or creating it if it does not.

    Parameters:
    - filepath_kaggle: Path to the file in the Kaggle environment.
    - filepath_github: Target path in the GitHub repository where the file should be uploaded.
    - commit_message: Message for the commit on GitHub.
    """
    if repo is None:
        raise ValueError("Repository is not defined. Make sure to call 'get_repo_from_github' to get the repo instance before using this function.")
    
    try:
        
        # Check if the file already exists in the GitHub repository
        contents = repo.get_contents(filepath_github)
        
        # If it exists, update the file
        with open(filepath_kaggle, "rb") as file:
            repo.update_file(
                contents.path, commit_message, file.read(), contents.sha
            )
        print(f"File '{filepath_github}' updated successfully.")
    
    except Exception as e:
        
        # If the file does not exist, create it
        with open(filepath_kaggle, "rb") as file:
            repo.create_file(
                filepath_github, commit_message, file.read()
            )
        print(f"File '{filepath_github}' created successfully.")


def create_submission(data_target_users_test, recommender_instance, output_file, cutoff=10):
    """
    Creates a submission file for a recommendation system challenge.

    Parameters:
    - data_target_users_test (pd.DataFrame): A DataFrame containing the IDs of the target users for whom recommendations need to be made.
    - recommender_instance (Recommender): The recommender system instance used to generate the recommendations.
    - cutoff (int): The number of items to recommend for each user (default is 10).
    - output_file (str): The file path where the submission file will be saved. 
    """

    target_result = []

    for target in data_target_users_test["user_id"]:
        target_result.append(recommender_instance.recommend(target, cutoff=cutoff, remove_seen_flag=True))

    user_ids = data_target_users_test["user_id"]
    formatted_data = {
        "user_id": user_ids,
        "item_list": [" ".join(map(str, items)) for items in target_result]
    }

    submission_df = pd.DataFrame(formatted_data)
    submission_df.to_csv(output_file, index=False, header=["user_id", "item_list"])

    print(f"Submission file saved as {output_file}")