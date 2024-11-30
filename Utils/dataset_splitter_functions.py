import numpy as np
import pandas as pd
from github import Github
import os
from kaggle_secrets import UserSecretsClient
import pandas as pd
import matplotlib.pyplot as plt


# A function to upload files to GitHub (it is more convenient to use this function than the one in the notebookFunctions.py :P)
def upload_file_2(filepath_kaggle, filepath_github, commit_message, repo_name, github_token):
    """
    Uploads a file from Kaggle to GitHub, updating it if it already exists or creating it otherwise.
    
    Parameters:
    - filepath_kaggle: Path to the file in the Kaggle environment.
    - filepath_github: Target path in the GitHub repository.
    - commit_message: Commit message for the GitHub update/create action.
    - repo_name: Name of the GitHub repository (e.g., "username/repository").
    - github_token: Personal Access Token for authenticating with GitHub.
    """
    try:
        # Authenticate with GitHub
        g = Github(github_token)
        repo = g.get_repo(repo_name)

        # Check if the file already exists in the repo
        try:
            contents = repo.get_contents(filepath_github)
            
            # Update the file if it exists
            with open(filepath_kaggle, "rb") as file:
                repo.update_file(contents.path, commit_message, file.read(), contents.sha)
            print(f"File '{filepath_github}' updated successfully.")

        except Exception as e:
            # If the file does not exist, create it
            with open(filepath_kaggle, "rb") as file:
                repo.create_file(filepath_github, commit_message, file.read())
            print(f"File '{filepath_github}' created successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")



def splitting_validation_dataset (URM_validation, 
                                  n_groups = 5,  # Number of groups you want
                                  destination_folder = "Dataset/splitted_datasets/validation",
                                  seed=42, 
                                  save_to_github=True, 
                                  repo_name = "madratak/RECsys_Challenge2024"):
    
    np.random.seed(seed)  # Set seed for reproducibility

    # Step 1: Calculate interaction counts for each user
    user_interaction_counts = np.diff(URM_validation.indptr)  # Interaction counts for each user

    # Step 2: Sort users by interaction counts
    sorted_user_indices = np.argsort(user_interaction_counts)  # Indices of users sorted by interaction counts

    # Step 3: Determine group sizes
    n_users = len(sorted_user_indices)
    users_per_group = n_users // n_groups
    remaining_users = n_users % n_groups  # Number of users that cannot be evenly distributed

    # Step 4: Split users into groups and save as CSV files
    csv_filepaths = []  # Store file paths to upload to GitHub
    groupped_datasets = []
    start_idx = 0

    for group_id in range(n_groups):
        # Distribute the remaining users to the first `remaining_users` groups
        extra_user = 1 if group_id < remaining_users else 0
        end_idx = start_idx + users_per_group + extra_user  # Adjust end index for the group
        
        group_user_indices = sorted_user_indices[start_idx:end_idx]  # Get user IDs in the group
        group_data = pd.DataFrame({
            "user_id": group_user_indices,
            "interaction_count": user_interaction_counts[group_user_indices]
        })
        
        # Save to CSV file
        csv_filename = f"group_{group_id}.csv"
        group_data.to_csv(csv_filename, index=False)
        csv_filepaths.append(csv_filename)  # Keep track of files
        groupped_datasets.append(group_data)

        print(f"Group {group_id}: Saved {len(group_user_indices)} users to {csv_filename}")
        
        # Update start_idx for the next group
        start_idx = end_idx


    if save_to_github:
        # Step 5: Upload files to GitHub
        # Replace these with your GitHub details
        
        github_token = UserSecretsClient().get_secret("Token")
        commit_message = "Added grouped user (validation) interaction files"

        for filepath in csv_filepaths:
            filepath_github = f"{destination_folder}/{os.path.basename(filepath)}"  # Path in the GitHub repo
            upload_file_2(filepath, filepath_github, commit_message, repo_name, github_token)

    return groupped_datasets
    
    

def splitting_target_dataset(group_number = 5,
                             data_train_path = "/kaggle/working/RECsys_Challenge2024/Dataset/data_train.csv",
                             data_target_users_path = "/kaggle/working/RECsys_Challenge2024/Dataset/data_target_users_test.csv",
                             save_to_github = True,
                             destination_folder = "Dataset/splitted_datasets/target",
                             repo_name = "madratak/RECsys_Challenge2024"):
 

    # Load datasets
    data_train = pd.read_csv(data_train_path)
    data_target_users = pd.read_csv(data_target_users_path)

    # Step 2: Count interactions for each user
    user_interactions = data_train.groupby("user_id").size().reset_index(name="interaction_count")

    # Step 3: Filter for target users
    target_user_interactions = user_interactions[user_interactions["user_id"].isin(data_target_users["user_id"])].copy()

    # Step 4: Sort by interaction count
    target_user_interactions = target_user_interactions.sort_values("interaction_count").reset_index(drop=True)

    # Step 5: Divide into 5 approximately equal groups
    n_users = len(target_user_interactions)
    group_size = n_users // group_number  # Number of users in each group
    remainder = n_users % group_number    # Handle uneven divisions

    groups = []
    # Assign users to groups
    start = 0
    for i in range(group_number):
        end = start + group_size + (1 if i < remainder else 0)  # Distribute remainder across first groups
        groups.append(target_user_interactions.iloc[start:end])
        
        start = end

    csv_filepaths = []
    # Step 6: Save or display groups
    for i, group in enumerate(groups, start=1):
        group.to_csv(f"group_{i}.csv", index=False)
        print(f"Group {i} has {len(group)} users.")
        csv_filepaths.append(f"group_{i}.csv")  # Keep track of files



    if save_to_github:
        # Step 6: Upload files to GitHub
        # Replace these with your GitHub details
        github_token = UserSecretsClient().get_secret("Token")
        commit_message = "Added grouped user(target) interaction files"

        for filepath in csv_filepaths:
            filepath_github = f"{destination_folder}/{os.path.basename(filepath)}"  # Path in the GitHub repo
            upload_file_2(filepath, filepath_github, commit_message, repo_name, github_token)


    return groups

def Plot_boxPlot(groups,
                 group_number = 5):

    # Load the groups from the previous code
    groups = []
    for i in range(1, group_number+1):
        group = pd.read_csv(f"group_{i}.csv")
        groups.append(group)

    # Create subplots for the box plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 6))  # 2 columns, 3 rows
    axes = axes.flatten()  # Flatten to make it easier to index

    # Plot each group with horizontal orientation
    for i, group in enumerate(groups):
        axes[i].boxplot(group["interaction_count"], notch=True, showmeans=True, vert=False)  # Horizontal boxplot
        axes[i].set_title(f"Group {i+1}", fontsize=14)
        axes[i].set_xlabel("Interaction Count", fontsize=12)
        axes[i].set_ylabel("Group", fontsize=12)
        axes[i].set_yticks([])  # Remove y-axis ticks for simplicity

    # Remove empty subplot (since we have only 5 plots)
    if len(groups) < len(axes):
        for j in range(len(groups), len(axes)):
            fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()
