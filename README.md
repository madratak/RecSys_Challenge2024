<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/it/b/be/Logo_Politecnico_Milano.png" width="250"/>
</p>

# RecSys Challenge 2024 @ POLIMI

## Overview

This repository was developed for the [Recommender System Competition 2024](https://www.kaggle.com/competitions/recommender-system-2024-challenge-polimi) held at Politecnico di Milano on Kaggle. 

The goal of the competition was to predict which new books a user would interact with, using a variety of recommender algorithms. The dataset provided by the organizers consisted of:

- **1.9 million interactions**
- **35k users**
- **38k items (books)**
- **94k item features**

The evaluation metric was **MAP\@10**.

## Index

1. [Model](#model)
2. [Results](#results)
3. [Team](#team)
4. [Acknowledgments](#acknowledgments)

## Model

The solution is based on an **XGBoostRanker**, integrated with a robust candidate generator and feature engineering pipeline.

### Candidate Generators

The following generators were initially fitted to maximize **Recall\@50** to ensure a comprehensive candidate pool.

1. **SLIMElasticNet**
2. **ItemKNN Collaborative Filtering (ItemKNN-CF)**
3. **RP3beta**

### Features

Features were extracted from the outputs of the following models, all tuned to maximize **MAP@10**:

- **RP3beta**
- **P3alpha**
- **ItemKNNCF**
- **ItemKNNCBF**
- **UserKNNCF**
- **FasterIALS**
- **NMF**
- **PureSVDItem**
- **ScaledPureSVD**
- **MultVAE**
- **SLIMElasticNet**
- **SLIM_BPR**

Furthermore, additional features were incorporated, including statistical metrics (mean, median, standard deviation, skewness, and kurtosis) to capture variability in item rankings, the count of items appearing in the top 10 across all recommenders, normalized item popularity based on interactions, and user profile length, calculated from the number of items a user interacted with.

Here’s how the table can be displayed:

```python
dataframe_XGBoostNoCont.head()
```

|    | UserID |  ItemID  | Label | feature_recommender_count | RP3beta_score | RP3beta_position | P3alpha_score | P3alpha_position | ItemKNNCF_score | ItemKNNCF_position | ... | SLIM_BPR_position | Mean_Position | Std_Position | Min_Position | Max_Position | Median_Position | Skew_Position | Kurtosis_Position | item_popularity | user_profile_len |
|----|--------|----------|-------|--------------------------|---------------|------------------|---------------|------------------|-----------------|--------------------|-----|-------------------|---------------|--------------|--------------|--------------|-----------------|----------------|-------------------|-----------------|-------------------|
|  0 |      0 |    399.0 | False |                        7 |        0.6433 |              6.0 |        0.4695 |             24.0 |          0.5144 |                 9  | ... |                 1 |        19.500 |        24.13 |          1.0 |         71.0 |             7.5 |          1.151 |            -0.043 |          0.1468 |            0.0178 |
|  1 |      0 |  13736.0 | False |                        0 |        0.2683 |             43.0 |        0.3418 |             39.0 |          0.0000 |                51  | ... |                53 |        36.917 |        17.24 |         11.0 |         59.0 |            41.0 |         -0.343 |            -1.406 |          0.0602 |            0.0178 |
|  2 |      0 |  11966.0 | False |                        2 |        0.0000 |             73.0 |        0.0000 |             77.0 |          0.0000 |                46  | ... |                52 |        39.500 |        27.27 |          7.0 |         87.0 |            28.5 |          0.541 |            -1.068 |          0.0317 |            0.0178 |
|  3 |      0 |  11919.0 | False |                        3 |        0.5328 |             11.0 |        0.6404 |              8.0 |          0.7749 |                 3  | ... |                24 |        24.333 |        15.93 |          3.0 |         57.0 |            25.5 |          0.462 |            -0.545 |          0.0222 |            0.0178 |
|  4 |      0 |  11548.0 | False |                        0 |        0.4916 |             13.0 |        0.0000 |             68.0 |          0.4807 |                11  | ... |                39 |        38.750 |        23.85 |         11.0 |         81.0 |            36.5 |          0.332 |            -1.110 |          0.0264 |            0.0178 |

## Results

The model achieved significant results in the competition:

- **Public Leaderboard:** 0.10522 (1st place)  
- **Private Leaderboard:** 0.10347 (1st place)

## Team

This repository was developed by "The Overfitters" team:

- [Mauro Orazio Drago](https://github.com/madratak)  
- [Sajjad Shaffaf](https://github.com/sajad002) 

## Acknowledgments

This project is based on [RecSys_Course_AT_PoliMi](https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi)’s repository.
