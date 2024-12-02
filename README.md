# Predicting NBA Player Retention

## Project Description

The NBA is widely considered to be the best basketball league in the world, and has grown over its seven-decade existence into a multibillion-dollar industry.  Central to this industry is the problem of roster construction, as team performance depends critically on selecting productive players for all 15 roster spots.

In this project, we aim to perform a _novel analysis_ of NBA statistics, salary, and transaction data in order to determine whether or not a given player will be in the NBA in the next season (i.e., we predict _NBA player retention_).  The resulting model has the potential to aid in the selection of players toward the end of the roster, which has long been one of the most challenging aspects of team construction.

## Stakeholders

A model for predicting which NBA players will play in the NBA in the next season is valuable to many decision makers including, but not limited to,
* NBA front offices planning roster changes and deciding which players to invest in,
* sport bettors making long term bets,
* and advertisers planning sponsorships.

## Key Performance Indicators (KPIs)

We use the following metrics to evaluate our model:
* Balanced accuracy
* Precision
* Recall (sensitivity)
* Specificity
* Negative predictive value (NPV)

## Data Collection and Cleaning

[Counting statistics](Data/CountingStats) data was collected from the official NBA site using an [API](https://github.com/swar/nba_api) and custom web scraping tools. [Advanced statistics](Data/AdvancedStats) were collected from [Kaggle](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats). [Transaction data](Data/TransactionData) and [salary data](Data/SalaryData) were scraped from [Basketball-Reference.com](https://www.basketball-reference.com) and [hoopshype.com](https://hoopshype.com/), respectively, using BeautifulSoup. The data was cleaned and missing data (e.g. missing salaries) were imputed. The data was split into a [training set](train_data.csv) consisting of seasons 1990-91 through 2016-17 and a [test set](test_data.csv) with seasons 2017-18 through 2022-23.

## Exploratory Data Analysis

We initially aimed to predict player transactions (whether a given player would be traded/waived), but this proved challenging due to weak correlations between statistics and transaction data (magnitude ~0.05). We found appreciable correlations with whether a player stayed in the NBA in the following season (NBA player retention), however.

![image](https://github.com/user-attachments/assets/a3c7cfb0-f21f-4817-934c-66f2282de4ba)

## Modeling

Predicting NBA player retention in the following season is a classification problem with time series structure and imbalanced classes. We evaluated 10 models using walk-forward cross validation with the balanced accuracy score as our central metric. Most models were trained on an augmented training set produced using the Synthetic Minority Oversampling Technique (SMOTE) to balance classes.

| Rank | Model | Balanced Accuracy | Precision | Recall (sensitivity) | NPV | Specificity | Hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost Classifier | 0.8116 | 0.9482 | 0.8156 | 0.5041 | 0.8076 | `n_estimators=400, learning_rate=0.005` |
| 2 | Logistic Regression w/ SMOTE | 0.8079 | 0.9566 | 0.7654 | 0.4561 | 0.8503 | `C=0.00125` |
| 3 | Random Forest Classifier | 0.8071 | 0.9494 | 0.7971 | 0.4824 | 0.8170 | `n_estimators=50, max_depth=5` |
| 4 | PCA + KNN | 0.8065 | 0.9664 | 0.7211 | 0.4253 | 0.8920 | `pca__n_components=30, knn__n_neighbors=84` |
| 5 | PCA + QDA | 0.8051 | 0.9553 | 0.7649 | 0.4547 | 0.8453 | `pca__n_components=30`, `qda__reg_param=0.3` |
| 6 | AdaBoost Classifier | 0.8024 | 0.9498 | 0.7833 | 0.4674 | 0.8214 | `n_estimators=500, learning_rate=0.1` |
| 7 | Decision Tree Classifier | 0.8009 | 0.9485 | 0.7868 | 0.4710 | 0.8150 | `criterion='gini', max_depth=5` |
| 8 | LDA | 0.8009 | 0.9637 | 0.7189 | 0.4208 | 0.8828 | `shrinkage=0.8` |
| 9 | Gaussian Naive Bayes | 0.7876 | 0.9713 | 0.6593 | 0.3839 | 0.9158 | `var_smoothing=0.1` |
| 10 | Logistic Regression | 0.7308 | 0.8950 | 0.9385 | 0.6637 | 0.5232 | `C=1` |

## Final Results

The best-performing model was XGBoost with hyperparameters chosen to maximize balanced accuracy on the (augmented) training data. We evaluated this model’s performance on the test set using walk-forward testing (with an expanding window) and achieved a balanced accuracy of about 81%. The model achieved a precision of about 95%, reflecting high confidence in predicting players who stayed, and a recall of about 78%, ensuring most players who stayed were correctly identified. Its specificity of about 84% further highlights its ability to correctly identify players who left. While the negative predictive value (NPV) of about 50% appears lower, this is expected given the model's prioritization of minimizing false positives (high precision) and achieving balanced accuracy (strong recall and specificity) on such imbalanced data. Incorporating additional factors such as G-League data, contract terms, and injuries could likely improve the NPV.

| Test Season | Balanced Accuracy | Precision | Recall (sensitivity) | NPV | Specificity |
|-----------|-----------|-----------|---------|--------|-------------|
| 2017-18 | 0.8054 | 0.9405 | 0.7670 | 0.5294 | 0.8438 |
| 2018-19 | 0.8079 | 0.9345 | 0.7850 | 0.5567 | 0.8308 |
| 2019-20 | 0.8041 | 0.9570 | 0.7678 | 0.4389 | 0.8404 |
| 2020-21 | 0.7878 | 0.9529 | 0.7644 | 0.4078 | 0.8111 |
| 2021-22 | 0.8459 | 0.9375 | 0.8314 | 0.6697 | 0.8605 |
| 2022-23 | 0.7982 | 0.9605 | 0.7556 | 0.4022 | 0.8409 |
| **Avg.** | **0.8082**| **0.9471**| **0.7785** | **0.5008** | **0.8379** |
| Baseline (always predicts 1) | 0.5000 | 0.7876 | 1.0000 | 0.0000 | 0.0000 |

In any case, looking at a random sample of our model’s false negative predictions, many of these players have at some point gone down to the G-league, suggesting that although these players were misclassified, our model was still able to successfully identify them as “fringe” players who were at risk of leaving the NBA in the near future.

![image](https://github.com/user-attachments/assets/0d915908-d279-4123-9f91-946ccd623c91)

## Future Directions

To build on our existing model, some additional data that we could incorporate include:
* contract terms,
* injury information,
* G-League data,
* and salary cap and Collective Bargaining Agreement (CBA) data.
Additionally, we could expand our model of NBA players to incorporate other professional basketball leagues, for example the EuroLeague, and track movement across leagues. We could also expand the problem to focus on assigning probabilities that a player will be waived, traded, or not retained in the league at all, rather than pure classification.
