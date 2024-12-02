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

## Modeling

Predicting NBA player retention in the following season is a classification problem with time series structure and imbalanced classes. We evaluated 10 models using walk-forward cross validation with the balanced accuracy score as our central metric. Most models were trained on an augmented training set produced using the Synthetic Minority Oversampling Technique (SMOTE) to balance classes.

## Final Results

The best-performing model was XGBoost with hyperparameters chosen to maximize balanced accuracy on the (augmented) training data. We evaluated this model’s performance on the test set using walk-forward testing (with an expanding window) and achieved a balanced accuracy of about 81%. The model achieved a precision of about 95%, reflecting high confidence in predicting players who stayed, and a recall of about 78%, ensuring most players who stayed were correctly identified. Its specificity of about 84% further highlights its ability to correctly identify players who left. While the negative predictive value (NPV) of about 50% appears lower, this is expected given the model's prioritization of minimizing false positives (high precision) and achieving balanced accuracy (strong recall and specificity) on such imbalanced data. Incorporating additional factors such as G-League data, contract terms, and injuries could likely improve the NPV. In any case, looking at a random sample of our model’s false negative predictions, many of these players have at some point gone down to the G-league, suggesting that although these players were misclassified, our model was still able to successfully identify them as “fringe” players who were at risk of leaving the NBA in the near future.

## Future Directions

To build on our existing model, some additional data that we could incorporate include:
* contract terms,
* injury information,
* G-League data,
* and salary cap and Collective Bargaining Agreement (CBA) data.
Additionally, we could expand our model of NBA players to incorporate other professional basketball leagues, for example the EuroLeague, and track movement across leagues. We could also expand the problem to focus on assigning probabilities that a player will be waived, traded, or not retained in the league at all, rather than pure classification.
