# Predicting NBA Player Retention

### Project Description

The NBA is widely considered to be the best basketball league in the world, and has grown over its seven-decade existence into a multibillion-dollar industry.  Central to this industry is the problem of roster construction, as team performance depends critically on selecting productive players for all 15 roster spots.

In this project, we aim to perform a _novel analysis_ of NBA statistics, salary, and transaction data in order to determine whether or not a given player will be in the NBA in the next season (i.e., we predict _NBA player retention_).  The resulting model has the potential to aid in the selection of players toward the end of the roster, which has long been one of the most challenging aspects of team construction.

**Old stuff below here**

### Description of Data

We plan to predict whether or not a player is moved using the following datasets:
* Player statistics during the season, obtained from NBA.com through the Python library nba_api
* Player transactions from prosportstransactions.com through the pro_sports_transactions Python library
* Player salaries, team total salary, salary cap data scraped from the web (see, for example, https://www.basketball-reference.com/)

### KPIs

* Accuracy
* Confusion matrix
* ROC curve, AUC
* Recall, precision, F-score
* Precision-recall curve
* Cross-entropy/log loss

### Stakeholders

Given the popularity of the sport and the amount of money involved, there are many potential stakeholders who would be interested in predicting movement of players. These include the players themselves and their agents as they would have a model that can estimate the likelihood of being moved and can act on that when it comes to contract negotiations. The teams’ front offices and coaches would be the next most natural stakeholders. In this case it isn’t only for the potential for the predictive power of the model, but also for the potential to impact decision making. For instance if our model indicates that a player is likely to be moved, maybe that can become something of a self-fulfilling prophesy in a way when it comes to the front office. A predictive model also has value to sports analysts. Lastly, with the recent boom in popularity from more accessible legalized sports betting, oddsmakers and bettors would see value in predicting player movement. This would be both direct (for example offering odds that a player moves as prop bets) and indirectly (for example will a team become better if they are likely to move an underperforming player).

