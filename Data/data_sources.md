# Description of Subfolders

In this folder, there are subfolders that contain how .csv files were gathered (CountingStats, PlayerIdentification, SalaryData, TransactionData) and a folder (DataToMerge) that contains the process for merging the different .csv files into one file. Below, we provide a brief description of each subfolder. 

### AdvancedStats
This subfolder contains two subfolders 'data_cleaning' that contains the advanced_stats.csv file, with a clean_advanced_stats.ipynb notebook that cleans that csv file. The other subfolder 'data_collection' has an Advanced.csv file that 
### CountingStats
To collect the NBA counting stats, we make use of the nba_api library to pull the counting stats for all players listed on the official NBA website. 

### PlayerIdentification
There are two notebooks in this file. One 'Adding_ID_Numbers.ipynb' which adds player identification numbers to the waived_data.csv file. There were some players that were not matched to a identification number, and so we created a waived_data filed with player id, and an unmatched_waivers csv file where players were not matched to identification numbers. 
	The second notebook is 'player_id_and_name_cleaning.ipynb'. In this notebook, we isolated out player names that were not unique for each season, creating player_id_name.csv, nonqunique_player_names.csv, player_id_season.csv, and nonqunique_player_id_season_pairs.csv files. 

### SalaryData
To collect the salary data, the notebook “ScrapingSalaryData.ipynb” scrapes hoopshype.com using beautiful soup, and gives the csv file ‘player_salaries.csv’. Then, we match those players with the player identification numbers, fixing those who were missing. 
 
### TransactionData

To collect transaction data, the notebook ‘web_scraping_transactions.ipynb’ scrapes basketball-reference.com using beautiful soup, and gives the csv file ‘transaction_data.csv’. We also give csv files that contain if a player was waived (waived_data.csv), traded (traded_data.csv), released (released_data.csv). 

### DataToMerge
This subfolder contains the data files that we intend to merge with the file 1_DataMerger.ipynb. The advanced_stats.csv file was gotten from kaggle, while the others were described above in other subfolders. 


- https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
    - License: CCO: Public Domain (https://creativecommons.org/publicdomain/zero/1.0/)
