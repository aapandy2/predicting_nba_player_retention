# Description of Subfolders

In this folder, there are subfolders that contain how .csv files were gathered (CountingStats, PlayerIdentification, SalaryData, TransactionData) and a folder (DataToMerge) that contains the process for merging the different .csv files into one file. Below, we provide a brief description of each subfolder. 

### CountingStats
To collect the NBA counting stats, we make use of the nba_api library to pull the counting stats for all players listed on the official NBA website. 

### PlayerIdentification
There are two notebooks in this file. One 'Adding_ID_Numbers.ipynb' which adds player identification numbers to the waived_data.csv file. There were some players that were not matched to a identification number, and so we created a waived_data filed with player id, and an unmatched_waivers csv file where players were not matched to identification numbers. 
	The second notebook is 'player_id_and_name_cleaning.ipynb'. In this notebook, we isolated out player names that were not unique for each season, creating player_id_name.csv, nonqunique_player_names.csv, player_id_season.csv, and nonqunique_player_id_season_pairs.csv files. 

### SalaryData

### TransactionData

### DataToMerge



- https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
    - License: CCO: Public Domain (https://creativecommons.org/publicdomain/zero/1.0/)
