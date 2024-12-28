import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from HelperFunctions import *

#--------------------Helper functions------------------------------------------#

@st.cache_data
def load_data():
    data = pd.read_csv("full_data.csv")
    return data

def visualize_data(year, x, y):
    df["Retained"] = "No"
    df.loc[df.IN_LEAGUE_NEXT == 1, "Retained"] = "Yes"
    
    fig = px.scatter(df.loc[df.SEASON_START==year], x=x, y=y, color="Retained",
                    hover_name="NAME", hover_data=["TEAMS_LIST"],
                    color_discrete_map={"Yes":"dodgerblue", "No":"lightcoral"})
    st.plotly_chart(fig)

    return None

def compute_model_predictions():
    df_scaled = ImputeAndScale(df.copy())

    df_scaled["PRED"] = 0
    df_scaled["PROB"] = 0

    #===define training set, do train-test split===#
    for season in range(2017, 2024):
        df_train = df_scaled.loc[df_scaled.SEASON_START < season].copy()
        df_tt, df_cal = train_test_split(df_train, test_size=0.2, shuffle=True,
                                         random_state=815, 
                                         stratify=df_train.IN_LEAGUE_NEXT)

        #===train/fit model and calibrate===#
        model = Pipeline([('smote', SMOTE(random_state=23)),
                          ('xgb', XGBClassifier(n_estimators=350, 
                                                learning_rate=0.005,
                                                random_state=206))])

        model.fit(df_tt[features], df_tt.IN_LEAGUE_NEXT)

        model_cal = CalibratedClassifierCV(model, cv="prefit")
        model_cal.fit(df_cal[features], df_cal.IN_LEAGUE_NEXT)

        #===save predictions in df_scaled===#
        df_scaled.PRED = model.predict(df_scaled.loc[df_scaled.SEASON_START == season][features])

    return df_preds 

#---------------------Load the data--------------------------------------------#

df = load_data()


#-------------------------Generate output--------------------------------------#

st.title('Predicting NBA Transactions')

st.subheader("Visualizing the data")

features = df.select_dtypes(include='number').columns.drop(['PLAYER_ID', 
                                            'SEASON_START', 'IN_LEAGUE_NEXT'])

stats = features.drop(['WAIVED', 'RELEASED', 'TRADED',        
       'WAIVED_OFF', 'WAIVED_REG', 'WAIVED_POST', 'RELEASED_OFF',               
       'RELEASED_REG', 'RELEASED_POST', 'TRADED_OFF', 'TRADED_REG',             
       'TRADED_POST'])

x_stat = st.selectbox("x-axis:", stats)
y_stat = st.selectbox("y-axis:", stats)
year   = st.slider(label="Season start year:", min_value=1990, max_value=2022)

visualize_data(year, x_stat, y_stat)

#----------------------------Model predictions---------------------------------#

st.subheader("Model predictions")

##grab scaled copy of df
#df_scaled = ImputeAndScale(df.copy())
##convert teams list from strings to actual lists
#df_scaled["TEAMS_AS_LIST"] = df_scaled.apply(lambda x: eval(x.TEAMS_LIST),
#                                             axis=1)
##explode out teams
#df_scaled_exploded = df_scaled.explode("TEAMS_AS_LIST", ignore_index=True)
#
##grab team list
#teams = df_scaled_exploded.TEAMS_AS_LIST.unique()
#teams.sort()
#
#selected_team = st.selectbox("Team:", teams)
#year_bar      = st.slider(label="Season start year:", min_value=2017, 
#                          max_value=2023)  

model_preds = compute_model_predictions()

st.dataframe(model_preds)
