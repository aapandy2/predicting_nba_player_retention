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

def load_model_preds():
    preds = pd.read_csv("model_predictions.csv")
    return preds

def visualize_data(year, x, y):
    df["Retained"] = "No"
    df.loc[df.IN_LEAGUE_NEXT == 1, "Retained"] = "Yes"
    
    fig = px.scatter(df.loc[df.SEASON_START==year], x=x, y=y, color="Retained",
                    hover_name="NAME", hover_data=["TEAMS_LIST"],
                    color_discrete_map={"Yes":"dodgerblue", "No":"lightcoral"})
    st.plotly_chart(fig)

    return None

def visualize_preds(season, team):
    df_temp = preds_exp.loc[(preds_exp.TEAMS_AS_LIST == team)
            & (preds_exp.SEASON_START == season)].copy()

    df_temp["Retained"] = "No"
    df_temp.loc[df_temp.IN_LEAGUE_NEXT == 1, "Retained"] = "Yes"

    df_temp["Predict retention?"] = "No"
    df_temp.loc[df_temp.PRED == 1, "Predict retention?"] = "Yes"

    if(season != 2023):
        fig = px.bar(df_temp, x="NAME", y="PROB", color="Retained",
                    labels={"NAME":"Player name",
                            "PROB":"Model retention probability",
                            "Retained":"Retained?"},
                    color_discrete_map={"Yes":'cornflowerblue',
                                        "No":'lightcoral'})
    else:
        fig = px.bar(df_temp, x="NAME", y="PROB", color="Predict retention?",
                    labels={"NAME":"Player name",
                            "PROB":"Model retention probability"},
                    color_discrete_map={"Yes":'cornflowerblue', "No":'lightcoral'})

    st.plotly_chart(fig)

    return None

#---------------------Load the data--------------------------------------------#

#load full set of raw training/test data
df = load_data()

#load model prediction data
preds = load_model_preds()

#convert teams list from strings to actual lists
preds["TEAMS_AS_LIST"] = preds.apply(lambda x: eval(x.TEAMS_LIST), axis=1)
#explode out teams
preds_exp = preds.explode("TEAMS_AS_LIST", ignore_index=True)

#grab sorted list of teams
teams = preds_exp.TEAMS_AS_LIST.unique()
teams.sort()

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

pred_year = st.slider(label="Season start year:", min_value=2017,
                      max_value=2023) 
pred_team = st.selectbox("Team:", teams)

visualize_preds(pred_year, pred_team)
