from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def ImputeAndScale(data):
    '''Impute missing data.  Fill stats data with zeros and salary with the
       mean salary for the given season.  Apply custom StandardScaler to
       normalize all numerical columns with respect to each season.'''

    # fill null values with 0 for all columns except SALARY
    null_cols = ['PER', 'TS_PERCENT', 'X3P_AR', 'F_TR', 'ORB_PERCENT',
                 'DRB_PERCENT', 'TRB_PERCENT', 'AST_PERCENT', 'STL_PERCENT',
                 'BLK_PERCENT', 'TOV_PERCENT', 'USG_PERCENT', 'WS_48']
    print(f"Filling missing values for {null_cols} with 0.")

    data[null_cols] = data[null_cols].fillna(0)
    
    mean_imputer = SimpleImputer(strategy='mean')
    
    # replace salaries of 0 with null
    data.loc[data['SALARY']==0, 'SALARY'] = None
    
    # apply mean imputer for SALARY within each season
    data['SALARY'] = (
        data
        .groupby('SEASON_START')['SALARY']
        .transform(lambda x: mean_imputer.fit_transform(x.values.reshape(-1,1)).ravel())
    )
    print("Filling missing SALARY data with season mean salary.")
    
    # rescale stats and salary columns within each season
    cols_to_rescale = data.select_dtypes(include=['float']).columns
    
    scaler = StandardScaler()
    
    # apply standard scaler within each season
    data[cols_to_rescale] = (
        data
        .groupby('SEASON_START')[cols_to_rescale]
        .transform(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).ravel())
    )
    print("Apply StandardScaler to scale data within each season.")

    return data
