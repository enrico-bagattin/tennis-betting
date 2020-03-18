##################################################
#   TO BE REMOVED: For easier debugging purpose  #
##################################################


# Import
import pandas as pd
import numpy as np
import glob
import seaborn as sns

# Modules
from utilities import *
from dataPreparation import *


# Import data
paths = list(glob.glob("matches/20*.xlsx"))
matches = [pd.read_excel(path) for path in paths]
# TODO: Load matches based on number of past years choosen
df = pd.concat(matches, ignore_index=True, sort=False)

neededCols = ['Location', 'Tournament', 'Series', 'Court', 'Surface',
       'Round', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts',
       'Comment', 'B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL']
df = df[neededCols]

df.columns = ['Location', 'Tournament', 'Series', 'Court', 'Surface',
       'Round', 'Player0', 'Player1', 'Rank0', 'Rank1', 'Pts0', 'Pts1',
       'Comment', 'B3650', 'B3651', 'PS0', 'PS1', 'Avg0', 'Avg1']

rankDefault = max(df['Rank0'].max(), df['Rank1'].max())+1
df.fillna({'Rank0': rankDefault, 'Rank1': rankDefault, 'Pts0': 0, 'Pts1': 0}, inplace=True)

nullOddsDf = df[df[['B3650', 'B3651', 'PS0', 'PS1', 'Avg0', 'Avg1']].isna().any(axis=1)]
for index, row in nullOddsDf.iterrows():
     if pd.isnull(row['Avg0']) or pd.isnull(row['Avg1']):
         Avg0, Avg1 = findOddsForRow(row, df.dropna(subset=['Avg0', 'Avg1']))
         df.at[index, 'Avg0'] = row['Avg0'] = Avg0
         df.at[index, 'Avg1'] = row['Avg1'] = Avg1
     if pd.isnull(row['B3650']):
         df.at[index, 'B3650'] = row['Avg0']
     if pd.isnull(row['B3651']):
         df.at[index, 'B3651'] = row['Avg1']
     if pd.isnull(row['PS0']):
         df.at[index, 'PS0'] = row['Avg0']
     if pd.isnull(row['PS1']):
         df.at[index, 'PS1'] = row['Avg1']

df.dropna(subset=['Avg0', 'Avg1'], inplace=True) # Drop rows that hasn't similar rank matches

# X = pd.get_dummies(df)

X = addEloRatingFeature(df)
