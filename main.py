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
df.info()



neededCols = ['Location', 'Tournament', 'Series', 'Court', 'Surface',
       'Round', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts',
       'Comment', 'B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL']
df = df[neededCols]


rankDefault = max(df['WRank'].max(), df['LRank'].max())+1
df.fillna({'WRank': rankDefault, 'LRank': rankDefault, 'WPts': 0, 'LPts': 0}, inplace=True)




nullOddsDf = df[df[['B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL']].isna().any(axis=1)]

for index, row in nullOddsDf.iterrows():
     if pd.isnull(row['AvgW']) or pd.isnull(row['AvgL']):
         AvgW, AvgL = findOddsForRow(row, df.dropna(subset=['AvgW', 'AvgL']))
         df.at[index, 'AvgW'] = row['AvgW'] = AvgW
         df.at[index, 'AvgL'] = row['AvgL'] = AvgL
     if pd.isnull(row['B365W']):
         df.at[index, 'B365W'] = row['AvgW']
     if pd.isnull(row['B365L']):
         df.at[index, 'B365L'] = row['AvgL']
     if pd.isnull(row['PSW']):
         df.at[index, 'PSW'] = row['AvgW']
     if pd.isnull(row['PSL']):
         df.at[index, 'PSL'] = row['AvgL']

df.dropna(subset=['AvgW', 'AvgL'], inplace=True) # Drop rows that hasn't similar rank matches
df.info()


# First analysis
# odd (inverse -> percentuale di vincita secondo il banco) --> Calcolo percentuale di successo

# Models
# Carefully-tuned random forest classifier
# Naive Bayes
