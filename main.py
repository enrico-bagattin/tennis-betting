##################################################
#   TO BE REMOVED: For easier debugging purpose  #
##################################################

# Import
import pandas as pd
import numpy as np
import glob

# Modules
from utilities import *
from dataPreparation import *

# Import data
years = [2017, 2018, 2019, 2020]
yearsForFeatures = [2016, 2017, 2018, 2019, 2020]
paths = []
for y in years:
    paths.append('matches/' + str(y) + '.xlsx')
availablePaths = list(glob.glob("matches/20*.xlsx"))
matches = [pd.read_excel(path) for path in paths]
yearZeroForFeatures = pd.read_excel('matches/' + str(years[0] - 1) + '.xlsx')
# TODO: Load matches based on number of past years choosen
df = pd.concat(matches, ignore_index=True, sort=False)
df.info()


df = removeWinnerLoserReference(df)
yearZeroForFeatures = removeWinnerLoserReference(yearZeroForFeatures)

rankDefault = max(df['Rank0'].max(), df['Rank1'].max()) + 1
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

df.dropna(subset=['Avg0', 'Avg1'], inplace=True)  # Drop rows that hasn't similar rank matches
df.info()


# TODO: Handle Round ????????

# X['Round'].value_counts()

# you might change this according to a notion of weight
# X['Round'] = X['Round'].map ({  '1st Round'    : 1,
#                                 '2nd Round'    : 2,
#                                 '3rd Round'    : 4,
#                                 '4th Round'    : 8,
#                                 'Quarterfinals': 16,
#                                 'Round Robin'  : 32,
#                                 'Semifinals'   : 32,
#                                 'The Final'    : 64})

X = addEloRatingFeature(df)

X = addMatchesPlayedAndWonFeatures(X, yearZeroForFeatures, yearsForFeatures)

X = addInjuriesAndWinningStreakFeatures(X, yearZeroForFeatures, yearsForFeatures)

X.to_csv('generated/beforeDuplication.csv', index=False)

duplication = X.copy()
duplication.columns = ['Date', 'Location', 'Tournament', 'Series', 'Court', 'Surface', 'Round',
                       'Player1', 'Player0', 'Rank1', 'Rank0', 'Pts1', 'Pts0', 'Comment',
                       'B3651', 'B3650', 'PS1', 'PS0', 'Avg1', 'Avg0', 'EloRating1',
                       'EloRating0', 'MatchesPlayed1', 'MatchesPlayed0', 'MatchesWon1',
                       'MatchesWon0', 'Injuries1', 'Injuries0', 'WinningStreak1',
                       'WinningStreak0']

# Add the winner column
X = X.assign(Winner=np.zeros(X.shape[0]))  # Player 0 always win
duplication = duplication.assign(Winner=np.ones(X.shape[0]))  # Player 1 always win

X = pd.concat([X, duplication], ignore_index=True)
X.reset_index(inplace=True)
X.sort_values(by='index', inplace=True)
X.drop(columns=['Date', 'Comment', 'index'], inplace=True)


X = pd.get_dummies(X)
print('Total number of columns:', len(X.columns))

X.to_csv('generated/finalDataset.csv', index=False)



from sklearn.model_selection import train_test_split

X = pd.read_csv('generated/finalDataset.csv')

X.head()

y = X.Winner.values
X.drop(columns='Winner', inplace=True)

test_size = len(X) // 5
X_test = X[-test_size:]
y_test = y[-test_size:]
X_train_80 = X[:-test_size]
y_train_80 = y[:-test_size]

# Random split for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X_train_80, y_train_80, test_size=0.25, random_state=0)

