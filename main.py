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

# {'kNN': [[0.5272562083585706, 1],
#   [0.5399757722592369, 2],
#   [0.5590551181102362, 3],
#   [0.5632949727437916, 4],
#   [0.5793458509993943, 5],
#   [0.5838885523924894, 6],
#   [0.5990308903694731, 7],
#   [0.5896426408237432, 8],
#   [0.6005451241671714, 9],
#   [0.595396729254997, 10],
#   [0.6053906723198061, 11],
#   [0.5993337371290127, 12],
#   [0.6035735917625682, 13],
#   [0.5999394306480921, 14],
#   [0.6078134463961236, 15]],
#  'Naive Bayes': [[0.8158691701998788, 1]],
#  'Decision Tree': [[0.8543307086614174, 5],
#   [0.8543307086614174, 10],
#   [0.853725015142338, 15],
#   [0.8479709267110842, 20],
#   [0.8479709267110842, 25],
#   [0.8458509993943065, 30],
#   [0.8461538461538461, 35],
#   [0.8452453058752272, 40],
#   [0.8449424591156874, 45],
#   [0.8452453058752272, 50],
#   [0.8452453058752272, 55],
#   [0.8440339188370685, 60],
#   [0.8440339188370685, 65],
#   [0.8437310720775287, 70],
#   [0.8428225317989098, 75],
#   [0.8422168382798304, 80],
#   [0.8410054512416717, 85],
#   [0.838885523924894, 90],
#   [0.838885523924894, 95],
#   [0.8394912174439734, 100]],
#  'Random Forest': [[0.728649303452453, 1],
#   [0.7216838279830405, 2],
#   [0.7846759539672925, 3],
#   [0.7992125984251969, 4],
#   [0.790732889158086, 5],
#   [0.8086008479709267, 6],
#   [0.8128407026044822, 7],
#   [0.8170805572380375, 8],
#   [0.8270745003028468, 9],
#   [0.8373712901271957, 10],
#   [0.837068443367656, 11],
#   [0.8364627498485766, 12],
#   [0.836159903089037, 13],
#   [0.8373712901271957, 14],
#   [0.8410054512416717, 15],
#   [0.8373712901271957, 16],
#   [0.8422168382798304, 17],
#   [0.839794064203513, 18],
#   [0.8410054512416717, 19],
#   [0.8358570563294972, 20],
#   [0.838885523924894, 21],
#   [0.841611144760751, 22],
#   [0.8379769836462749, 23],
#   [0.8422168382798304, 24],
#   [0.8452453058752272, 25],
#   [0.8434282253179891, 26],
#   [0.8382798304058147, 27],
#   [0.84251968503937, 28],
#   [0.8419139915202908, 29],
#   [0.8434282253179891, 30],
#   [0.8440339188370685, 31],
#   [0.8385826771653543, 32],
#   [0.8455481526347668, 33],
#   [0.8437310720775287, 34],
#   [0.8443367655966081, 35],
#   [0.8485766202301636, 36],
#   [0.8394912174439734, 37],
#   [0.838885523924894, 38],
#   [0.8455481526347668, 39],
#   [0.8491823137492429, 40],
#   [0.8467595396729255, 41],
#   [0.84251968503937, 42],
#   [0.8503937007874016, 43],
#   [0.8476680799515445, 44],
#   [0.8482737734706238, 45],
#   [0.8479709267110842, 46],
#   [0.8485766202301636, 47],
#   [0.8482737734706238, 48],
#   [0.8476680799515445, 49],
#   [0.8497880072683223, 50]]}
