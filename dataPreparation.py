import pandas as pd
import numpy as np
from utilities import *

def findOddsForRow (row,  df):
    """
    Call in a loop to create terminal progress bar
    @params:
        row   - Selected row (Series)
        df    - Original dataframe (Dataframe)
    """
    # Search a row with similar ranks
    foundRows = pd.DataFrame()
    nearRank = 10
    while foundRows.empty and nearRank <= 100:
        foundRows = df[((row.Rank0 - nearRank) < df.Rank0) & (df.Rank0 < (row.Rank0 + nearRank)) & ((row.Rank1 - nearRank) < df.Rank1) & (df.Rank1 < (row.Rank1 + nearRank))]
        nearRank += 10

    return ( foundRows["Avg0"].mean(), foundRows["Avg1"].mean() ) if not foundRows.empty else ( None, None )

def expectedScore(A, B):
    """
    Calculate expected score of A in a match against B
    @params:
        A   - Elo rating for player A
        B   - Elo rating for player B
    """
    return 1 / (1 + 10 ** ((B - A) / 400))

def eloRating(old_elo, expected_score, actual_score, k_factor = 32):
    """
    Calculate the new Elo rating for a player
    @params:
        old_elo         - The previous Elo rating
        expected_score  - The expected score for this match
        actual_score    - The actual score for this match
        k_factor        - The k-factor for Elo (default: 32)
    """
    return old_elo + k_factor * (actual_score - expected_score)

def addEloRatingFeature(X, defaultElo = 1500):
    """
    Add the Elo Rating for each match of the dataset
    @params:
        X            - The dataset
        kFactor      - The k-factor for Elo (default: 32)
        defaultElo   - The initial value for each player

    K-factor for players below 2100, between 2100–2400 and above 2400 of 32, 24 and 16, respectively
    """
    players = pd.concat([X.Player0, X.Player1]).unique()
    oldEloRatings = pd.Series(np.ones(players.size) * defaultElo, index=players)
    kFactor = 32 if players.size < 2100 else (24 if 2100 <= players.size <= 2400 else 16)

    PLAYER_0_SCORE = 1 # We assume that player0 is always the winner, the score is 1 for winner, 0 for the loser
    PLAYER_1_SCORE = 0

    # New feature columns
    player0EloRating = pd.Series(np.ones(X.shape[0]) * defaultElo)
    player1EloRating = pd.Series(np.ones(X.shape[0]) * defaultElo)

    printProgressBar(0, X.shape[0], prefix='Progress:', suffix='Complete')
    for i, row in X.iterrows():
        oldEloRatingPlayer0 = oldEloRatings[row.Player0]
        oldEloRatingPlayer1 = oldEloRatings[row.Player1]

        expectedScorePlayer0 = expectedScore(oldEloRatingPlayer0, oldEloRatingPlayer1)
        expectedScorePlayer1 = expectedScore(oldEloRatingPlayer1, oldEloRatingPlayer0)

        # Save the new rate in players for the next match and in the new feature column
        oldEloRatings[row.Player0] = player0EloRating[i] = eloRating(oldEloRatingPlayer0, expectedScorePlayer0, PLAYER_0_SCORE, k_factor=kFactor)
        oldEloRatings[row.Player1] = player1EloRating[i] = eloRating(oldEloRatingPlayer1, expectedScorePlayer1, PLAYER_1_SCORE, k_factor=kFactor)

        printProgressBar(i, X.shape[0], prefix='Progress:', suffix='Complete')

    X = X.assign(EloRating0 = player0EloRating.values)
    X = X.assign(EloRating1 = player1EloRating.values)

    return X
