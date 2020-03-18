import pandas as pd

def findOddsForRow (row,  df):
    """
    Call in a loop to create terminal progress bar
    @params:
        row   - Required  : selected row (Series)
        df    - Required  : original dataframe (Dataframe)
    """
    # Search a row with similar ranks
    foundRows = pd.DataFrame()
    nearRank = 10
    while foundRows.empty and nearRank <= 100:
        foundRows = df[((row.WRank - nearRank) < df.WRank) & (df.WRank < (row.WRank + nearRank)) & ((row.LRank - nearRank) < df.LRank) & (df.LRank < (row.LRank + nearRank))]
        nearRank += 10

    return ( foundRows["AvgW"].mean(), foundRows["AvgL"].mean() ) if not foundRows.empty else ( None, None )
