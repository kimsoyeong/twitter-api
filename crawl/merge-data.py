import pandas as pd
import glob
import os
import csv

def merge():
    # path to the files
    files = os.path.join(".", "tweets-new-*.csv")

    # list of merged files
    files = glob.glob(files)

    # joining files
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv("data.csv", index=False)

def remove():
    # read file
    df = pd.read_csv("data.csv")
    print(df)
    # drop tweets without emoji
    df = df[df['emojis'].ne('[]')]
    print(df)
    # write the data to a new dataset
    df.to_csv("data-rm.csv", index=False)

remove()