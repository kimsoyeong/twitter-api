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

def remove_column():
    # read file
    df = pd.read_csv("data.csv")
    # remove a column
    df.drop('hate_words', inplace=True, axis=1)
    # write the data to a new dataset
    df.to_csv("data-no-hateword.csv", index=False)

def no_duplicate():
    # read file
    df = pd.read_csv("data-no-hateword.csv")
    # remove duplicated lines
    df = df.drop_duplicates()
    # write the data to a new dataset
    df.to_csv("data-no-dupl.csv", index=False)