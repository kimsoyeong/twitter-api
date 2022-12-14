import pandas as pd
import glob
import os

# path to the files
files = os.path.join(".", "tweets-new-*.csv")

# list of merged files
files = glob.glob(files)

# joining files
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.to_csv("data.csv", index=False)
print(df)
