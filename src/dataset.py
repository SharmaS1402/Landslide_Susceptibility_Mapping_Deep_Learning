import pandas as pd


def load_data(file_path="data/raw/final_data.csv"):
    data = pd.read_csv(file_path)
    return data


# print(df.head(2))
