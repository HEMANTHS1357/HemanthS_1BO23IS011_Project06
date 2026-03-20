import pandas as pd
import numpy as np

def load_and_preprocess():
    df = pd.read_csv('data/heart.csv')
    
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Fill missing values with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Target column
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    print("\nCleaning done!")
    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())