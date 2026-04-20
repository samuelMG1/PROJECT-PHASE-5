import pandas as pd
import numpy as np


def load_data(file_path):
    """Loads data from a specified CSV file.""" 
    return pd.read_csv(file_path)


def validate_data(df):
    """Validates the DataFrame for missing values and correct data types."""
    if df.isnull().sum().any():
        raise ValueError('Data contains missing values')
    return True


def clean_data(df):
    """Cleans the DataFrame by removing duplicates."""
    return df.drop_duplicates()


def detect_outliers(df, threshold=3):
    """Detects outliers using Z-score method."""
    z_scores = np.abs((df - df.mean()) / df.std())
    return df[(z_scores < threshold).all(axis=1)]


def split_data(df, target_column, test_size=0.2):
    """Splits the data into training and testing sets.""" 
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def scale_features(X):
    """Scales features using Min-Max scaling.""" 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)