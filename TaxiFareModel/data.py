import pandas as pd
from sklearn.model_selection import train_test_split

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
BUCKET_NAME ='wagon-data-589-jauffret'
BUCKET_TRAIN_DATA_PATH='data/train_1k.csv'

def get_data(nrows=1000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    return df

def getXy(df,col_target):
    y = df.pop(col_target)
    X = df
    return X,y

def getholdout(X,y,test_size=0.15):
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size = test_size)
    return X_train, X_val, y_train, y_val

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data(nrows=1000)
    df = clean_data(df)
    X,y = getXy(df,"fare_amount")
    X_train, X_val, y_train, y_val = getholdout(X,y,test_size=0.15)
    print(X_train)
