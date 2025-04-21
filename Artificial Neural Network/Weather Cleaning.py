import pandas as pd
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
df = pd.read_csv("weather_classification_data.csv")

df['cloud_cover_code'] = df["Cloud Cover"].astype('category').cat.codes
df.drop("Cloud Cover", axis = 1, inplace=True)

df['season_code'] = df["Season"].astype('category').cat.codes
df.drop("Season", axis = 1, inplace=True)

df['location_code'] = df["Location"].astype('category').cat.codes
df.drop("Location", axis = 1, inplace=True)

df.dropna(inplace=True)

# 0 : Cloudy
# 1 : Rainy
# 2 : Snowy
# 3 : Sunny
df['weather_code'] = df["Weather Type"].astype('category').cat.codes
df.drop("Weather Type", axis = 1, inplace=True)

X = df.iloc[:, :-1].copy().to_numpy()
y = df.iloc[:, -1].copy().to_numpy()

train, validate = train_test_split(df, test_size=0.3)

train.to_csv("train_set.csv")
validate.to_csv("validate_set.csv")