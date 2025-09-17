import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(file_path):
    """Loads and preprocesses the traffic data."""
    df = pd.read_csv(file_path)

    # One-hot encode the 'Weather_Condition'
    encoder = OneHotEncoder(sparse_output=False)
    weather_encoded = encoder.fit_transform(df[['Weather_Condition']])
    weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['Weather_Condition']))
    df = pd.concat([df, weather_df], axis=1)
    df.drop('Weather_Condition', axis=1, inplace=True)
    df.drop('Traffic_Light_State', axis=1, inplace=True)


    # For simplicity, we'll convert all columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any remaining NaN values with the mean of the column
    df.fillna(df.mean(), inplace=True)


    return df