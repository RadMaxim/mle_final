from functools import wraps
import pandas as pd
def apply_steps(*steps):
    def decorator(func):
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            data = func(data)

            for step in steps:
                data = step(data)

            return data

        return wrapper

    return decorator
def fill_missing_values(data):

    cols_with_nans = data.columns[data.isnull().any()]

    for col in cols_with_nans:

        if data[col].dtype in [float, int]:
            fill_value = data[col].mean()
        elif data[col].dtype == 'object':
            fill_value = data[col].mode().iloc[0]

        data[col] = data[col].fillna(fill_value)
    
    return data 
def remove_v(data):
    cols_to_check = [
        "target",
        "kitchen_area",
        "living_area",
        "total_area",
        "price",
        "ceiling_height",
        "rooms",
        "floor",
        "flats_count",
        "floors_total",
        "build_year",
    ]

    threshold = 1.5
    potential_outliers = pd.DataFrame()

    for col in cols_to_check:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        margin = threshold*IQR
        lower = Q1 - margin
        upper = Q3 + margin
        potential_outliers[col] = ~data[col].between(lower, upper)

    outliers = potential_outliers.any(axis=1)
    cleaned_data = data[~outliers]
    return cleaned_data 
    
def remove_duplicates(data):
    feature_cols = data.columns.tolist()
    is_duplicated_features = data.duplicated(subset=feature_cols, keep=False)
    data = data[~is_duplicated_features].reset_index(drop=True)
    return data 