import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attrs, cat_attrs):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("nums", num_pipeline, num_attrs),
        ("cats", cat_pipeline, cat_attrs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("housing.csv")
    df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    for sett in (df, train_set, test_set):
        sett.drop("income_cat", axis=1, inplace=True)

    test_set.to_csv("Testing_data.csv", index=False)

    housing = train_set.copy()
    housing_labels = train_set["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attrs = housing_features.drop("ocean_proximity", axis=1).columns.tolist() # we need to give column names
    cat_attrs = ["ocean_proximity"]


    pipeline = build_pipeline(num_attrs, cat_attrs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved")

else:
    # Inference phase
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    df = pd.read_csv("Testing_data.csv")
    transformed_data = pipeline.transform(df)
    predictions = model.predict(transformed_data)

    df["median_house_value"] = predictions
    df.to_csv("Predictions.csv", index=False)
    print("Inference saved to Predictions.csv")


