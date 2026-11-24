"""Create a 300-record synthetic SCADA dataset."""

import numpy as np
import pandas as pd

from generator.scada_generator import load_source_data
from wtpm.predict import predict_turbine_status
from schemas.message_schema import load_feature_names

OUTPUT = "data/processed/synthetic_scada_300.csv"
RANDOM_SEED = 42

def main():
    np.random.seed(RANDOM_SEED)

    print("Loading historical SCADA records...")
    df = load_source_data(1000)

    features = load_feature_names()
    df = df[["asset_id", "time_stamp"] + features].copy()

    records = df.to_dict(orient="records")

    print("Running model on historical records...")
    predictions = predict_turbine_status(records, feature_set="avg")

    df["model_prediction"] = [p["prediction"] for p in predictions]
    df["anomaly_probability"] = [
        p["anomaly_probability"] for p in predictions
    ]

    normal = df[df["model_prediction"] == "normal"].sample(
        n=200, random_state=RANDOM_SEED
    )

    anomaly_seeds = df[df["model_prediction"] == "anomaly"]

    if len(anomaly_seeds) == 0:
        raise ValueError("No historical anomaly seeds found.")

    anomaly_features = [
        "sensor_225_avg",
        "sensor_74_avg",
        "sensor_227_avg",
        "sensor_44_avg",
        "sensor_186_avg",
        "sensor_210_avg",
        "sensor_212_avg",
        "sensor_43_avg",
    ]

    std_values = df[anomaly_features].std().fillna(0)

    synthetic_anomalies = []

    for i in range(100):
        seed = anomaly_seeds.iloc[i % len(anomaly_seeds)]
        record = seed[["asset_id", "time_stamp"] + features].to_dict()

        for feature in anomaly_features:
            value = record[feature]
            if pd.notna(value):
                noise = np.random.normal(0, 0.02 * std_values[feature])
                record[feature] = float(value + noise)

        record["synthetic_label"] = "anomaly"
        synthetic_anomalies.append(record)

    normal_records = normal[["asset_id", "time_stamp"] + features].copy()
    normal_records["synthetic_label"] = "normal"

    anomaly_df = pd.DataFrame(synthetic_anomalies)

    result = pd.concat(
        [normal_records, anomaly_df],
        ignore_index=True
    )

    result = result.sample(frac=1, random_state=RANDOM_SEED)
    result.to_csv(OUTPUT, index=False)

    print("\nSynthetic dataset created!")
    print("Total records:", len(result))
    print(result["synthetic_label"].value_counts())
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
