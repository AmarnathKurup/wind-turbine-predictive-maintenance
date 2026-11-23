"""Publish synthetic SCADA records from CSV to Kafka."""

import json
import time
import pandas as pd

from kafka import KafkaProducer
from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from schemas.message_schema import load_feature_names, validate_message

CSV_PATH = "data/processed/synthetic_scada_300.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    features = load_feature_names()

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: str(key).encode("utf-8"),
        acks="all",
        retries=5,
    )

    try:
        for index, row in df.iterrows():
            message = {
                "asset_id": int(row["asset_id"]),
                "time_stamp": str(row["time_stamp"]),
            }

            for feature in features:
                value = row[feature]
                message[feature] = (
                    None if pd.isna(value) else float(value)
                )

            validate_message(message)

            future = producer.send(
                KAFKA_TOPIC,
                key=message["asset_id"],
                value=message,
            )

            metadata = future.get(timeout=10)

            print(
                f"Sent {index + 1}/{len(df)} | "
                f"Offset: {metadata.offset} | "
                f"Asset: {message['asset_id']}"
            )

            time.sleep(0.05)

    finally:
        producer.flush()
        producer.close()

    print(f"\nSuccessfully published {len(df)} synthetic records.")


if __name__ == "__main__":
    main()
