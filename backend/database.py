"""Small PyMongo integration layer. MongoDB is never exposed to the browser."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo import UpdateOne

from .train_model import BASE_DIR

load_dotenv(BASE_DIR / ".env")

DATASET_INDEX_NAME = "dataset_seed_student_unique"
DATASET_PARTIAL_FILTER = {
    "record_type": "dataset_record",
    "source_dataset": "lpu_student_performance.csv",
}


def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is not configured.")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client


def save_prediction(record: dict) -> None:
    client = get_client()
    try:
        client["lpu_student_performance_db"]["predictions"].insert_one(
            {**record, "created_at": datetime.now(timezone.utc)}
        )
    finally:
        client.close()


def recent_predictions(limit: int) -> list[dict]:
    client = get_client()
    try:
        documents = client["lpu_student_performance_db"]["predictions"].find(
            {}, {"_id": 0}
        ).sort("created_at", -1).limit(limit)
        return list(documents)
    finally:
        client.close()


def _ensure_dataset_index(collection) -> None:
    existing_index = collection.index_information().get(DATASET_INDEX_NAME)
    if existing_index and existing_index.get("partialFilterExpression") != DATASET_PARTIAL_FILTER:
        collection.drop_index(DATASET_INDEX_NAME)
    collection.create_index(
        [("record_type", 1), ("source_dataset", 1), ("student_id", 1)],
        unique=True,
        name=DATASET_INDEX_NAME,
        partialFilterExpression=DATASET_PARTIAL_FILTER,
    )


def seed_dataset(records: list[dict]) -> int:
    """Upsert validated CSV records without duplicating an existing seed."""
    client = get_client()
    try:
        collection = client["lpu_student_performance_db"]["predictions"]
        _ensure_dataset_index(collection)
        operations = [
            UpdateOne(
                {
                    "record_type": "dataset_record",
                    "source_dataset": "lpu_student_performance.csv",
                    "student_id": record["student_id"],
                },
                {"$set": record},
                upsert=True,
            )
            for record in records
        ]
        if not operations:
            return 0
        result = collection.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count
    finally:
        client.close()
