"""Small PyMongo integration layer. MongoDB is never exposed to the browser."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

from .train_model import BASE_DIR

load_dotenv(BASE_DIR / ".env")


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
