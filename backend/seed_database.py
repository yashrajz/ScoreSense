"""Import the validated CSV into MongoDB without duplicating dataset records.

Run from the project root with:

    python -m backend.seed_database
"""

from datetime import datetime, timezone

from .database import seed_dataset
from .train_model import load_and_validate_dataset


def main() -> None:
    data = load_and_validate_dataset()
    records = []
    seeded_at = datetime.now(timezone.utc)
    for row in data.to_dict(orient="records"):
        records.append(
            {
                **row,
                "record_type": "dataset_record",
                "source_dataset": "lpu_student_performance.csv",
                "is_estimate": False,
                "created_at": seeded_at,
                "seeded_at": seeded_at,
            }
        )
    count = seed_dataset(records)
    print(f"Seeded or refreshed {count} dataset records in lpu_student_performance_db.predictions.")


if __name__ == "__main__":
    main()
