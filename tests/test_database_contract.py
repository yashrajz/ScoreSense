import unittest

from backend.database import DATASET_INDEX_NAME, DATASET_PARTIAL_FILTER, _ensure_dataset_index


class FakeCollection:
    def __init__(self, indexes=None):
        self.indexes = indexes or {}
        self.dropped = []
        self.created = []

    def index_information(self):
        return self.indexes

    def drop_index(self, name):
        self.dropped.append(name)
        del self.indexes[name]

    def create_index(self, keys, **options):
        self.created.append((keys, options))


class DatabaseContractTests(unittest.TestCase):
    def test_dataset_index_only_applies_to_seeded_records(self):
        collection = FakeCollection()

        _ensure_dataset_index(collection)

        self.assertEqual(collection.dropped, [])
        self.assertEqual(collection.created[0][1]["name"], DATASET_INDEX_NAME)
        self.assertTrue(collection.created[0][1]["unique"])
        self.assertEqual(
            collection.created[0][1]["partialFilterExpression"],
            DATASET_PARTIAL_FILTER,
        )

    def test_old_index_is_replaced_for_existing_seeded_database(self):
        collection = FakeCollection(
            {
                DATASET_INDEX_NAME: {
                    "key": [
                        ("record_type", 1),
                        ("source_dataset", 1),
                        ("student_id", 1),
                    ],
                    "unique": True,
                }
            }
        )

        _ensure_dataset_index(collection)

        self.assertEqual(collection.dropped, [DATASET_INDEX_NAME])
        self.assertEqual(
            collection.created[0][1]["partialFilterExpression"],
            DATASET_PARTIAL_FILTER,
        )


if __name__ == "__main__":
    unittest.main()
