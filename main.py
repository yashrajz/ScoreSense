"""Compatibility entry point for explicit model training.

Run ``python main.py`` after installing the backend dependencies to train the
models from the existing CSV and generate real evaluation artifacts.
"""

from backend.train_model import main


if __name__ == "__main__":
    main()
