import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("IRIS_DB_PATH", "iris.db")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS input_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sepal_length REAL NOT NULL,
                sepal_width REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_id INTEGER NOT NULL UNIQUE,
                prediction INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (input_id) REFERENCES input_data(id)
            );
            """
        )


def insert_input(features: list[float]) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO input_data (sepal_length, sepal_width, petal_length, petal_width) VALUES (?, ?, ?, ?)",
            features,
        )
        return cur.lastrowid


def fetch_unpredicted() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT i.id, i.sepal_length, i.sepal_width, i.petal_length, i.petal_width
            FROM input_data i
            LEFT JOIN predictions p ON p.input_id = i.id
            WHERE p.id IS NULL
            ORDER BY i.id
            """
        ).fetchall()


def insert_prediction(input_id: int, prediction: int, label: str):
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions (input_id, prediction, prediction_label) VALUES (?, ?, ?)",
            (input_id, prediction, label),
        )
