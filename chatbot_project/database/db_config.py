import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).with_name("database.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS TextEmbeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    connection.commit()
    connection.close()


def insert_data(content):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO data (content) VALUES (?)", (content,))
    connection.commit()
    connection.close()


def process_and_insert(file_path):
    from backend import preprocess_text, read_data

    raw_data = read_data(file_path)
    processed_data = preprocess_text(raw_data)
    insert_data(processed_data)


if __name__ == "__main__":
    init_db()
