import sqlite3

def init_db():
    connection = sqlite3.connect('chatbot.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    connection.commit()
    connection.close()

if __name__ == "__main__":
    init_db()