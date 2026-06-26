import sqlite3

def init_db():
    connection = sqlite3.connect('c:\Users\Administrator\Desktop\aifriday\friday-ai-lab\chatbot_project\database\database.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS TextEmbeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    connection.commit()
    connection.close()

if __name__ == "__main__":
    init_db()

def init_db():
    connection = sqlite3.connect('c:\\Users\\Administrator\\Desktop\\aifriday\\friday-ai-lab\\chatbot_project\\database\\database.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    connection.commit()
    connection.close()

def insert_data(content):
    connection = sqlite3.connect('c:\\Users\\Administrator\\Desktop\\aifriday\\friday-ai-lab\\chatbot_project\\database\\database.db')
    cursor = connection.cursor()
    cursor.execute('''INSERT INTO data (content) VALUES (?)''', (content,))
    connection.commit()
    connection.close()

def process_and_insert(file_path):
    raw_data = read_data(file_path)  # Ensure this function is defined
    processed_data = preprocess_text(raw_data)  # Ensure this function is defined
    insert_data(processed_data)

if __name__ == '__main__':

    init_db()
    # Example usage
    process_and_insert('data.pdf')  # Replace with your file path