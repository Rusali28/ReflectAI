import sqlite3
from datetime import datetime
import pandas as pd

DB_NAME = "journal.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            content TEXT NOT NULL,
            sleep_hours INTEGER,
            stress_level INTEGER,
            emotions TEXT,
            triggers TEXT,
            risk_flag INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_entry(content, sleep, stress, emotions, triggers, risk_flag):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO entries (date, content, sleep_hours, stress_level, emotions, triggers, risk_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (date_str, content, sleep, stress, emotions, triggers, 1 if risk_flag else 0))
    
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY date DESC", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()