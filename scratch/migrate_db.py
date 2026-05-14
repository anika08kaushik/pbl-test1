import sqlite3
from pathlib import Path

DB_PATH = Path("data/app.db")

def migrate():
    if not DB_PATH.exists():
        print("Database does not exist. Skipping migration.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Columns to add to 'users' table
    user_columns = [
        ("full_name", "TEXT"),
        ("bio", "TEXT"),
        ("location", "TEXT"),
        ("avatar_url", "TEXT"),
        ("skills", "JSON"),
        ("experience_years", "INTEGER DEFAULT 0")
    ]

    for col_name, col_type in user_columns:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            print(f"Added column {col_name} to users table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists in users table.")
            else:
                print(f"Error adding {col_name} to users: {e}")

    # Columns to add to 'assessments' table
    assessment_columns = [
        ("behavior_summary", "JSON"),
        ("interview_feedback", "JSON")
    ]

    for col_name, col_type in assessment_columns:
        try:
            cursor.execute(f"ALTER TABLE assessments ADD COLUMN {col_name} {col_type}")
            print(f"Added column {col_name} to assessments table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists in assessments table.")
            else:
                print(f"Error adding {col_name} to assessments: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
