"""
Admin Seeding Script for Dental AI Chatbot Knowledge Hub.
Creates an SQLite admin database and initializes default credentials.
"""

import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).resolve().parent / "admin.db"

DEFAULT_ADMIN_EMAIL = "admin@dentalcare.com"
DEFAULT_ADMIN_PASS = "DentalAdmin2026!"


def init_db():
    """Create admins table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def seed_admin(email: str = DEFAULT_ADMIN_EMAIL, password: str = DEFAULT_ADMIN_PASS):
    """Seed or update default administrator credentials."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    pwd_hash = generate_password_hash(password)

    cursor.execute("SELECT id FROM admins WHERE email = ?", (email,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("UPDATE admins SET password_hash = ? WHERE email = ?", (pwd_hash, email))
        conn.commit()
        print(f"\n[OK] Admin '{email}' already existed. Password successfully updated!")
    else:
        cursor.execute("INSERT INTO admins (email, password_hash) VALUES (?, ?)", (email, pwd_hash))
        conn.commit()
        print(f"\n[OK] Default Administrator successfully created!")

    conn.close()

    print("=" * 60)
    print("  [ADMIN] DENTAL ADMIN LOGIN CREDENTIALS:")
    print(f"     Email / Username : {email}")
    print(f"     Password         : {password}")
    print("     Login URL        : http://localhost:5000/login")
    print("=" * 60 + "\n")


def verify_admin_login(email: str, password: str) -> bool:
    """Verify administrator email and password against hash."""
    if not DB_PATH.exists():
        init_db()
        seed_admin()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM admins WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    return check_password_hash(row[0], password)


if __name__ == "__main__":
    seed_admin()
