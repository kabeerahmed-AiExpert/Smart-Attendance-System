"""
================================================================================
  Database Module
================================================================================
  SQLite attendance database for the Smart Attendance System.
  Handles table creation, attendance insertion with duplicate prevention,
  and query functions for the Streamlit dashboard.
================================================================================
"""

import os
import sqlite3
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


def get_connection():
    """Create and return a SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Initialize the attendance database.
    Creates the attendance table if it doesn't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL,
            confidence  REAL NOT NULL,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def is_already_marked(name, date):
    """
    Check if a person is already marked present on a given date.
    Prevents duplicate attendance entries per session/day.

    Args:
        name : Student full name (folder name).
        date : Date string (YYYY-MM-DD).

    Returns:
        bool : True if already marked, False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE name = ? AND date = ? AND status = 'Present'",
        (name, date)
    )
    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def insert_attendance(name, status, confidence):
    """
    Insert an attendance record into the database.

    Args:
        name       : Student name or "Unknown".
        status     : "Present" or "Unknown".
        confidence : Recognition confidence score (float).

    Returns:
        inserted : bool — True if record was inserted, False if duplicate.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Prevent duplicate entries for recognized students on the same day
    if status == "Present" and is_already_marked(name, date_str):
        return False

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO attendance (name, status, confidence, date, time) VALUES (?, ?, ?, ?, ?)",
        (name, status, round(confidence, 4), date_str, time_str)
    )

    conn.commit()
    conn.close()

    return True


def get_today_attendance():
    """
    Retrieve all attendance records for today.

    Returns:
        List of dicts with keys: id, name, status, confidence, date, time.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC",
        (today,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_all_attendance():
    """
    Retrieve all attendance records.

    Returns:
        List of dicts with keys: id, name, status, confidence, date, time.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_attendance_summary():
    """
    Get attendance summary: count of unique present students today.

    Returns:
        dict with keys: total_present, total_unknown, records.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(DISTINCT name) FROM attendance WHERE date = ? AND status = 'Present'",
        (today,)
    )
    total_present = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Unknown'",
        (today,)
    )
    total_unknown = cursor.fetchone()[0]

    conn.close()

    return {
        "total_present": total_present,
        "total_unknown": total_unknown,
    }


def clear_today_attendance():
    """Delete all attendance records for today (for reset functionality)."""
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM attendance WHERE date = ?", (today,))
    conn.commit()
    conn.close()
