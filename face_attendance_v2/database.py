import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables, creates default admin, and migrates dataset folders."""
    conn = get_db()
    cursor = conn.cursor()

    # 1. Students Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            section TEXT DEFAULT 'A',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Admins Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            created_by INTEGER,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (created_by) REFERENCES admins (id)
        )
    """)

    # 4. Attendance Table with DB-level UNIQUE constraint on (student_id, session_id)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (session_id) REFERENCES sessions (id),
            UNIQUE(student_id, session_id)
        )
    """)

    # 5. Attempts Table (Full Audit Trail)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id INTEGER,
            result TEXT NOT NULL,
            student_id INTEGER,
            distance REAL,
            snapshot_path TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id),
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    """)

    # Check and create default admin account (admin / admin123)
    cursor.execute("SELECT * FROM admins WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        hashed_pw = generate_password_hash("admin123")
        cursor.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)", ("admin", hashed_pw))
        print("[DB INFO] Created default admin user: 'admin' (password: 'admin123')")

    conn.commit()
    conn.close()

    # Migrate existing dataset folders from parent/local dataset directory into DB
    migrate_existing_dataset()

def migrate_existing_dataset():
    """Migrates any existing student folders in dataset/ to the database."""
    base_dir = os.path.dirname(__file__)
    dataset_dirs = [
        os.path.join(base_dir, "dataset"),
        os.path.join(os.path.dirname(base_dir), "dataset")
    ]

    conn = get_db()
    cursor = conn.cursor()

    for ddir in dataset_dirs:
        if os.path.exists(ddir):
            student_folders = [f for f in os.listdir(ddir) if os.path.isdir(os.path.join(ddir, f))]
            for folder in student_folders:
                roll = folder.replace(" ", "_").upper()
                name = folder.replace("_", " ").title()
                cursor.execute("SELECT id FROM students WHERE roll_no = ?", (roll,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO students (name, roll_no, section) VALUES (?, ?, ?)", (name, roll, "A"))
                    print(f"[DB MIGRATION] Added student '{name}' (Roll: {roll}) from dataset folder.")

    conn.commit()
    conn.close()

# Authentication Helpers
def get_admin_by_username(username):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username = ?", (username,))
    admin = cursor.fetchone()
    conn.close()
    return admin

def get_student_by_roll_no(roll_no):
    return get_student_by_identifier(roll_no)

def get_student_by_identifier(identifier):
    if not identifier:
        return None
    conn = get_db()
    cursor = conn.cursor()
    clean_id = identifier.strip()
    id_upper = clean_id.upper()
    id_underscore = id_upper.replace(" ", "_")
    id_space = id_upper.replace("_", " ")

    cursor.execute("""
        SELECT * FROM students 
        WHERE UPPER(roll_no) = ? OR UPPER(roll_no) = ? 
           OR UPPER(name) = ? OR UPPER(name) = ?
        ORDER BY id ASC LIMIT 1
    """, (id_upper, id_underscore, id_upper, id_space))
    student = cursor.fetchone()
    conn.close()
    return student

def add_student(name, roll_no, section="A"):
    conn = get_db()
    cursor = conn.cursor()
    roll_clean = roll_no.strip().upper()
    try:
        cursor.execute("INSERT INTO students (name, roll_no, section) VALUES (?, ?, ?)", (name.strip(), roll_clean, section.strip()))
        conn.commit()
        student_id = cursor.lastrowid
        conn.close()
        return student_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_all_students():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY name ASC")
    students = cursor.fetchall()
    conn.close()
    return students

# Session Management Helpers
def create_session(class_name, created_by=1):
    conn = get_db()
    cursor = conn.cursor()
    # Close any currently active session first
    cursor.execute("UPDATE sessions SET status = 'closed', end_time = ? WHERE status = 'active'", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO sessions (class_name, start_time, created_by, status) VALUES (?, ?, ?, 'active')", (class_name, start_time, created_by))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def close_active_session():
    conn = get_db()
    cursor = conn.cursor()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE sessions SET status = 'closed', end_time = ? WHERE status = 'active'", (end_time,))
    conn.commit()
    conn.close()

def get_active_session():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1")
    session = cursor.fetchone()
    conn.close()
    return session

def get_all_sessions():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, a.username as admin_name,
               (SELECT COUNT(*) FROM attendance WHERE session_id = s.id) as attendance_count
        FROM sessions s
        LEFT JOIN admins a ON s.created_by = a.id
        ORDER BY s.id DESC
    """)
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# Attendance & Attempts Helpers
def log_attendance(student_id, session_id):
    """Attempts to write attendance. Enforced by UNIQUE(student_id, session_id). Returns True if added."""
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        cursor.execute("""
            INSERT INTO attendance (student_id, session_id, date, time)
            VALUES (?, ?, ?, ?)
        """, (student_id, session_id, date_str, time_str))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Already marked for this session

def log_attempt(session_id, result, student_id=None, distance=None, snapshot_path=None):
    """Logs every recognition frame event to attempts audit trail."""
    conn = get_db()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO attempts (timestamp, session_id, result, student_id, distance, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, session_id, result, student_id, distance, snapshot_path))
    conn.commit()
    conn.close()

def get_all_attendance(session_id=None):
    conn = get_db()
    cursor = conn.cursor()
    if session_id:
        cursor.execute("""
            SELECT a.*, st.name as student_name, st.roll_no, st.section, se.class_name
            FROM attendance a
            JOIN students st ON a.student_id = st.id
            JOIN sessions se ON a.session_id = se.id
            WHERE a.session_id = ?
            ORDER BY a.time DESC
        """, (session_id,))
    else:
        cursor.execute("""
            SELECT a.*, st.name as student_name, st.roll_no, st.section, se.class_name
            FROM attendance a
            JOIN students st ON a.student_id = st.id
            JOIN sessions se ON a.session_id = se.id
            ORDER BY a.id DESC
        """)
    records = cursor.fetchall()
    conn.close()
    return records

def get_student_attendance(student_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.*, se.class_name, se.start_time as session_start
        FROM attendance a
        JOIN sessions se ON a.session_id = se.id
        WHERE a.student_id = ?
        ORDER BY a.id DESC
    """, (student_id,))
    records = cursor.fetchall()

    # Total sessions
    cursor.execute("SELECT COUNT(*) as total FROM sessions")
    total_sessions = cursor.fetchone()["total"]

    conn.close()
    return records, total_sessions

def get_all_attempts(limit=100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT att.*, st.name as student_name, st.roll_no, se.class_name
        FROM attempts att
        LEFT JOIN students st ON att.student_id = st.id
        LEFT JOIN sessions se ON att.session_id = se.id
        ORDER BY att.id DESC
        LIMIT ?
    """, (limit,))
    attempts = cursor.fetchall()
    conn.close()
    return attempts

def get_dashboard_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM students")
    total_students = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM sessions")
    total_sessions = cursor.fetchone()["count"]

    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) as count FROM attendance WHERE date = ?", (today_str,))
    today_attendance = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM attempts WHERE result = 'liveness_fail'")
    liveness_alerts = cursor.fetchone()["count"]

    conn.close()
    return {
        "total_students": total_students,
        "total_sessions": total_sessions,
        "today_attendance": today_attendance,
        "liveness_alerts": liveness_alerts
    }
