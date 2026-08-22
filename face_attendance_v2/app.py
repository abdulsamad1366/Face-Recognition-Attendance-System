import os
import io
import csv
import base64
from functools import wraps
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, send_file
from werkzeug.security import check_password_hash

import database
import face_engine
from camera_stream import stream_manager

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_face_attendance_key_2026")

# Initialize SQLite Database on startup
with app.app_context():
    database.init_db()

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_role" not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("login"))
            if role and session.get("user_role") != role and role != "any":
                flash("Unauthorized access.", "danger")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/")
def index():
    if "user_role" in session:
        if session["user_role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        elif session["user_role"] == "student":
            return redirect(url_for("student_dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_type = request.form.get("login_type")  # 'admin' or 'student'

        if login_type == "admin":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            
            admin = database.get_admin_by_username(username)
            if admin and check_password_hash(admin["password_hash"], password):
                session["user_role"] = "admin"
                session["user_id"] = admin["id"]
                session["username"] = admin["username"]
                flash("Logged in successfully as Admin.", "success")
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Invalid admin username or password.", "danger")

        elif login_type == "student":
            roll_no = request.form.get("roll_no", "").strip().upper()
            student = database.get_student_by_roll_no(roll_no)
            if student:
                session["user_role"] = "student"
                session["student_id"] = student["id"]
                session["student_name"] = student["name"]
                session["roll_no"] = student["roll_no"]
                flash(f"Welcome back, {student['name']}!", "success")
                return redirect(url_for("student_dashboard"))
            else:
                flash(f"Roll Number '{roll_no}' not found in system.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ADMIN ROUTES
@app.route("/admin/dashboard")
@login_required(role="admin")
def admin_dashboard():
    stats = database.get_dashboard_stats()
    active_session = database.get_active_session()
    recent_attempts = database.get_all_attempts(limit=10)
    return render_template("admin_dashboard.html", stats=stats, active_session=active_session, attempts=recent_attempts)

@app.route("/admin/session", methods=["GET", "POST"])
@login_required(role="admin")
def session_control():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "start":
            class_name = request.form.get("class_name", "General Class").strip()
            session_id = database.create_session(class_name, created_by=session.get("user_id", 1))
            flash(f"New attendance session started for '{class_name}'!", "success")
        elif action == "stop":
            database.close_active_session()
            flash("Active attendance session stopped.", "info")
        return redirect(url_for("session_control"))

    active_session = database.get_active_session()
    sessions = database.get_all_sessions()
    return render_template("session_control.html", active_session=active_session, sessions=sessions)

@app.route("/admin/register_student", methods=["GET", "POST"])
@login_required(role="admin")
def register_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip().upper()
        section = request.form.get("section", "A").strip()

        if not name or not roll_no:
            flash("Student Name and Roll Number are required.", "warning")
            return redirect(url_for("register_student"))

        # Save student in database
        student_id = database.add_student(name, roll_no, section)
        if not student_id:
            flash(f"Student with Roll Number '{roll_no}' already exists.", "danger")
            return redirect(url_for("register_student"))

        # Create dataset directory
        dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", roll_no)
        os.makedirs(dataset_dir, exist_ok=True)

        # Handle uploaded images or webcam snapshots
        snapshots_data = request.form.getlist("snapshots")  # list of base64 data URLs
        saved_paths = []

        for idx, b64_str in enumerate(snapshots_data, 1):
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            try:
                img_bytes = base64.b64decode(b64_str)
                file_path = os.path.join(dataset_dir, f"{idx}.jpg")
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(file_path)
            except Exception as e:
                print(f"[REGISTRATION ERROR] Failed to save image {idx}: {e}")

        # If files uploaded via standard file input
        uploaded_files = request.files.getlist("image_files")
        for idx, file_obj in enumerate(uploaded_files, len(saved_paths) + 1):
            if file_obj and file_obj.filename:
                file_path = os.path.join(dataset_dir, f"{idx}.jpg")
                file_obj.save(file_path)
                saved_paths.append(file_path)

        if saved_paths:
            # Trigger Face Engine Encoding
            enc_count = face_engine.encode_student_images(roll_no, saved_paths)
            flash(f"Successfully registered '{name}'! {enc_count} face encodings generated.", "success")
        else:
            flash(f"Student '{name}' registered in database, but no face photos were captured.", "warning")

        return redirect(url_for("admin_dashboard"))

    students = database.get_all_students()
    return render_template("register_student.html", students=students)

@app.route("/admin/live_feed")
@login_required(role="admin")
def live_feed():
    active_session = database.get_active_session()
    return render_template("live_feed.html", active_session=active_session)

@app.route("/video_feed")
@login_required(role="admin")
def video_feed():
    """MJPEG streaming route."""
    return Response(stream_manager.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/admin/attendance")
@login_required(role="admin")
def view_attendance():
    session_id = request.args.get("session_id", type=int)
    records = database.get_all_attendance(session_id=session_id)
    sessions = database.get_all_sessions()
    return render_template("attendance.html", records=records, sessions=sessions, selected_session=session_id)

@app.route("/admin/export_csv")
@login_required(role="admin")
def export_csv():
    session_id = request.args.get("session_id", type=int)
    records = database.get_all_attendance(session_id=session_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student Name", "Roll Number", "Section", "Class Session", "Date", "Time"])

    for r in records:
        writer.writerow([r["student_name"], r["roll_no"], r["section"], r["class_name"], r["date"], r["time"]])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=attendance_export.csv"}
    )

@app.route("/admin/attempt_log")
@login_required(role="admin")
def attempt_log():
    attempts = database.get_all_attempts(limit=100)
    return render_template("attempt_log.html", attempts=attempts)

# STUDENT ROUTES
@app.route("/student/dashboard")
@login_required(role="student")
def student_dashboard():
    student_id = session.get("student_id")
    student_name = session.get("student_name")
    roll_no = session.get("roll_no")

    records, total_sessions = database.get_student_attendance(student_id)
    present_count = len(records)
    percentage = round((present_count / total_sessions * 100), 1) if total_sessions > 0 else 0.0

    return render_template(
        "student_dashboard.html",
        student_name=student_name,
        roll_no=roll_no,
        records=records,
        present_count=present_count,
        total_sessions=total_sessions,
        percentage=percentage
    )

if __name__ == "__main__":
    print("==================================================")
    print("  Face Recognition Attendance System v2 (Flask)   ")
    print("  Running on http://127.0.0.1:5000               ")
    print("==================================================")
    app.run(host="127.0.0.1", port=5000, debug=True)
