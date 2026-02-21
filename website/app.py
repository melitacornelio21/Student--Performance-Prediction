import os
import pickle
import joblib
import pandas as pd
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

BASE_DIR = os.path.dirname(__file__)

# ================= DATABASE =================

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

create_users_table()

# ================= MODEL =================

POSSIBLE_ARTIFACT_DIRS = [
    os.path.join(BASE_DIR, "artifacts"),
    os.path.join(BASE_DIR, "notebook", "data"),
    BASE_DIR
]

def find_file(filename):
    for d in POSSIBLE_ARTIFACT_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = find_file("model.pkl")
PREPROCESSOR_PATH = find_file("preprocessor.pkl")

def load_object(path):
    try:
        return joblib.load(path)
    except:
        with open(path, "rb") as f:
            return pickle.load(f)

model = load_object(MODEL_PATH)
preprocessor = load_object(PREPROCESSOR_PATH)

FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score"
]

# ================= UTIL FUNCTIONS =================

def build_record_from_form(form):
    return {
        "gender": form.get("gender"),
        "race_ethnicity": form.get("ethnicity"),
        "parental_level_of_education": form.get("parental_level_of_education"),
        "lunch": form.get("lunch"),
        "test_preparation_course": form.get("test_preparation_course"),
        "reading_score": form.get("reading_score"),
        "writing_score": form.get("writing_score")
    }

def convert_types(record):
    for key in ["reading_score", "writing_score"]:
        record[key] = float(record[key])
    return record

# ================= ROUTES =================

# ---------- LOGIN PAGE ----------
@app.route("/", methods=["GET"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")
            return redirect(url_for("login"))

    return render_template("login.html")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if user:
            conn.close()
            flash("Username already exists")
            return redirect(url_for("signup"))

        hashed = generate_password_hash(password)

        conn.execute("INSERT INTO users (username,password) VALUES (?,?)",
                     (username, hashed))
        conn.commit()
        conn.close()

        flash("Account created! Please login")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ---------- HOME ----------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("home.html", username=session["user"])

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ================= PREDICTION =================

@app.route("/predict_datapoint", methods=["POST"])
def predict_datapoint():

    if "user" not in session:
        return redirect(url_for("login"))

    try:
        raw_record = build_record_from_form(request.form)
        record = convert_types(raw_record)

        X = pd.DataFrame([record], columns=FEATURES)

        if preprocessor is not None:
            X_pre = preprocessor.transform(X)
        else:
            X_pre = X.values

        prediction = model.predict(X_pre)[0]

        return render_template(
            "home.html",
            results=round(float(prediction), 2),
            username=session["user"]
        )

    except Exception as e:
        flash("Prediction failed: " + str(e))
        return redirect(url_for("home"))

# ================= RUN =================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
