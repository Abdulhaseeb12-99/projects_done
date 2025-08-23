import sqlite3
from datetime import datetime, timedelta, time
import dateparser
import re
import os
import streamlit as st
import requests


TOGETHER_API_KEY = "7f1385f475c5c7a89ad25c1cf689cd64eaffa0a132e229a0bbf0455041cbe8cd" 
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  

def ask_togetherai(question: str, knowledge: str) -> str:
    """Send user question + data.txt knowledge to Together.ai API"""
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer strictly using the provided knowledge base."},
            {"role": "user", "content": f"Knowledge Base:\n{knowledge}\n\nQuestion: {question}"}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }
    try:
        resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


DB_PATH = os.environ.get("CLINIC_DB_PATH", "clinic.db")

CLINIC_OPEN = time(9, 0)
CLINIC_CLOSE = time(17, 0)
SLOT_MINUTES = 30
WORKING_DAYS = set([0,1,2,3,4,5]) 

DEPARTMENTS = ["General Medicine", "Pediatrics", "Dermatology", "Orthopedics", "Dentistry"]

def init_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            department TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            notes TEXT,
            status TEXT NOT NULL DEFAULT 'scheduled'
        )
        """
    )
    conn.commit()

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    return conn

def round_to_slot(dt: datetime) -> datetime:
    minute = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
    return dt.replace(minute=minute, second=0, microsecond=0)

def parse_when(text: str):
    return dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})

def within_hours(dt: datetime) -> bool:
    if dt.weekday() not in WORKING_DAYS:
        return False
    t = dt.time()
    return CLINIC_OPEN <= t <= (datetime.combine(dt.date(), CLINIC_CLOSE) - timedelta(minutes=SLOT_MINUTES)).time()

def has_conflict(conn, start: datetime, end: datetime, exclude_id=None) -> bool:
    q = """
        SELECT COUNT(*) FROM appointments
        WHERE status='scheduled'
          AND ((datetime(start_time) < ? AND datetime(end_time) > ?)
               OR (datetime(start_time) >= ? AND datetime(start_time) < ?))
    """
    params = (end.isoformat(), start.isoformat(), start.isoformat(), end.isoformat())
    if exclude_id is not None:
        q += " AND id != ?"
        params = params + (exclude_id,)
    cur = conn.execute(q, params)
    (count,) = cur.fetchone()
    return count > 0

def pretty_dt(dt: datetime):
    return dt.strftime("%a, %d %b %Y %I:%M %p")

st.set_page_config(page_title="Clinic Chatbot", page_icon="🤖")
st.title("🤖 CareBot – Clinic Appointment Chatbot")

menu = st.sidebar.radio("Choose an action", ["Book", "Reschedule", "Cancel", "List Appointments", "Enquiry"])

conn = get_conn()

# Book
if menu == "Book":
    st.header("📅 Book a new appointment")
    with st.form("book_form"):
        name = st.text_input("Patient Name")
        phone = st.text_input("Phone Number")
        department = st.selectbox("Department", DEPARTMENTS)
        when_text = st.text_input("When would you like the appointment? (e.g., 'tomorrow 3 pm')")
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Book Appointment")

        if submitted:
            if not re.fullmatch(r"[0-9+\-() ]{7,20}", phone or ""):
                st.error("Please enter a valid phone number.")
            else:
                dt = parse_when(when_text)
                if not dt:
                    st.error("Could not understand the time. Try again.")
                else:
                    dt = round_to_slot(dt)
                    if not within_hours(dt):
                        st.error("That time is outside clinic hours (09:00–17:00, Mon–Sat).")
                    elif has_conflict(conn, dt, dt+timedelta(minutes=SLOT_MINUTES)):
                        st.error(f"Sorry, {pretty_dt(dt)} is already booked.")
                    else:
                        conn.execute(
                            "INSERT INTO appointments (patient_name, phone, department, start_time, end_time, notes) VALUES (?, ?, ?, ?, ?, ?)",
                            (name, phone, department, dt.isoformat(), (dt+timedelta(minutes=SLOT_MINUTES)).isoformat(), notes)
                        )
                        conn.commit()
                        appt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        st.success(f"Appointment booked! Your ID is {appt_id}")

# Reschedule
elif menu == "Reschedule":
    st.header(" Reschedule Appointment")
    appt_id = st.number_input("Appointment ID", step=1, min_value=1)
    new_time = st.text_input("New Time (e.g., 'next Monday 10 am')")
    if st.button("Reschedule"):
        appt = conn.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
        if not appt:
            st.error("Appointment not found.")
        else:
            dt = parse_when(new_time)
            if not dt:
                st.error("Could not parse time.")
            else:
                dt = round_to_slot(dt)
                if not within_hours(dt):
                    st.error("Outside clinic hours.")
                elif has_conflict(conn, dt, dt+timedelta(minutes=SLOT_MINUTES), exclude_id=appt_id):
                    st.error("That slot is already booked.")
                else:
                    conn.execute("UPDATE appointments SET start_time=?, end_time=? WHERE id=?",
                                 (dt.isoformat(), (dt+timedelta(minutes=SLOT_MINUTES)).isoformat(), appt_id))
                    conn.commit()
                    st.success("Appointment rescheduled!")

# Cancel
elif menu == "Cancel":
    st.header("🗑️ Cancel Appointment")
    appt_id = st.number_input("Appointment ID to cancel", step=1, min_value=1)
    if st.button("Cancel Appointment"):
        appt = conn.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
        if not appt:
            st.error("Appointment not found.")
        else:
            conn.execute("UPDATE appointments SET status='canceled' WHERE id=?", (appt_id,))
            conn.commit()
            st.success("🗑️ Appointment canceled successfully.")

# List
elif menu == "List Appointments":
    st.header("📋 Your Appointments")
    phone = st.text_input("Enter your phone number")
    if st.button("Show Appointments"):
        cur = conn.execute("SELECT id, patient_name, department, start_time, end_time, status FROM appointments WHERE phone=?", (phone,))
        rows = cur.fetchall()
        if not rows:
            st.warning("No appointments found.")
        else:
            for r in rows:
                st.write(f"**ID {r[0]}** | {r[1]} | {r[2]} | {pretty_dt(datetime.fromisoformat(r[3]))} - {pretty_dt(datetime.fromisoformat(r[4]))} | Status: {r[5]}")

# Enquiry (NEW)
elif menu == "Enquiry":
    st.header("❓ Enquiry Section")
    user_q = st.text_input("Ask your question")
    if st.button("Get Answer"):
        if not os.path.exists("data.txt"):
            st.error("⚠️ data.txt file not found in current directory.")
        elif not TOGETHER_API_KEY:
            st.error("⚠️ Together API key not set. Please set TOGETHER_API_KEY environment variable.")
        else:
            with open("data.txt", "r", encoding="utf-8") as f:
                knowledge = f.read()
            answer = ask_togetherai(user_q, knowledge)
            st.success("Answer:")
            st.write(answer)
