"""SQLite persistence layer for risk profiles, audit logs, and investigator actions."""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd

DB_PATH = "data/revguard.db"

def get_connection():
    """
    Establish a connection to the SQLite database.
    Returns a connection object with row_factory set to sqlite3.Row.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database and create necessary tables if they don't exist.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Table risk_profiles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_profiles (
                user_id TEXT PRIMARY KEY,
                risk_score REAL,
                risk_band TEXT,
                financial_exposure REAL,
                total_returns INTEGER,
                return_frequency REAL,
                days_active INTEGER,
                reason_diversity REAL,
                top_reason_ratio REAL,
                last_updated TEXT,
                override_status TEXT DEFAULT NULL,
                override_analyst TEXT DEFAULT NULL,
                override_timestamp TEXT DEFAULT NULL,
                override_notes TEXT DEFAULT NULL
            )
        ''')

        # Table audit_logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                detail TEXT NOT NULL,
                user_id TEXT DEFAULT NULL
            )
        ''')

        # Table investigator_actions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS investigator_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                analyst_name TEXT NOT NULL,
                notes TEXT DEFAULT NULL
            )
        ''')

        # Table pipeline_runs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                total_users INTEGER,
                high_risk_count INTEGER,
                medium_risk_count INTEGER,
                low_risk_count INTEGER,
                avg_risk_score REAL,
                total_financial_exposure REAL
            )
        ''')

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error during init_db: {e}")

def save_risk_profiles(user_features_df: pd.DataFrame) -> None:
    """
    Upsert risk profiles from a DataFrame into the risk_profiles table.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for _, row in user_features_df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO risk_profiles (
                    user_id, risk_score, risk_band, financial_exposure, 
                    total_returns, return_frequency, days_active, 
                    reason_diversity, top_reason_ratio, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['user_id'], 
                float(row['Risk Score']), 
                row['Risk Band'], 
                float(row['Financial Exposure ($)']), 
                int(row['Total Returns']),
                float(row['Return Frequency']),
                int(row['Days Active']),
                float(row['Reason Diversity']),
                float(row['Top Reason Ratio']),
                current_time
            ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error in save_risk_profiles: {e}")

def get_risk_profile(user_id: str) -> Optional[dict]:
    """
    Retrieve one user row from risk_profiles as a dictionary.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM risk_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"DB error in get_risk_profile: {e}")
        return None

def add_audit_log(action: str, actor: str, detail: str, user_id: str = None) -> None:
    """
    Insert a system log entry into the audit_logs table.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (timestamp, action, actor, detail, user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, actor, detail, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error in add_audit_log: {e}")

def get_audit_logs() -> list:
    """
    Return all audit logs ordered by timestamp DESC.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"DB error in get_audit_logs: {e}")
        return []

def save_investigator_action(user_id: str, action_type: str, analyst_name: str, notes: str = None) -> None:
    """
    Record an investigator action and add a corresponding audit log.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO investigator_actions (timestamp, user_id, action_type, analyst_name, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, action_type, analyst_name, notes))
        conn.commit()
        conn.close()
        
        add_audit_log(
            action="Investigator Action",
            actor=analyst_name,
            detail=f"Analyst {analyst_name} marked {user_id} as {action_type}",
            user_id=user_id
        )
    except Exception as e:
        print(f"DB error in save_investigator_action: {e}")

def update_user_override(user_id: str, override_status: str, analyst_name: str, notes: str = None) -> None:
    """
    Update override fields in risk_profiles for a given user_id.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE risk_profiles 
            SET override_status = ?, 
                override_analyst = ?, 
                override_timestamp = ?, 
                override_notes = ?
            WHERE user_id = ?
        ''', (override_status, analyst_name, current_time, notes, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error in update_user_override: {e}")

def save_pipeline_run(total_users: int, high_risk: int, medium_risk: int, low_risk: int, avg_score: float, total_exposure: float) -> None:
    """
    Save the results of a pipeline run.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO pipeline_runs (
                run_timestamp, total_users, high_risk_count, 
                medium_risk_count, low_risk_count, avg_risk_score, 
                total_financial_exposure
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            total_users, high_risk, medium_risk, low_risk, 
            avg_score, total_exposure
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error in save_pipeline_run: {e}")

def get_pipeline_runs() -> list:
    """
    Return all pipeline runs ordered by run_timestamp DESC.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_runs ORDER BY run_timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"DB error in get_pipeline_runs: {e}")
        return []

init_db()
