"""
Database layer for Customer Churn Management using SQLite.
Persists customers, churn prediction scores, SHAP risk drivers, and intervention records.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "churn_records.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a SQLite database connection with row factory enabled.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """
    Initializes database tables if they do not exist.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Customers & Churn Scores Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                tenure REAL NOT NULL,
                monthly_charges REAL NOT NULL,
                total_charges REAL NOT NULL,
                contract_type TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                internet_service TEXT NOT NULL,
                support_tickets INTEGER NOT NULL,
                last_login_days_ago REAL NOT NULL,
                usage_frequency REAL NOT NULL,
                churn_probability REAL NOT NULL,
                predicted_churn INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                top_risk_factors TEXT NOT NULL,
                top_protective_factors TEXT NOT NULL,
                recommended_interventions TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # Interventions History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Pending', 'Sent', 'Responded', 'Resolved')),
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
            );
        """)

        # Index for efficient customer lookups and filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_risk ON customers(risk_tier);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_churn_prob ON customers(churn_probability);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_customer ON interventions(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_status ON interventions(status);")

        conn.commit()


def save_or_update_customer(
    customer_raw: Dict[str, Any],
    prediction_result: Dict[str, Any],
    recommended_interventions: List[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Inserts or updates a customer along with their predicted churn score and SHAP drivers.
    """
    now = datetime.now(timezone.utc).isoformat()
    customer_id = customer_raw.get("customer_id")
    if not customer_id:
        raise ValueError("customer_id is required to persist customer")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customers (
                customer_id, tenure, monthly_charges, total_charges,
                contract_type, payment_method, internet_service,
                support_tickets, last_login_days_ago, usage_frequency,
                churn_probability, predicted_churn, risk_tier,
                top_risk_factors, top_protective_factors, recommended_interventions,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(customer_id) DO UPDATE SET
                tenure = excluded.tenure,
                monthly_charges = excluded.monthly_charges,
                total_charges = excluded.total_charges,
                contract_type = excluded.contract_type,
                payment_method = excluded.payment_method,
                internet_service = excluded.internet_service,
                support_tickets = excluded.support_tickets,
                last_login_days_ago = excluded.last_login_days_ago,
                usage_frequency = excluded.usage_frequency,
                churn_probability = excluded.churn_probability,
                predicted_churn = excluded.predicted_churn,
                risk_tier = excluded.risk_tier,
                top_risk_factors = excluded.top_risk_factors,
                top_protective_factors = excluded.top_protective_factors,
                recommended_interventions = excluded.recommended_interventions,
                updated_at = excluded.updated_at;
        """, (
            customer_id,
            float(customer_raw.get("tenure", 0)),
            float(customer_raw.get("monthly_charges", 0.0)),
            float(customer_raw.get("total_charges", 0.0)),
            str(customer_raw.get("contract_type", "Month-to-month")),
            str(customer_raw.get("payment_method", "Electronic check")),
            str(customer_raw.get("internet_service", "Fiber optic")),
            int(customer_raw.get("support_tickets", 0)),
            float(customer_raw.get("last_login_days_ago", 0)),
            float(customer_raw.get("usage_frequency", 0.0)),
            float(prediction_result["churn_probability"]),
            int(prediction_result["predicted_churn"]),
            str(prediction_result["risk_tier"]),
            json.dumps(prediction_result.get("top_risk_factors", [])),
            json.dumps(prediction_result.get("top_protective_factors", [])),
            json.dumps(recommended_interventions),
            now,
            now
        ))
        conn.commit()

    return get_customer(customer_id, db_path=db_path)


def get_customer(customer_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single customer record by customer_id, including parsed JSON fields and latest intervention.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        row = cursor.fetchone()
        if not row:
            return None

        data = dict(row)
        data["top_risk_factors"] = json.loads(data["top_risk_factors"])
        data["top_protective_factors"] = json.loads(data["top_protective_factors"])
        data["recommended_interventions"] = json.loads(data["recommended_interventions"])

        # Fetch latest intervention if any
        cursor.execute("""
            SELECT * FROM interventions
            WHERE customer_id = ?
            ORDER BY id DESC LIMIT 1
        """, (customer_id,))
        latest_inv = cursor.fetchone()
        data["latest_intervention"] = dict(latest_inv) if latest_inv else None

        return data


def get_customers(
    limit: int = 100,
    offset: int = 0,
    risk_tier: Optional[str] = None,
    min_churn_probability: Optional[float] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves a paginated list of customers with stored churn scores and latest intervention status.
    """
    query = """
        SELECT c.*,
               i.action_type AS latest_action_type,
               i.status AS latest_intervention_status,
               i.created_at AS latest_intervention_at
        FROM customers c
        LEFT JOIN (
            SELECT customer_id, action_type, status, created_at,
                   ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY id DESC) as rn
            FROM interventions
        ) i ON c.customer_id = i.customer_id AND i.rn = 1
        WHERE 1=1
    """
    params = []

    if risk_tier:
        query += " AND c.risk_tier = ?"
        params.append(risk_tier)

    if min_churn_probability is not None:
        query += " AND c.churn_probability >= ?"
        params.append(min_churn_probability)

    query += " ORDER BY c.churn_probability DESC, c.updated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            item = dict(r)
            item["top_risk_factors"] = json.loads(item["top_risk_factors"])
            item["top_protective_factors"] = json.loads(item["top_protective_factors"])
            item["recommended_interventions"] = json.loads(item["recommended_interventions"])
            results.append(item)

        return results


def count_customers(
    risk_tier: Optional[str] = None,
    min_churn_probability: Optional[float] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Counts total customers matching query criteria.
    """
    query = "SELECT COUNT(*) FROM customers WHERE 1=1"
    params = []
    if risk_tier:
        query += " AND risk_tier = ?"
        params.append(risk_tier)
    if min_churn_probability is not None:
        query += " AND churn_probability >= ?"
        params.append(min_churn_probability)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()[0]


def log_intervention(
    customer_id: str,
    action_type: str,
    status: str,
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Logs an intervention for a customer with status (Pending, Sent, Responded, Resolved).
    """
    valid_statuses = {"Pending", "Sent", "Responded", "Resolved"}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interventions (customer_id, action_type, status, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (customer_id, action_type, status, notes, now, now))
        conn.commit()
        inv_id = cursor.lastrowid

        cursor.execute("SELECT * FROM interventions WHERE id = ?", (inv_id,))
        return dict(cursor.fetchone())


def get_customer_interventions(customer_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves full intervention history for a given customer.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM interventions
            WHERE customer_id = ?
            ORDER BY id DESC
        """, (customer_id,))
        return [dict(r) for r in cursor.fetchall()]


def get_all_interventions(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves all intervention records across customers.
    """
    query = "SELECT * FROM interventions WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def update_intervention_status(
    intervention_id: int,
    status: str,
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """
    Updates the status and optional notes of an existing intervention record.
    """
    valid_statuses = {"Pending", "Sent", "Responded", "Resolved"}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if notes is not None:
            cursor.execute("""
                UPDATE interventions
                SET status = ?, notes = ?, updated_at = ?
                WHERE id = ?
            """, (status, notes, now, intervention_id))
        else:
            cursor.execute("""
                UPDATE interventions
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status, now, intervention_id))
        conn.commit()

        cursor.execute("SELECT * FROM interventions WHERE id = ?", (intervention_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_dashboard_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Computes summary statistics for the dashboard.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Total customers
        cursor.execute("SELECT COUNT(*) FROM customers")
        total_customers = cursor.fetchone()[0]
        
        # Risk distribution
        cursor.execute("SELECT risk_tier, COUNT(*) as count FROM customers GROUP BY risk_tier")
        risk_dist = [{"risk_tier": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # High risk pct
        high_risk = sum(item["count"] for item in risk_dist if item["risk_tier"] == "CRITICAL RISK")
        high_risk_pct = round((high_risk / total_customers * 100) if total_customers else 0, 1)
        
        # Revenue at risk
        cursor.execute("SELECT SUM(monthly_charges) FROM customers WHERE risk_tier = 'CRITICAL RISK'")
        revenue_at_risk = cursor.fetchone()[0] or 0.0
        
        # Avg churn probability
        cursor.execute("SELECT AVG(churn_probability) FROM customers")
        avg_churn = cursor.fetchone()[0] or 0.0
        
        # Scoring trend (mocked based on updated_at for now, group by date)
        cursor.execute("SELECT DATE(updated_at) as date, COUNT(*) as count FROM customers GROUP BY DATE(updated_at) ORDER BY date DESC LIMIT 7")
        trend = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
        trend.reverse() # chronological order
        
        return {
            "total_customers": total_customers,
            "high_risk_pct": high_risk_pct,
            "revenue_at_risk": revenue_at_risk,
            "avg_churn_probability": avg_churn,
            "risk_distribution": risk_dist,
            "scoring_trend": trend
        }
