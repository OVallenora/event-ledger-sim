import sqlite3
import threading
from typing import Optional
from model import SpatialRiskDetector, TransactionEvent


class ThreadSafeLedger:
    def __init__(self):
        # In-memory shared SQLite database across threads
        self.conn = sqlite3.connect("file:memdb1?mode=memory&cache=shared", check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    user_id TEXT PRIMARY KEY,
                    balance REAL NOT NULL CHECK (balance >= 0)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def create_account(self, user_id: str, initial_balance: float) -> None:
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO accounts (user_id, balance) VALUES (?, ?);",
                (user_id, initial_balance)
            )

    def get_balance(self, user_id: str) -> Optional[float]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT balance FROM accounts WHERE user_id = ?;", (user_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def process_transaction(
        self,
        event: TransactionEvent,
        detector: SpatialRiskDetector
    ) -> bool:
        """
        Executes atomic debit and credit operations. Rejects transactions failing spatial checks.
        """
        if event.amount <= 0:
            raise ValueError("Amount must be strictly positive.")

        # Spatial Anomaly Evaluation
        if detector.is_anomalous(event.latitude, event.longitude):
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT INTO audit_log (sender_id, recipient_id, amount, status) VALUES (?, ?, ?, ?);",
                    (event.sender_id, event.recipient_id, event.amount, "REJECTED_SPATIAL_ANOMALY")
                )
            return False

        # Thread-safe atomic execution
        with self.lock, self.conn:
            cur = self.conn.cursor()
            
            # Fetch balances
            cur.execute("SELECT balance FROM accounts WHERE user_id = ?;", (event.sender_id,))
            sender_row = cur.fetchone()
            cur.execute("SELECT balance FROM accounts WHERE user_id = ?;", (event.recipient_id,))
            recipient_row = cur.fetchone()

            if not sender_row or not recipient_row:
                raise KeyError("One or both accounts do not exist.")

            if sender_row[0] < event.amount:
                cur.execute(
                    "INSERT INTO audit_log (sender_id, recipient_id, amount, status) VALUES (?, ?, ?, ?);",
                    (event.sender_id, event.recipient_id, event.amount, "REJECTED_INSUFFICIENT_FUNDS")
                )
                return False

            # Transfer balances
            cur.execute(
                "UPDATE accounts SET balance = balance - ? WHERE user_id = ?;",
                (event.amount, event.sender_id)
            )
            cur.execute(
                "UPDATE accounts SET balance = balance + ? WHERE user_id = ?;",
                (event.amount, event.recipient_id)
            )
            cur.execute(
                "INSERT INTO audit_log (sender_id, recipient_id, amount, status) VALUES (?, ?, ?, ?);",
                (event.sender_id, event.recipient_id, event.amount, "CLEARED")
            )
            return True
