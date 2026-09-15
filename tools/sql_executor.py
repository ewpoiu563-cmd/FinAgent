"""Bounded, read-only SQLite execution without changing the supplied SQL."""

import math
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

from .sql_validator import validate_sql


def execute_sql(
    sql: str,
    db_path: str = "data/finance.db",
    max_rows: int = 100,
    timeout_seconds: float = 5.0,
) -> dict:
    """Execute a SELECT; latency includes validation, fetching and connection close.

    max_rows must be a positive integer. The deadline covers execution and
    fetching; SQLite's lock wait is also bounded by the remaining time.
    Progress callbacks interrupt SQLite VM work, not arbitrary blocking OS I/O.
    """
    started = perf_counter()
    result = {
        "success": False, "columns": [], "rows": [], "row_count": 0,
        "truncated": False, "latency_ms": 0.0, "error_type": None, "error": None,
    }
    connection = None
    expired = False

    def fail(kind, message):
        result.update(error_type=kind, error=message)

    try:
        validation = validate_sql(sql)
        if not validation["valid"]:
            fail("validation_error", validation["error"])
        elif type(max_rows) is not int or not 1 <= max_rows < sys.maxsize:
            fail("validation_error", "max_rows must be a positive bounded integer")
        elif (type(timeout_seconds) not in (int, float)
              or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            fail("validation_error", "timeout_seconds must be finite and positive")
        else:
            deadline = started + timeout_seconds

            def progress():
                nonlocal expired
                expired = perf_counter() >= deadline
                return int(expired)

            # as_uri escapes URI metacharacters in filesystem paths.
            uri = Path(db_path).resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(
                uri, uri=True, timeout=max(0.0, deadline - perf_counter())
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")

            def authorize(action, arg1, arg2, database, source):
                if action == sqlite3.SQLITE_FUNCTION:
                    return (sqlite3.SQLITE_DENY if (arg2 or "").lower()
                            == "load_extension" else sqlite3.SQLITE_OK)
                return (sqlite3.SQLITE_OK if action in (
                    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                ) else sqlite3.SQLITE_DENY)

            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1000)
            if progress():
                raise TimeoutError("query deadline exceeded")
            cursor = connection.execute(sql)
            fetched = cursor.fetchmany(max_rows + 1)
            rows = [dict(row) for row in fetched[:max_rows]]
            if progress():
                raise TimeoutError("query deadline exceeded")
            result.update(
                success=True, columns=[column[0] for column in cursor.description],
                rows=rows, row_count=len(rows), truncated=len(fetched) > max_rows,
            )
    except TimeoutError as exc:
        fail("timeout", str(exc))
    except sqlite3.Error as exc:
        timed_out = expired or (
            connection is not None and perf_counter() >= deadline
        )
        fail("timeout" if timed_out else "sqlite_error", str(exc))
    except (TypeError, ValueError, OSError, OverflowError) as exc:
        fail("validation_error", str(exc))
    finally:
        if connection is not None:
            connection.close()
        result["latency_ms"] = round((perf_counter() - started) * 1000, 4)
    return result
