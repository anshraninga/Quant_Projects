"""
trade_memory.py
---------------
SQLite-backed memory store for Alpha-Seeker trade outcomes.

Schema:
  trade_memory   — one row per analysis; outcomes filled in later
  agent_accuracy — rolling win/loss counters per agent × symbol
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

_DB_PATH = config.SQLITE_DB_PATH

_DDL = """
CREATE TABLE IF NOT EXISTS trade_memory (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    tech_direction    TEXT,
    tech_confidence   REAL,
    sent_direction    TEXT,
    sent_confidence   REAL,
    fund_direction    TEXT,
    fund_confidence   REAL,
    final_direction   TEXT,
    final_conviction  REAL,
    entry_price       REAL,
    outcome_24h       REAL,
    outcome_72h       REAL,
    outcome_correct   INTEGER,
    outcome_checked   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_accuracy (
    agent         TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    total_calls   INTEGER DEFAULT 0,
    correct_calls INTEGER DEFAULT 0,
    PRIMARY KEY (agent, symbol)
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_DDL)
    logger.info("Memory DB initialised at %s", _DB_PATH)


# ── Write ──────────────────────────────────────────────────

def save_analysis(
    symbol:           str,
    tech_direction:   str | None,
    tech_confidence:  float | None,
    sent_direction:   str | None,
    sent_confidence:  float | None,
    fund_direction:   str | None,
    fund_confidence:  float | None,
    final_direction:  str | None,
    final_conviction: float | None,
    entry_price:      float | None,
) -> int:
    """Insert a new analysis record and return its id."""
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO trade_memory
               (symbol, timestamp,
                tech_direction, tech_confidence,
                sent_direction, sent_confidence,
                fund_direction, fund_confidence,
                final_direction, final_conviction, entry_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, ts,
             tech_direction, tech_confidence,
             sent_direction, sent_confidence,
             fund_direction, fund_confidence,
             final_direction, final_conviction, entry_price),
        )
        return cur.lastrowid


# ── Outcome tracking ───────────────────────────────────────

def check_pending_outcomes() -> int:
    """
    Fetch price outcomes for all records older than 24 h that have not
    been checked yet.  Updates outcome_24h, outcome_72h, outcome_correct,
    and agent_accuracy.

    Returns the number of records updated.
    """
    import yfinance as yf
    from datetime import timedelta

    updated = 0
    with _conn() as con:
        rows = con.execute(
            """SELECT id, symbol, timestamp, final_direction, entry_price
               FROM trade_memory
               WHERE outcome_checked = 0
                 AND timestamp <= datetime('now', '-24 hours')"""
        ).fetchall()

    for row in rows:
        rec_id    = row["id"]
        symbol    = row["symbol"]
        ts_str    = row["timestamp"]
        direction = row["final_direction"]
        entry     = row["entry_price"]

        if not entry or not direction:
            _mark_checked(rec_id)
            continue

        try:
            ticker   = yf.Ticker(f"{symbol}-USD")
            hist     = ticker.history("5d")
            if hist.empty:
                continue
            ts_dt    = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            closes   = hist["Close"]
            closes.index = closes.index.tz_convert("UTC")
            after    = closes[closes.index >= ts_dt]
            if len(after) < 1:
                continue

            p24 = float(after.iloc[min(1, len(after) - 1)])
            p72 = float(after.iloc[min(3, len(after) - 1)])

            chg24 = (p24 - entry) / entry * 100
            chg72 = (p72 - entry) / entry * 100

            correct_24h = (
                (chg24 > 0 and direction == "LONG") or
                (chg24 < 0 and direction == "SHORT")
            )

            with _conn() as con:
                con.execute(
                    """UPDATE trade_memory
                       SET outcome_24h=?, outcome_72h=?,
                           outcome_correct=?, outcome_checked=1
                       WHERE id=?""",
                    (round(chg24, 3), round(chg72, 3),
                     int(correct_24h), rec_id),
                )
                _update_accuracy(con, symbol, row, correct_24h)

            updated += 1
            logger.info("Outcome for record %d (%s): 24h=%+.2f%% correct=%s",
                        rec_id, symbol, chg24, correct_24h)
        except Exception as exc:
            logger.warning("Outcome check failed for record %d: %s", rec_id, exc)

    return updated


def _mark_checked(rec_id: int) -> None:
    with _conn() as con:
        con.execute("UPDATE trade_memory SET outcome_checked=1 WHERE id=?", (rec_id,))


def _update_accuracy(con: sqlite3.Connection, symbol: str, row: sqlite3.Row,
                     correct: bool) -> None:
    for agent in ("technical", "sentiment", "fundamental"):
        direction_col = f"{agent[:4]}_direction"
        agent_dir = row[direction_col] if direction_col in row.keys() else None
        if agent_dir is None:
            continue
        con.execute(
            """INSERT INTO agent_accuracy (agent, symbol, total_calls, correct_calls)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(agent, symbol) DO UPDATE SET
                 total_calls   = total_calls + 1,
                 correct_calls = correct_calls + ?""",
            (agent, symbol, int(correct), int(correct)),
        )


# ── Read ───────────────────────────────────────────────────

def get_memory_context(symbol: str, n: int = 5) -> str:
    """Return a human-readable summary of the last n completed analyses."""
    with _conn() as con:
        rows = con.execute(
            """SELECT timestamp, final_direction, final_conviction,
                      outcome_24h, outcome_correct
               FROM trade_memory
               WHERE symbol = ? AND outcome_checked = 1
               ORDER BY id DESC LIMIT ?""",
            (symbol, n),
        ).fetchall()

    if not rows:
        return f"No history yet for {symbol}."

    total   = len(rows)
    correct = sum(1 for r in rows if r["outcome_correct"] == 1)
    lines   = [f"Last {total} {symbol} analyses:"]

    for r in rows:
        direction = r["final_direction"] or "?"
        chg       = r["outcome_24h"]
        ok        = "✓" if r["outcome_correct"] else "✗"
        chg_str   = f"{chg:+.2f}%" if chg is not None else "pending"
        lines.append(f"  {r['timestamp'][:10]}  {direction:9}  24h={chg_str}  {ok}")

    lines.append(f"Correct: {correct}/{total} ({correct/total*100:.0f}%)")
    return "\n".join(lines)


def get_memory_stats(symbol: str) -> dict:
    """Return accuracy stats and recent records for the /memory endpoint."""
    with _conn() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM trade_memory WHERE symbol=?", (symbol,)
        ).fetchone()[0]

        acc_rows = con.execute(
            """SELECT agent, total_calls, correct_calls
               FROM agent_accuracy WHERE symbol=?""",
            (symbol,),
        ).fetchall()

        recent = con.execute(
            """SELECT id, timestamp, final_direction, final_conviction,
                      outcome_24h, outcome_correct
               FROM trade_memory WHERE symbol=?
               ORDER BY id DESC LIMIT 5""",
            (symbol,),
        ).fetchall()

    agent_accuracy = {}
    for r in acc_rows:
        pct = round(r["correct_calls"] / r["total_calls"] * 100, 1) if r["total_calls"] else 0.0
        agent_accuracy[r["agent"]] = {
            "correct": r["correct_calls"],
            "total":   r["total_calls"],
            "pct":     pct,
        }

    return {
        "symbol":         symbol,
        "total_analyses": total,
        "agent_accuracy": agent_accuracy,
        "recent":         [dict(r) for r in recent],
    }
