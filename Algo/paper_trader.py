"""
paper_trader.py
---------------
Automated daily paper-trading journal for Alpha-Seeker.

Schedule with Windows Task Scheduler to run once at 08:00 UTC daily.
No FastAPI server needed — calls the graph directly.

Usage:
    python paper_trader.py           # run daily cycle (close + open)
    python paper_trader.py --status  # show open positions only
    python paper_trader.py --history # show last 30 trades
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: ensures sys.path is set up
from alpha_seeker.graph.graph import run_analysis

# ── Settings ───────────────────────────────────────────────────
COINS              = ["BTC", "ETH", "SOL", "BNB", "XRP"]
PORTFOLIO_USD      = 10_000.0
MIN_CONVICTION     = 0.55
MAX_POSITIONS      = 3

def _alloc(conviction: float) -> float:
    if conviction >= 0.75: return 0.20
    if conviction >= 0.65: return 0.15
    return 0.10

DB_PATH    = Path("data/paper_trading.db")
EXCEL_PATH = Path("data/paper_trading.xlsx")


# ── Database ────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_date      TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            conviction      REAL NOT NULL,
            agent_agreement TEXT,
            tech_dir        TEXT,
            sent_dir        TEXT,
            fund_dir        TEXT,
            allocation_usd  REAL NOT NULL,
            reasoning       TEXT
        );
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_date      TEXT NOT NULL,
            exit_date       TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            pnl_pct         REAL NOT NULL,
            pnl_usd         REAL NOT NULL,
            conviction      REAL NOT NULL,
            agent_agreement TEXT,
            tech_dir        TEXT,
            sent_dir        TEXT,
            fund_dir        TEXT,
            allocation_usd  REAL NOT NULL,
            reasoning       TEXT
        );
    """)
    conn.commit()
    return conn


# ── Price fetcher ───────────────────────────────────────────────

def _price(symbol: str) -> float | None:
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        hist   = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    # Fallback: Binance spot
    try:
        import requests
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": f"{symbol}USDT"}, timeout=5
        )
        return float(r.json()["price"])
    except Exception:
        return None


# ── Close yesterday's positions ─────────────────────────────────

def close_positions(conn: sqlite3.Connection) -> list[dict]:
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    positions = conn.execute(
        "SELECT * FROM positions WHERE entry_date != ?", (today,)
    ).fetchall()

    closed = []
    for pos in positions:
        exit_price = _price(pos["symbol"])
        if exit_price is None:
            print(f"  [WARN] Could not fetch price for {pos['symbol']} — skipping close")
            continue

        direction = pos["direction"]
        multiplier = 1.0 if direction == "LONG" else -1.0
        pnl_pct   = multiplier * (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        pnl_usd   = pos["allocation_usd"] * pnl_pct / 100

        conn.execute("""
            INSERT INTO trades
              (symbol, direction, entry_date, exit_date,
               entry_price, exit_price, pnl_pct, pnl_usd,
               conviction, agent_agreement, tech_dir, sent_dir, fund_dir,
               allocation_usd, reasoning)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos["symbol"], direction, pos["entry_date"], today,
            pos["entry_price"], exit_price, round(pnl_pct, 3), round(pnl_usd, 2),
            pos["conviction"], pos["agent_agreement"],
            pos["tech_dir"], pos["sent_dir"], pos["fund_dir"],
            pos["allocation_usd"], pos["reasoning"],
        ))
        conn.execute("DELETE FROM positions WHERE id = ?", (pos["id"],))
        conn.commit()

        sign = "+" if pnl_pct >= 0 else ""
        print(f"  CLOSED {pos['symbol']:4s} {direction:<5s} "
              f"entry={pos['entry_price']:>10,.2f}  exit={exit_price:>10,.2f}  "
              f"P&L {sign}{pnl_pct:.2f}%  ({sign}${pnl_usd:.2f})")
        closed.append(dict(pos) | {"exit_price": exit_price, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd})

    return closed


# ── Run Alpha-Seeker for one coin ───────────────────────────────

async def _analyse(symbol: str) -> dict | None:
    try:
        state = await run_analysis(symbol)
        rec   = state.get("final_recommendation")
        tech  = state.get("tech_thesis")
        sent  = state.get("sent_thesis")
        fund  = state.get("fund_thesis")
        if rec is None:
            return None
        return {
            "symbol":          symbol,
            "direction":       rec.direction,
            "conviction":      rec.conviction,
            "agent_agreement": rec.agent_agreement,
            "reasoning":       rec.reasoning[:300],
            "tech_dir":        tech.direction if tech else "N/A",
            "sent_dir":        sent.direction if sent else "N/A",
            "fund_dir":        fund.direction if fund else "N/A",
        }
    except Exception as exc:
        print(f"  [ERROR] Analysis failed for {symbol}: {exc}")
        return None


# ── Open new positions ──────────────────────────────────────────

async def open_positions(conn: sqlite3.Connection) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # How many slots are free?
    open_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    slots = MAX_POSITIONS - open_count
    if slots <= 0:
        print(f"  Max positions ({MAX_POSITIONS}) already open — no new entries.")
        return []

    print(f"\n  Running analysis for: {', '.join(COINS)}")
    results = await asyncio.gather(*[_analyse(c) for c in COINS])

    # Filter and rank
    signals = [
        r for r in results
        if r and r["direction"] in ("LONG", "SHORT")
        and r["conviction"] >= MIN_CONVICTION
    ]
    signals.sort(key=lambda x: x["conviction"], reverse=True)

    opened = []
    for sig in signals[:slots]:
        symbol = sig["symbol"]

        # Skip if already holding this coin
        already = conn.execute(
            "SELECT id FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if already:
            print(f"  SKIP   {symbol} — already in portfolio")
            continue

        entry_price = _price(symbol)
        if entry_price is None:
            print(f"  SKIP   {symbol} — price unavailable")
            continue

        alloc_usd = PORTFOLIO_USD * _alloc(sig["conviction"])
        conn.execute("""
            INSERT INTO positions
              (symbol, direction, entry_date, entry_price, conviction,
               agent_agreement, tech_dir, sent_dir, fund_dir,
               allocation_usd, reasoning)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol, sig["direction"], today, entry_price, sig["conviction"],
            sig["agent_agreement"], sig["tech_dir"], sig["sent_dir"], sig["fund_dir"],
            alloc_usd, sig["reasoning"],
        ))
        conn.commit()

        print(f"  OPENED {symbol:4s} {sig['direction']:<5s} "
              f"@ {entry_price:>10,.2f}  conviction={sig['conviction']:.2f}  "
              f"({sig['agent_agreement']})  alloc=${alloc_usd:.0f}")
        opened.append(sig | {"entry_price": entry_price, "allocation_usd": alloc_usd})

    if not signals:
        print("  No signals met entry criteria today.")

    return opened


# ── Excel export ────────────────────────────────────────────────

def _pct_colour(val: float) -> str:
    return "00AA44" if val >= 0 else "CC2222"


def export_excel(conn: sqlite3.Connection) -> None:
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        print("  [INFO] openpyxl not installed — skipping Excel export (pip install openpyxl)")
        return

    wb = openpyxl.Workbook()

    # ── Sheet 1: Trade Log ──────────────────────────────────────
    ws = wb.active
    ws.title = "Trade Log"
    headers = ["Symbol","Direction","Entry Date","Exit Date",
               "Entry $","Exit $","P&L %","P&L $",
               "Conviction","Agreement","Tech","Sent","Fund","Alloc $"]
    ws.append(headers)
    for c in ws[1]:
        c.font      = Font(bold=True, color="FFFFFF")
        c.fill      = PatternFill("solid", fgColor="1A1A2E")
        c.alignment = Alignment(horizontal="center")

    trades = conn.execute(
        "SELECT * FROM trades ORDER BY exit_date DESC"
    ).fetchall()
    for row in trades:
        ws.append([
            row["symbol"], row["direction"],
            row["entry_date"], row["exit_date"],
            round(row["entry_price"], 2), round(row["exit_price"], 2),
            round(row["pnl_pct"], 2), round(row["pnl_usd"], 2),
            round(row["conviction"], 2), row["agent_agreement"],
            row["tech_dir"], row["sent_dir"], row["fund_dir"],
            round(row["allocation_usd"], 0),
        ])
        # Colour the P&L columns
        pnl_cell = ws.cell(ws.max_row, 7)
        pnl_cell.fill = PatternFill("solid", fgColor=_pct_colour(row["pnl_pct"]))
        pnl_cell.font = Font(bold=True, color="FFFFFF")

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 14

    # ── Sheet 2: Open Positions ─────────────────────────────────
    ws2 = wb.create_sheet("Open Positions")
    ws2.append(["Symbol","Direction","Entry Date","Entry $",
                "Current $","Unrealised %","Conviction","Agreement",
                "Tech","Sent","Fund","Alloc $"])
    for c in ws2[1]:
        c.font      = Font(bold=True, color="FFFFFF")
        c.fill      = PatternFill("solid", fgColor="1A1A2E")
        c.alignment = Alignment(horizontal="center")

    positions = conn.execute("SELECT * FROM positions").fetchall()
    for pos in positions:
        cur = _price(pos["symbol"])
        if cur:
            mult    = 1.0 if pos["direction"] == "LONG" else -1.0
            unreal  = mult * (cur - pos["entry_price"]) / pos["entry_price"] * 100
        else:
            cur    = "N/A"
            unreal = 0.0
        ws2.append([
            pos["symbol"], pos["direction"], pos["entry_date"],
            round(pos["entry_price"], 2),
            round(cur, 2) if isinstance(cur, float) else cur,
            round(unreal, 2),
            round(pos["conviction"], 2), pos["agent_agreement"],
            pos["tech_dir"], pos["sent_dir"], pos["fund_dir"],
            round(pos["allocation_usd"], 0),
        ])
        if isinstance(unreal, float):
            cell = ws2.cell(ws2.max_row, 6)
            cell.fill = PatternFill("solid", fgColor=_pct_colour(unreal))
            cell.font = Font(bold=True, color="FFFFFF")

    for col in ws2.columns:
        ws2.column_dimensions[col[0].column_letter].width = 14

    # ── Sheet 3: Performance ────────────────────────────────────
    ws3 = wb.create_sheet("Performance")
    all_trades = conn.execute("SELECT * FROM trades").fetchall()
    if all_trades:
        total     = len(all_trades)
        wins      = sum(1 for t in all_trades if t["pnl_pct"] >= 0)
        win_rate  = wins / total * 100
        total_pnl = sum(t["pnl_usd"] for t in all_trades)
        avg_win   = sum(t["pnl_pct"] for t in all_trades if t["pnl_pct"] >= 0) / max(wins, 1)
        avg_loss  = sum(t["pnl_pct"] for t in all_trades if t["pnl_pct"] < 0) / max(total - wins, 1)

        rows = [
            ["Total trades",    total],
            ["Win rate",        f"{win_rate:.1f}%"],
            ["Total P&L ($)",   round(total_pnl, 2)],
            ["Avg win (%)",     round(avg_win, 2)],
            ["Avg loss (%)",    round(avg_loss, 2)],
            ["Portfolio start", f"${PORTFOLIO_USD:,.0f}"],
            ["Portfolio now",   f"${PORTFOLIO_USD + total_pnl:,.2f}"],
        ]

        # Breakdown by agreement type
        for agreement in ("CONSENSUS", "SPLIT", "OPPOSED"):
            subset = [t for t in all_trades if t["agent_agreement"] == agreement]
            if subset:
                w = sum(1 for t in subset if t["pnl_pct"] >= 0)
                rows.append([f"Win rate ({agreement})", f"{w/len(subset)*100:.1f}% ({len(subset)} trades)"])

        for r in rows:
            ws3.append(r)
            ws3.cell(ws3.max_row, 1).font = Font(bold=True)

    wb.save(EXCEL_PATH)
    print(f"\n  Excel updated: {EXCEL_PATH.resolve()}")


# ── Status / History commands ───────────────────────────────────

def show_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT * FROM positions").fetchall()
    if not rows:
        print("No open positions.")
        return
    print(f"\n{'Symbol':<6} {'Dir':<6} {'Entry Date':<12} {'Entry $':>10} {'Current $':>10} {'Unreal %':>9}")
    print("-" * 60)
    for pos in rows:
        cur  = _price(pos["symbol"])
        if cur:
            mult   = 1.0 if pos["direction"] == "LONG" else -1.0
            unreal = mult * (cur - pos["entry_price"]) / pos["entry_price"] * 100
            cur_s  = f"{cur:>10,.2f}"
            unr_s  = f"{unreal:>+9.2f}%"
        else:
            cur_s, unr_s = "     N/A  ", "        —"
        print(f"{pos['symbol']:<6} {pos['direction']:<6} {pos['entry_date']:<12} "
              f"{pos['entry_price']:>10,.2f} {cur_s} {unr_s}")


def show_history(conn: sqlite3.Connection, n: int = 30) -> None:
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY exit_date DESC LIMIT ?", (n,)
    ).fetchall()
    if not rows:
        print("No completed trades yet.")
        return
    print(f"\n{'Symbol':<6} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L%':>7} {'P&L$':>8}  {'Agreement'}")
    print("-" * 70)
    for t in rows:
        sign = "+" if t["pnl_pct"] >= 0 else ""
        print(f"{t['symbol']:<6} {t['direction']:<6} {t['entry_price']:>10,.2f} "
              f"{t['exit_price']:>10,.2f} {sign}{t['pnl_pct']:>6.2f}% "
              f"{sign}${t['pnl_usd']:>7.2f}  {t['agent_agreement']}")

    # Summary
    wins     = sum(1 for t in rows if t["pnl_pct"] >= 0)
    total    = len(rows)
    total_pnl = sum(t["pnl_usd"] for t in rows)
    sign = "+" if total_pnl >= 0 else ""
    print(f"\n  {total} trades | Win rate {wins/total*100:.1f}% | Total P&L {sign}${total_pnl:.2f}")


# ── Main ────────────────────────────────────────────────────────

async def _main(args) -> None:
    conn = _db()

    if args.status:
        show_status(conn)
        return

    if args.history:
        show_history(conn)
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}")
    print(f"  ALPHA-SEEKER PAPER TRADER  —  {now}")
    print(f"{'='*60}")

    print("\n[1/3] Closing open positions...")
    closed = close_positions(conn)
    if not closed:
        print("  No positions to close.")

    print("\n[2/3] Running today's analysis...")
    opened = await open_positions(conn)

    print("\n[3/3] Updating Excel...")
    export_excel(conn)

    # Print daily summary
    all_trades = conn.execute("SELECT * FROM trades").fetchall()
    total_pnl  = sum(t["pnl_usd"] for t in all_trades)
    portfolio  = PORTFOLIO_USD + total_pnl
    sign = "+" if total_pnl >= 0 else ""
    print(f"\n{'='*60}")
    print(f"  Portfolio value : ${portfolio:>10,.2f}  ({sign}${total_pnl:.2f} total P&L)")
    print(f"  Open positions  : {conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0]}")
    print(f"  Closed today    : {len(closed)}")
    print(f"  Opened today    : {len(opened)}")
    print(f"{'='*60}\n")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",  action="store_true", help="Show open positions")
    parser.add_argument("--history", action="store_true", help="Show last 30 trades")
    args = parser.parse_args()
    asyncio.run(_main(args))
