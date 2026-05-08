"""
=============================================================
  agent.py — Scorer, Reporter & Main Entry Point
  Run: python agent.py
  Output: stock_screener_results.csv + printed leaderboard
=============================================================
"""

import logging
import pandas as pd
from datetime import datetime

from config import UNIVERSE, WEIGHTS
from fetcher import GBPConverter, StockDataFetcher
from screener import (
    screen_fundamentals,
    screen_valuation,
    screen_governance,
    screen_technical,
    screen_uk_risk,
)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────────────────────

def layer_score(layer: dict) -> float:
    """Score a single layer: % of criteria that passed, scaled 0–100."""
    if not layer:
        return 0.0
    passed = sum(1 for m in layer.values() if m.get("pass", False))
    return round((passed / len(layer)) * 100, 1)


def compute_total_score(f, v, g, t, u) -> dict:
    """Weighted composite score across all five layers."""
    fs = layer_score(f)
    vs = layer_score(v)
    gs = layer_score(g)
    ts = layer_score(t)
    us = layer_score(u)

    total = (
        fs * WEIGHTS["fundamentals"] +
        vs * WEIGHTS["valuation"]    +
        gs * WEIGHTS["governance"]   +
        ts * WEIGHTS["technical"]    +
        us * WEIGHTS["uk_risk"]
    )
    return {
        "total_score":         round(total, 1),
        "fundamentals_score":  fs,
        "valuation_score":     vs,
        "governance_score":    gs,
        "technical_score":     ts,
        "uk_risk_score":       us,
    }


# ─────────────────────────────────────────────────────────────
# REPORTER — per-stock summary to terminal
# ─────────────────────────────────────────────────────────────

def print_stock_summary(name: str, ticker: str, scores: dict, layers: dict):
    score_bar = "█" * int(scores["total_score"] / 5)
    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  {name[:40]:<40}  ║")
    print(f"  ║  {ticker:<10}  Total Score: {scores['total_score']:5.1f}/100  {score_bar:<20}║")
    print(f"  ╠══════════════════════════════════════════════╣")
    layer_labels = {
        "fundamentals": ("Fundamentals ", scores["fundamentals_score"]),
        "valuation":    ("Valuation   ", scores["valuation_score"]),
        "governance":   ("Governance  ", scores["governance_score"]),
        "technical":    ("Technical   ", scores["technical_score"]),
        "uk_risk":      ("UK Risk     ", scores["uk_risk_score"]),
    }
    for key, (label, sc) in layer_labels.items():
        bar  = "█" * int(sc / 10)
        data = layers.get(key, {})
        n    = len(data)
        p    = sum(1 for m in data.values() if m.get("pass", False))
        print(f"  ║  {label}: {sc:5.1f}  [{bar:<10}]  ({p}/{n} passed)  ║")
    print(f"  ╚══════════════════════════════════════════════╝")


def print_metric_detail(layers: dict):
    """Print pass/fail for each metric in each layer."""
    layer_names = {
        "fundamentals": "── LAYER 1: FUNDAMENTALS ──",
        "valuation":    "── LAYER 2: VALUATION ──",
        "governance":   "── LAYER 3: GOVERNANCE ──",
        "technical":    "── LAYER 4: TECHNICAL ──",
        "uk_risk":      "── LAYER 5: UK RISK ──",
    }
    for key, heading in layer_names.items():
        metrics = layers.get(key, {})
        if not metrics:
            continue
        print(f"\n     {heading}")
        for metric, info in metrics.items():
            icon  = "✅" if info.get("pass") else "❌"
            value = info.get("value", "N/A")
            note  = f"  [{info['note']}]" if info.get("note") else ""
            print(f"       {icon}  {metric:<30} {value}{note}")


def flatten_row(ticker, name, sector, country, currency, scores, layers) -> dict:
    """Build a flat dict for the output CSV."""
    row = {
        "Ticker":   ticker,
        "Company":  name,
        "Sector":   sector,
        "Country":  country,
        "Currency": currency,
        **scores,
    }
    prefixes = {"fundamentals": "F", "valuation": "V",
                "governance": "G", "technical": "T", "uk_risk": "UK"}
    for layer_key, prefix in prefixes.items():
        for metric, info in layers.get(layer_key, {}).items():
            icon = "✅" if info.get("pass") else "❌"
            row[f"{prefix}_{metric}"] = f"{info.get('value', '')} {icon}"
    return row


# ─────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────

def run_agent(
    tickers: list = None,
    output_csv: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full 5-layer screening pipeline.

    Args:
        tickers    : List of ticker strings. Defaults to UNIVERSE in config.py.
        output_csv : Path to write results CSV. Auto-named with timestamp if None.
        verbose    : If True, print detailed per-stock metric breakdown.

    Returns:
        pd.DataFrame sorted by total_score descending.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_csv or f"stock_screener_results_{timestamp}.csv"

    all_tickers = tickers or [t for region in UNIVERSE.values() for t in region]

    banner = f"""
╔══════════════════════════════════════════════════════════╗
║     GLOBAL STOCK SCREENING AGENT — UK GBP INVESTOR      ║
║     Stocks to screen: {len(all_tickers):<5}  Date: {datetime.now():%Y-%m-%d}     ║
║                                                          ║
║  Layers:  Fundamentals | Valuation | Governance          ║
║           Technical    | UK Risk (WHT, FX, ISA)         ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

    # Initialise services
    gbp     = GBPConverter()
    fetcher = StockDataFetcher(gbp)

    results = []

    for i, ticker in enumerate(all_tickers, 1):
        print(f"\n[{i}/{len(all_tickers)}] Processing {ticker} …")

        # ── Fetch ────────────────────────────────────────────
        data = fetcher.fetch(ticker)
        if data.get("error"):
            print(f"  ✗ Skipped — {data['error']}")
            continue

        name     = data.get("name",     ticker)
        sector   = data.get("sector",   "Unknown")
        country  = data.get("country",  "Unknown")
        currency = data.get("currency", "USD")

        # ── Screen ───────────────────────────────────────────
        layers = {
            "fundamentals": screen_fundamentals(data),
            "valuation":    screen_valuation(data),
            "governance":   screen_governance(data),
            "technical":    screen_technical(data),
            "uk_risk":      screen_uk_risk(data),
        }

        # ── Score ────────────────────────────────────────────
        scores = compute_total_score(
            layers["fundamentals"], layers["valuation"],
            layers["governance"],  layers["technical"],
            layers["uk_risk"],
        )

        # ── Print ────────────────────────────────────────────
        print_stock_summary(name, ticker, scores, layers)
        if verbose:
            print_metric_detail(layers)

        # ── Collect ──────────────────────────────────────────
        results.append(flatten_row(ticker, name, sector, country, currency, scores, layers))

    # ── Sort & Save ──────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("total_score", ascending=False).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    # ── Final Leaderboard ────────────────────────────────────
    print("\n\n" + "═" * 80)
    print(f"  🏆  SCREENING COMPLETE — {len(results)} stocks analysed")
    print("═" * 80)

    cols = ["Ticker", "Company", "Sector", "Country", "total_score",
            "fundamentals_score", "valuation_score", "governance_score",
            "technical_score", "uk_risk_score"]
    top = df[cols].head(15)

    print(f"\n{'Rank':<5} {'Ticker':<10} {'Company':<30} {'Sector':<20} {'Total':>6} "
          f"{'Fund':>6} {'Val':>6} {'Gov':>6} {'Tech':>6} {'UK':>6}")
    print("─" * 100)
    for rank, row in enumerate(top.itertuples(), 1):
        print(f"{rank:<5} {row.Ticker:<10} {row.Company[:28]:<30} {row.Sector[:18]:<20} "
              f"{row.total_score:>6.1f} {row.fundamentals_score:>6.1f} "
              f"{row.valuation_score:>6.1f} {row.governance_score:>6.1f} "
              f"{row.technical_score:>6.1f} {row.uk_risk_score:>6.1f}")

    print(f"\n📁  Full results saved to: {output_csv}")
    print(f"\n💡  Tips:")
    print(f"    • Submit W-8BEN form to your broker for US stocks (reduces WHT to 15%)")
    print(f"    • Hold top picks in your Trading 212 Stocks & Shares ISA (£20k/yr)")
    print(f"    • UK stamp duty (0.5%) applies only on UK-listed share purchases")
    print(f"    • Review FX risk — consider GBP-hedged ETFs for heavy FX exposure\n")

    return df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option A: Run full universe from config.py ────────────
    df = run_agent()

    # ── Option B: Run a custom shortlist ─────────────────────
    # df = run_agent(["AAPL", "MSFT", "ULVR.L", "ASML.AS", "NVO"])

    # ── Option C: Quick single-stock check ───────────────────
    # df = run_agent(["MSFT"], verbose=True)
