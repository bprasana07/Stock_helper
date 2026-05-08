"""
=============================================================
  GLOBAL STOCK SCREENING AGENT — UK GBP INVESTOR
  config.py — All thresholds, weights, universe, and constants
=============================================================
"""

# ── API KEYS ─────────────────────────────────────────────────
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# API Keys
FMP_API_KEY = os.getenv("FMP_API_KEY")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in environment")

if not EXCHANGE_RATE_API_KEY:
    raise ValueError("EXCHANGE_RATE_API_KEY not found in environment")

# ── STOCK UNIVERSE ───────────────────────────────────────────
# Add/remove tickers per region. UK suffix = .L, Netherlands = .AS, Germany = .DE
UNIVERSE = {
    "US_LARGE_CAP": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JNJ", "PG", "KO", "V", "MA", "UNH", "HD",
        "AVGO", "LLY", "COST", "MCD", "AXP", "BRK-B"
    ],
    "UK": [
        "ULVR.L", "AZN.L", "LSEG.L", "REL.L", "EXPN.L",
        "DGE.L", "RKT.L", "ABF.L", "BATS.L", "GSK.L"
    ],
    "EUROPE": [
        "ASML.AS", "SAP.DE", "NESN.SW", "NOVO-B.CO",
        "MC.PA", "SIE.DE", "LVMH.PA"
    ],
    "GLOBAL_QUALITY": [
        "NVO", "TSM", "IDEXY", "SONY", "TM"
    ],
}

# ── SCORING WEIGHTS (must sum to 1.0) ────────────────────────
WEIGHTS = {
    "fundamentals": 0.30,   # Core financial health
    "valuation":    0.20,   # Not overpaying
    "governance":   0.20,   # Management alignment
    "technical":    0.15,   # Price trend & momentum
    "uk_risk":      0.15,   # UK-specific: FX, WHT, ISA, stamp duty
}

# ── SCREENING THRESHOLDS ─────────────────────────────────────
THRESHOLDS = {
    # Fundamentals
    "roe_min":                  15.0,   # Return on Equity %
    "roce_min":                 15.0,   # Return on Capital Employed %
    "roic_min":                 12.0,   # Return on Invested Capital %
    "de_max":                   0.80,   # Debt-to-Equity ratio
    "revenue_cagr_5y_min":      10.0,   # Revenue 5Y CAGR %
    "profit_cagr_5y_min":       10.0,   # Net Profit 5Y CAGR %
    "fcf_positive_years_min":   4,      # Positive FCF in at least 4 of last 5 years
    "ocf_ni_ratio_min":         1.0,    # Operating CF / Net Income (earnings quality)
    "interest_coverage_min":    5.0,    # EBIT / Interest Expense
    "gross_margin_expanding":   True,   # Gross margin trend must be upward

    # Valuation
    "pe_max":                   45.0,   # Trailing P/E ceiling (broad filter)
    "forward_pe_max":           35.0,   # Forward P/E ceiling
    "pfcf_max":                 25.0,   # Price / Free Cash Flow
    "peg_max":                  1.5,    # PEG ratio
    "ev_ebitda_max":            20.0,   # EV / EBITDA
    "pb_max":                   10.0,   # Price to Book
    "dcf_margin_of_safety_min": 10.0,   # % discount to DCF intrinsic value

    # Governance
    "insider_ownership_min":    5.0,    # % of shares held by insiders
    "inst_ownership_min":       40.0,   # % institutional (floor — avoid orphan stocks)
    "inst_ownership_max":       85.0,   # % institutional (cap — avoid over-crowding)
    "short_ratio_max":          5.0,    # Days to cover short interest
    "net_insider_buys_min":     0,      # Net buy transactions (buys minus sells)
    "buyback_years_min":        3,      # Years with buyback activity in last 5

    # Technical
    "price_cagr_10y_min":       10.0,   # 10Y price CAGR in local currency
    "price_cagr_5y_min":        8.0,    # 5Y price CAGR
    "rsi_max":                  70,     # RSI — avoid overbought entries
    "beta_max":                 1.3,    # Max beta (volatility vs market)
    "pct_of_52w_high_min":      75.0,   # Must be within 25% of 52-week high

    # UK Risk
    "wht_max":                  15.0,   # Max withholding tax rate %
    "gbp_cagr_10y_min":         10.0,   # GBP-adjusted 10Y CAGR estimate
}

# ── WITHHOLDING TAX RATES (on dividends, by country) ─────────
# UK residents can claim treaty rate; W-8BEN needed for US stocks
WHT_RATES = {
    "United States":        15.0,   # Treaty rate with W-8BEN
    "United Kingdom":        0.0,   # No WHT for UK residents
    "Germany":              26.375,
    "France":               12.8,   # Treaty rate
    "Switzerland":          35.0,   # Reclaim available but complex
    "Denmark":              27.0,
    "Sweden":               30.0,
    "Netherlands":          15.0,   # Treaty rate
    "Japan":                15.315,
    "Australia":            30.0,
    "Ireland":              25.0,
    "Canada":               25.0,
    "Singapore":             0.0,
    "Taiwan":               21.0,
    "China":                10.0,
    "South Korea":          22.0,
}

# FX drag estimates vs GBP (rough annual %, based on 10Y trends)
FX_DRAG_VS_GBP = {
    "USD":  -1.5,   # USD has strengthened vs GBP historically (negative drag = tailwind)
    "EUR":  -0.5,
    "CHF":  -1.0,
    "JPY":   2.0,   # JPY has weakened vs GBP (positive = headwind)
    "AUD":   1.0,
    "CAD":   0.5,
    "DKK":  -0.5,
    "SEK":   0.5,
    "TWD":  -0.5,
}

# Low-risk countries for investment (rule of law, political stability)
LOW_RISK_COUNTRIES = [
    "United States", "United Kingdom", "Germany", "Switzerland",
    "France", "Netherlands", "Japan", "Australia", "Canada",
    "Sweden", "Denmark", "Norway", "Singapore", "Ireland",
]
