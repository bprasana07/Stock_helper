"""
═══════════════════════════════════════════════════════════
  GLOBAL STOCK SCREENER — Single File App
  Run:  python stock_screener_app.py
  Open: http://localhost:5000

  Users can type company names (Apple, Microsoft, Unilever)
  OR ticker symbols (AAPL, MSFT, ULVR.L) — both work.
═══════════════════════════════════════════════════════════
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import requests as req_lib
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# FALLBACK NAME -> TICKER LOOKUP
# Used when Yahoo Finance search is unavailable
# ─────────────────────────────────────────────

FALLBACK_LOOKUP = {
    # US Tech
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA",
    "netflix": "NFLX", "salesforce": "CRM", "adobe": "ADBE",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "broadcom": "AVGO", "oracle": "ORCL", "servicenow": "NOW",
    "snowflake": "SNOW", "palantir": "PLTR", "uber": "UBER",
    "airbnb": "ABNB", "shopify": "SHOP", "spotify": "SPOT",
    "zoom": "ZM", "paypal": "PYPL", "block": "SQ",
    "booking": "BKNG", "priceline": "BKNG", "expedia": "EXPE",
    "ebay": "EBAY", "doordash": "DASH",
    # US Finance
    "visa": "V", "mastercard": "MA", "american express": "AXP",
    "amex": "AXP", "jpmorgan": "JPM", "jp morgan": "JPM",
    "bank of america": "BAC", "goldman sachs": "GS",
    "morgan stanley": "MS", "wells fargo": "WFC",
    "citigroup": "C", "citi": "C", "blackrock": "BLK",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    # US Healthcare
    "johnson and johnson": "JNJ", "johnson & johnson": "JNJ",
    "pfizer": "PFE", "merck": "MRK", "abbvie": "ABBV",
    "eli lilly": "LLY", "lilly": "LLY", "unitedhealth": "UNH",
    "united health": "UNH", "cvs": "CVS", "medtronic": "MDT",
    "novo nordisk": "NVO",
    # US Consumer
    "procter and gamble": "PG", "procter & gamble": "PG",
    "coca cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "mcdonalds": "MCD", "mcdonald's": "MCD", "starbucks": "SBUX",
    "nike": "NKE", "walmart": "WMT", "costco": "COST",
    "home depot": "HD", "target": "TGT", "colgate": "CL",
    # US Energy/Industrial
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "caterpillar": "CAT", "boeing": "BA", "lockheed": "LMT",
    "3m": "MMM", "honeywell": "HON", "ups": "UPS", "fedex": "FDX",
    # UK
    "unilever": "ULVR.L", "astrazeneca": "AZN.L", "hsbc": "HSBA.L",
    "lseg": "LSEG.L", "london stock exchange": "LSEG.L",
    "relx": "REL.L", "experian": "EXPN.L", "diageo": "DGE.L",
    "reckitt": "RKT.L", "reckitt benckiser": "RKT.L",
    "associated british foods": "ABF.L",
    "british american tobacco": "BATS.L", "bat": "BATS.L",
    "gsk": "GSK.L", "glaxosmithkline": "GSK.L", "glaxo": "GSK.L",
    "rolls royce": "RR.L", "bp": "BP.L", "shell": "SHEL.L",
    "rio tinto": "RIO.L", "bhp": "BHP.L", "barclays": "BARC.L",
    "lloyds": "LLOY.L", "natwest": "NWG.L", "vodafone": "VOD.L",
    "bt": "BT-A.L", "bt group": "BT-A.L",
    "marks and spencer": "MKS.L", "marks & spencer": "MKS.L",
    "tesco": "TSCO.L", "sainsbury": "SBRY.L", "next": "NXT.L",
    "burberry": "BRBY.L", "prudential": "PRU.L", "aviva": "AV.L",
    "legal and general": "LGEN.L", "legal & general": "LGEN.L",
    "standard chartered": "STAN.L", "haleon": "HLN.L",
    "smith and nephew": "SN.L", "smith & nephew": "SN.L",
    "ihg": "IHG.L", "intercontinental hotels": "IHG.L",
    "whitbread": "WTB.L", "compass": "CPG.L", "rentokil": "RTO.L",
    # Europe
    "asml": "ASML.AS", "sap": "SAP.DE", "nestle": "NESN.SW",
    "lvmh": "MC.PA", "louis vuitton": "MC.PA",
    "loreal": "OR.PA", "l'oreal": "OR.PA",
    "siemens": "SIE.DE", "volkswagen": "VOW3.DE",
    "bmw": "BMW.DE", "mercedes": "MBG.DE", "mercedes benz": "MBG.DE",
    "adidas": "ADS.DE", "allianz": "ALV.DE", "basf": "BAS.DE",
    "bayer": "BAYN.DE", "airbus": "AIR.PA",
    "totalenergies": "TTE.PA", "total": "TTE.PA",
    "bnp paribas": "BNP.PA", "hermes": "RMS.PA",
    "kering": "KER.PA", "gucci": "KER.PA",
    "stellantis": "STLA", "ferrari": "RACE",
    "ahold": "AD.AS", "ing": "INGA.AS", "philips": "PHIA.AS",
    "heineken": "HEIA.AS", "abb": "ABBN.SW", "novartis": "NOVN.SW",
    "roche": "ROG.SW", "ubs": "UBSG.SW", "richemont": "CFR.SW",
    # Asia/Global
    "samsung": "005930.KS", "toyota": "TM", "sony": "SONY",
    "taiwan semiconductor": "TSM", "tsmc": "TSM",
    "alibaba": "BABA", "tencent": "TCEHY", "baidu": "BIDU",
    "softbank": "SFTBY", "honda": "HMC", "infosys": "INFY",
}


# ─────────────────────────────────────────────
# NAME -> TICKER RESOLUTION
# ─────────────────────────────────────────────

def resolve_ticker(query: str):
    """
    Given a company name or ticker, return (ticker, resolved_name).
    1. Try Yahoo Finance search API (live)
    2. Fall back to FALLBACK_LOOKUP
    3. Use query itself as ticker
    """
    q_lower = query.strip().lower()

    # Step 1: Yahoo Finance live search
    try:
        resp = req_lib.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0, "listsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        if resp.ok:
            quotes = [q for q in resp.json().get("quotes", [])
                      if q.get("quoteType") in ("EQUITY", "ETF")]
            if quotes:
                best = quotes[0]
                return (best["symbol"],
                        best.get("shortname") or best.get("longname") or best["symbol"])
    except Exception as e:
        logger.warning(f"Yahoo search unavailable: {e}")

    # Step 2: Local fallback — exact match
    if q_lower in FALLBACK_LOOKUP:
        return FALLBACK_LOOKUP[q_lower], query.title()

    # Step 2b: Partial match
    for name, ticker in FALLBACK_LOOKUP.items():
        if q_lower in name or name.startswith(q_lower):
            return ticker, name.title()

    # Step 3: Treat as ticker
    return query.strip().upper(), query.strip().upper()


def search_suggestions(query: str):
    """Return up to 8 autocomplete suggestions."""
    if len(query) < 2:
        return []

    # Try Yahoo Finance live search
    try:
        resp = req_lib.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0, "listsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        if resp.ok:
            return [
                {
                    "symbol":   q.get("symbol", ""),
                    "name":     q.get("shortname") or q.get("longname") or q.get("symbol", ""),
                    "exchange": q.get("exchDisp") or q.get("exchange", ""),
                    "type":     q.get("quoteType", ""),
                }
                for q in resp.json().get("quotes", [])
                if q.get("quoteType") in ("EQUITY", "ETF")
            ][:8]
    except Exception:
        pass

    # Fallback: local dictionary search
    q = query.lower()
    results = []
    for name, ticker in FALLBACK_LOOKUP.items():
        if name.startswith(q) or q in name:
            results.append({
                "symbol":   ticker,
                "name":     name.title(),
                "exchange": "LSE" if ticker.endswith(".L") else
                            "AMS" if ticker.endswith(".AS") else
                            "XETRA" if ticker.endswith(".DE") else "NYSE/NASDAQ",
                "type":     "EQUITY",
            })
        if len(results) >= 8:
            break
    return results


# ─────────────────────────────────────────────
# SCREENER HELPERS
# ─────────────────────────────────────────────

def safe_cagr(start, end, years):
    try:
        if start <= 0 or end <= 0 or years <= 0:
            return None
        return round(((end / start) ** (1.0 / years) - 1) * 100, 2)
    except Exception:
        return None

def get_row(df, *keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None

def row_vals(df, n, *keys):
    row = get_row(df, *keys)
    if row is None:
        return []
    return [float(v) for v in row.iloc[:n] if pd.notna(v)]

def is_upward(vals):
    if len(vals) < 2:
        return False
    chron = list(reversed(vals))
    slope = float(np.polyfit(np.arange(len(chron)), chron, 1)[0])
    return slope > 0

def criterion(name, category, passed, value, threshold, note=""):
    return {"name": name, "category": category, "passed": passed,
            "value": value, "threshold": threshold, "note": note}


# ─────────────────────────────────────────────
# CORE SCREENER
# ─────────────────────────────────────────────

def screen_stock(ticker):
    yf_ticker = yf.Ticker(ticker)
    info      = yf_ticker.info
    h10       = yf_ticker.history(period="10y", auto_adjust=True)
    income    = yf_ticker.financials
    balance   = yf_ticker.balance_sheet
    cashflow  = yf_ticker.cashflow

    name     = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency", "USD")
    sector   = info.get("sector",   "")
    country  = info.get("country",  "")
    exchange = info.get("exchange", "")
    price    = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    criteria = []

    # 1 - 10-Year Price Trend
    try:
        prices = h10["Close"].dropna() if not h10.empty else pd.Series(dtype=float)
        if len(prices) >= 252 * 7:
            yrs  = len(prices) / 252
            cagr = safe_cagr(float(prices.iloc[0]), float(prices.iloc[-1]), yrs)
            criteria.append(criterion(
                "10-Year Price Trend", "Technical",
                cagr is not None and cagr >= 10.0,
                f"{cagr:.1f}% p.a." if cagr else "N/A",
                "> 10% p.a. CAGR",
                "Proven long-term compounding track record"
            ))
        else:
            criteria.append(criterion("10-Year Price Trend", "Technical", False,
                "< 10 years of data", "> 10% p.a. CAGR", "Insufficient price history"))
    except Exception as e:
        criteria.append(criterion("10-Year Price Trend", "Technical", False,
            "Error", "> 10% p.a.", str(e)))

    # 2 - 5-Year Revenue Growth
    try:
        rev = row_vals(income, 5, "Total Revenue") if not income.empty else []
        if len(rev) >= 2:
            chron = list(reversed(rev))
            cagr  = safe_cagr(chron[0], chron[-1], len(chron) - 1)
            trend = is_upward(chron)
            criteria.append(criterion(
                "5-Year Revenue Growth", "Fundamentals",
                cagr is not None and cagr >= 10.0 and trend,
                f"{cagr:.1f}% p.a." if cagr else "N/A",
                "> 10% p.a. + upward trend",
                f"Trend: {'Upward' if trend else 'Declining'}"
            ))
        else:
            criteria.append(criterion("5-Year Revenue Growth", "Fundamentals",
                False, "No data", "> 10% p.a."))
    except Exception as e:
        criteria.append(criterion("5-Year Revenue Growth", "Fundamentals",
            False, "Error", "> 10% p.a.", str(e)))

    # 3 - 5-Year Profit Growth
    try:
        ni = row_vals(income, 5, "Net Income") if not income.empty else []
        if len(ni) >= 2:
            chron   = list(reversed(ni))
            all_pos = all(v > 0 for v in chron)
            cagr    = safe_cagr(chron[0], chron[-1], len(chron) - 1) if all_pos else None
            trend   = is_upward(chron)
            criteria.append(criterion(
                "5-Year Profit Growth", "Fundamentals",
                cagr is not None and cagr >= 10.0 and trend,
                f"{cagr:.1f}% p.a." if cagr else ("Loss-making period" if not all_pos else "N/A"),
                "> 10% p.a. + upward trend",
                f"Profitable throughout: {'Yes' if all_pos else 'No'} | Trend: {'Up' if trend else 'Down'}"
            ))
        else:
            criteria.append(criterion("5-Year Profit Growth", "Fundamentals",
                False, "No data", "> 10% p.a."))
    except Exception as e:
        criteria.append(criterion("5-Year Profit Growth", "Fundamentals",
            False, "Error", "> 10% p.a.", str(e)))

    # 4 - ROE
    try:
        roe = info.get("returnOnEquity")
        if roe is not None:
            pct = round(roe * 100, 2)
            criteria.append(criterion(
                "Return on Equity (ROE)", "Fundamentals", pct >= 15.0,
                f"{pct}%", "> 15%",
                "Profitability per pound of shareholder equity"
            ))
        else:
            criteria.append(criterion("Return on Equity (ROE)", "Fundamentals",
                False, "N/A", "> 15%"))
    except Exception as e:
        criteria.append(criterion("Return on Equity (ROE)", "Fundamentals",
            False, "Error", "> 15%", str(e)))

    # 5 - ROCE
    try:
        ebit = row_vals(income,  1, "EBIT", "Operating Income") if not income.empty  else []
        ta   = row_vals(balance, 1, "Total Assets")              if not balance.empty else []
        cl   = row_vals(balance, 1, "Current Liabilities",
                        "Total Current Liabilities Net Minority Interest") if not balance.empty else []
        if ebit and ta and cl:
            ce   = ta[0] - cl[0]
            roce = round((ebit[0] / ce) * 100, 2) if ce > 0 else None
            criteria.append(criterion(
                "Return on Capital Employed (ROCE)", "Fundamentals",
                roce is not None and roce >= 15.0,
                f"{roce}%" if roce else "N/A", "> 15%",
                "EBIT / (Total Assets - Current Liabilities)"
            ))
        else:
            criteria.append(criterion("Return on Capital Employed (ROCE)", "Fundamentals",
                False, "No data", "> 15%"))
    except Exception as e:
        criteria.append(criterion("Return on Capital Employed (ROCE)", "Fundamentals",
            False, "Error", "> 15%", str(e)))

    # 6 - Debt-to-Equity
    try:
        de = info.get("debtToEquity")
        if de is not None:
            de_val = round(de / 100, 2)
            criteria.append(criterion(
                "Debt-to-Equity Ratio", "Fundamentals", de_val <= 0.8,
                f"{de_val}x", "< 0.8x", "Manageable financial leverage"
            ))
        else:
            criteria.append(criterion("Debt-to-Equity Ratio", "Fundamentals",
                False, "N/A", "< 0.8x"))
    except Exception as e:
        criteria.append(criterion("Debt-to-Equity Ratio", "Fundamentals",
            False, "Error", "< 0.8x", str(e)))

    # 7 - Free Cash Flow
    ocf_vals, capex_vals = [], []
    try:
        if not cashflow.empty:
            ocf_vals   = row_vals(cashflow, 5, "Operating Cash Flow",
                                  "Cash Flow From Continuing Operating Activities")
            capex_vals = row_vals(cashflow, 5, "Capital Expenditure",
                                  "Purchase Of Property Plant And Equipment")
        if ocf_vals and capex_vals:
            n   = min(len(ocf_vals), len(capex_vals))
            fcf = [ocf_vals[i] + capex_vals[i] for i in range(n)]
            pos = sum(1 for f in fcf if f > 0)
            criteria.append(criterion(
                "Free Cash Flow", "Fundamentals", pos >= 4,
                f"{pos}/{n} positive years", ">= 4 of 5 years positive",
                "Profit backed by real cash generation"
            ))
        else:
            criteria.append(criterion("Free Cash Flow", "Fundamentals",
                False, "No data", ">= 4/5 years positive"))
    except Exception as e:
        criteria.append(criterion("Free Cash Flow", "Fundamentals",
            False, "Error", ">= 4/5 years positive", str(e)))

    # 8 - Management Ownership
    try:
        ins = info.get("heldPercentInsiders")
        if ins is not None:
            pct = round(ins * 100, 2)
            criteria.append(criterion(
                "Management Ownership", "Governance", pct >= 5.0,
                f"{pct}%", "> 5% insider ownership",
                "Executives & directors with skin in the game"
            ))
        else:
            criteria.append(criterion("Management Ownership", "Governance",
                False, "N/A", "> 5%"))
    except Exception as e:
        criteria.append(criterion("Management Ownership", "Governance",
            False, "Error", "> 5%", str(e)))

    # 9 - Interest Coverage
    try:
        ebit_ic  = row_vals(income, 1, "EBIT", "Operating Income") if not income.empty else []
        interest = row_vals(income, 1, "Interest Expense", "Net Interest Income") if not income.empty else []
        if ebit_ic and interest and interest[0] != 0:
            ic = round(abs(ebit_ic[0] / interest[0]), 2)
            criteria.append(criterion(
                "Interest Coverage", "Fundamentals", ic >= 5.0,
                f"{ic}x", "> 5x (EBIT / Interest)",
                "Ability to comfortably service debt from operations"
            ))
        else:
            criteria.append(criterion(
                "Interest Coverage", "Fundamentals", True,
                "Negligible / no debt", "> 5x",
                "No material interest expense detected"
            ))
    except Exception as e:
        criteria.append(criterion("Interest Coverage", "Fundamentals",
            False, "Error", "> 5x", str(e)))

    # 10 - Valuation
    try:
        peg      = info.get("pegRatio")
        mktcap   = info.get("marketCap")
        pfcf_val = None
        if mktcap and ocf_vals and capex_vals:
            latest_fcf = ocf_vals[0] + capex_vals[0]
            if latest_fcf > 0:
                pfcf_val = round(mktcap / latest_fcf, 1)

        peg_ok  = peg      is not None and 0 < peg      <= 1.5
        pfcf_ok = pfcf_val is not None and pfcf_val <= 25.0

        parts = []
        if peg      is not None: parts.append(f"PEG {peg:.2f}")
        if pfcf_val is not None: parts.append(f"P/FCF {pfcf_val:.1f}x")

        criteria.append(criterion(
            "Valuation", "Valuation", peg_ok or pfcf_ok,
            " | ".join(parts) if parts else "N/A",
            "PEG < 1.5  OR  P/FCF < 25x",
            "At least one valuation metric must be reasonable"
        ))
    except Exception as e:
        criteria.append(criterion("Valuation", "Valuation",
            False, "Error", "PEG < 1.5 or P/FCF < 25x", str(e)))

    # Verdict
    passed_count = sum(1 for c in criteria if c["passed"])
    total        = len(criteria)
    score_pct    = round((passed_count / total) * 100)

    if   score_pct >= 80: verdict, vc = "Strong Buy Candidate",   "green"
    elif score_pct >= 60: verdict, vc = "Potential Opportunity",  "amber"
    elif score_pct >= 40: verdict, vc = "Needs Further Research", "orange"
    else:                 verdict, vc = "Does Not Meet Criteria", "red"

    return {
        "ticker": ticker, "name": name, "sector": sector,
        "country": country, "exchange": exchange,
        "currency": currency, "price": price,
        "criteria": criteria, "passed": passed_count,
        "total": total, "score_pct": score_pct,
        "verdict": verdict, "verdict_color": vc,
    }


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(search_suggestions(q))
    except Exception:
        return jsonify([])


@app.route("/api/screen")
def api_screen():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please enter a company name or ticker symbol"}), 400
    try:
        ticker, resolved_name = resolve_ticker(query)
        logger.info(f"Resolved '{query}' -> {ticker} ({resolved_name})")
        result = screen_stock(ticker)
        result["resolved_from"] = query
        result["resolved_name"] = resolved_name
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Could not analyse '{query}': {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


# ─────────────────────────────────────────────
# FRONTEND HTML
# ─────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Stock Screener</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#06070f;
  --s1:#0c0e1e;
  --s2:#10132a;
  --s3:#141836;
  --b1:#1e2448;
  --b2:#252d5a;
  --text:#e2e8ff;
  --muted:#6272a4;
  --dim:#2a3260;

  --p1:#7c3aed;
  --p2:#6025c0;
  --c1:#06b6d4;
  --c2:#0891b2;

  --green:#10b981;
  --green-bg:rgba(16,185,129,.1);
  --green-border:rgba(16,185,129,.25);
  --red:#f43f5e;
  --red-bg:rgba(244,63,94,.1);
  --red-border:rgba(244,63,94,.25);
  --amber:#f59e0b;
  --amber-bg:rgba(245,158,11,.1);

  --grad:linear-gradient(135deg,var(--p1),var(--c1));
  --grad2:linear-gradient(135deg,var(--p2),var(--c2));

  --sans:'Space Grotesk',sans-serif;
  --mono:'Space Mono',monospace;
}

html{scroll-behavior:smooth}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* Ambient glow background */
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 600px 400px at 10% 20%,rgba(124,58,237,.12),transparent),
    radial-gradient(ellipse 500px 300px at 90% 70%,rgba(6,182,212,.10),transparent),
    radial-gradient(ellipse 400px 400px at 50% 50%,rgba(96,37,192,.06),transparent);
}

/* Grid lines */
body::after{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(124,58,237,.04) 1px,transparent 1px),
    linear-gradient(90deg,rgba(6,182,212,.04) 1px,transparent 1px);
  background-size:50px 50px;
}

.shell{position:relative;z-index:1;max-width:940px;margin:0 auto;padding:48px 24px 100px}

/* ── HEADER ── */
.header{text-align:center;margin-bottom:48px}

.logo-row{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:20px}
.logo-icon{
  width:44px;height:44px;border-radius:12px;
  background:var(--grad);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;box-shadow:0 0 24px rgba(124,58,237,.4)
}
.logo-text{font-family:var(--mono);font-size:13px;letter-spacing:3px;
  text-transform:uppercase;background:var(--grad);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}

.header h1{
  font-size:clamp(28px,5vw,52px);font-weight:700;letter-spacing:-1.5px;
  line-height:1.05;margin-bottom:14px;
  background:linear-gradient(135deg,#fff 0%,#c4b5fd 50%,var(--c1) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.header p{color:var(--muted);font-size:15px;font-weight:400;max-width:480px;margin:0 auto}

/* ── SEARCH ── */
.search-outer{position:relative;margin-bottom:14px}

.search-wrap{
  display:flex;gap:0;
  background:var(--s1);
  border:1px solid var(--b1);
  border-radius:14px;
  padding:6px;
  transition:border-color .25s, box-shadow .25s;
  box-shadow:0 4px 24px rgba(0,0,0,.3)
}
.search-wrap:focus-within{
  border-color:var(--p1);
  box-shadow:0 0 0 3px rgba(124,58,237,.15), 0 4px 24px rgba(0,0,0,.3)
}

.search-icon{
  display:flex;align-items:center;padding:0 14px 0 16px;color:var(--muted);font-size:18px
}

.search-wrap input{
  flex:1;background:none;border:none;outline:none;
  font-family:var(--sans);font-size:16px;font-weight:500;color:#fff;
  padding:12px 4px;
}
.search-wrap input::placeholder{color:var(--dim);font-weight:400}

.btn-screen{
  background:var(--grad);color:#fff;
  font-family:var(--mono);font-size:11px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;
  border:none;border-radius:10px;padding:14px 26px;
  cursor:pointer;transition:opacity .2s,transform .1s,box-shadow .2s;
  white-space:nowrap;flex-shrink:0;
  box-shadow:0 4px 16px rgba(124,58,237,.35)
}
.btn-screen:hover{opacity:.9;box-shadow:0 6px 24px rgba(124,58,237,.5)}
.btn-screen:active{transform:scale(.97)}
.btn-screen:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}

/* ── AUTOCOMPLETE ── */
.autocomplete{
  position:absolute;top:calc(100% + 6px);left:0;right:0;z-index:200;
  background:var(--s1);border:1px solid var(--b2);border-radius:12px;
  overflow:hidden;display:none;
  box-shadow:0 16px 48px rgba(0,0,0,.6),0 0 0 1px rgba(124,58,237,.1)
}
.autocomplete.open{display:block}

.ac-item{
  display:flex;align-items:center;gap:14px;padding:12px 18px;
  cursor:pointer;transition:background .12s;border-bottom:1px solid var(--b1)
}
.ac-item:last-child{border-bottom:none}
.ac-item:hover,.ac-item.hi{background:var(--s2)}

.ac-sym{
  font-family:var(--mono);font-size:12px;font-weight:700;
  min-width:72px;
  background:var(--grad);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text
}
.ac-name{font-size:13px;color:var(--text);flex:1;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;font-weight:500}
.ac-ex{font-family:var(--mono);font-size:10px;color:var(--muted);
  background:var(--s3);padding:2px 7px;border-radius:4px}
.ac-msg{padding:16px 18px;font-size:13px;color:var(--muted);font-family:var(--mono)}

/* ── EXAMPLES ── */
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:40px;align-items:center}
.ex-label{font-family:var(--mono);font-size:10px;color:var(--dim);
  letter-spacing:1px;text-transform:uppercase}
.ex-chip{
  font-size:12px;font-weight:500;color:var(--muted);
  border:1px solid var(--b1);padding:6px 14px;border-radius:20px;
  cursor:pointer;transition:all .18s;background:var(--s1)
}
.ex-chip:hover{
  color:#fff;border-color:var(--p1);
  background:rgba(124,58,237,.15);
  box-shadow:0 0 12px rgba(124,58,237,.2)
}

/* ── LOADER ── */
.loader{display:none;text-align:center;padding:64px 0}
.loader.on{display:block}
.loader-ring{
  width:48px;height:48px;margin:0 auto 16px;
  border:2px solid var(--b2);border-top-color:var(--p1);
  border-radius:50%;animation:spin .7s linear infinite
}
@keyframes spin{to{transform:rotate(360deg)}}
.loader p{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:3px}

/* ── ERROR ── */
.err{
  display:none;
  background:var(--red-bg);border:1px solid var(--red-border);border-radius:10px;
  padding:16px 20px;font-family:var(--mono);font-size:13px;color:var(--red)
}
.err.on{display:block}

/* ── RESULTS ── */
#results{display:none}
#results.on{display:block;animation:fu .45s cubic-bezier(.4,0,.2,1)}
@keyframes fu{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

/* Resolution banner */
.rbanner{
  display:none;
  border-radius:10px;padding:11px 18px;margin-bottom:16px;font-size:13px;
  font-family:var(--mono);font-weight:400;
  background:linear-gradient(135deg,rgba(124,58,237,.12),rgba(6,182,212,.08));
  border:1px solid rgba(124,58,237,.25);color:#c4b5fd;
}
.rbanner.on{display:flex;align-items:center;gap:10px}
.rbanner-arrow{font-size:16px}

/* ── STOCK HEADER ── */
.shead{
  background:var(--s1);
  border:1px solid var(--b1);
  border-radius:16px;padding:28px 32px;margin-bottom:24px;
  display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;
  position:relative;overflow:hidden;
  box-shadow:0 8px 32px rgba(0,0,0,.3)
}
.shead::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(124,58,237,.12),transparent 70%);
  pointer-events:none
}

.smeta{flex:1;min-width:200px}
.sticker{
  font-family:var(--mono);font-size:30px;font-weight:700;
  letter-spacing:2px;
  background:var(--grad);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:4px
}
.sname{font-size:16px;color:#fff;font-weight:600;margin-bottom:12px}
.stags{display:flex;gap:8px;flex-wrap:wrap}
.tag{
  font-family:var(--mono);font-size:10px;letter-spacing:.5px;
  color:var(--muted);border:1px solid var(--b2);
  padding:4px 10px;border-radius:6px;text-transform:uppercase;
  background:var(--s2)
}
.tag.price{
  background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(6,182,212,.1));
  border-color:rgba(124,58,237,.3);color:#c4b5fd
}

/* ── SCORE RING ── */
.srwrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.sring{position:relative;width:100px;height:100px}
.sring svg{transform:rotate(-90deg)}
.sring circle{fill:none;stroke-width:7}
.rbg{stroke:var(--b1)}
.rfill{
  stroke-linecap:round;stroke-dasharray:270;stroke-dashoffset:270;
  transition:stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1)
}
.rfill.green{stroke:url(#gGreen)}.rfill.amber{stroke:url(#gAmber)}
.rfill.orange{stroke:url(#gOrange)}.rfill.red{stroke:url(#gRed)}

.rinner{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;font-family:var(--mono)
}
.rscore{font-size:22px;font-weight:700;color:#fff;line-height:1}
.rsub{font-size:9px;color:var(--muted);margin-top:3px;letter-spacing:1px}

.vpill{
  font-family:var(--mono);font-size:10px;letter-spacing:1px;
  text-transform:uppercase;text-align:center;max-width:110px;
  line-height:1.5;padding:5px 12px;border-radius:20px;
}
.vpill.green{color:var(--green);background:var(--green-bg);border:1px solid var(--green-border)}
.vpill.amber{color:var(--amber);background:var(--amber-bg);border:1px solid rgba(245,158,11,.25)}
.vpill.orange{color:#fb923c;background:rgba(251,146,60,.1);border:1px solid rgba(251,146,60,.25)}
.vpill.red{color:var(--red);background:var(--red-bg);border:1px solid var(--red-border)}

/* ── CRITERIA GRID ── */
.sec-header{
  display:flex;align-items:center;gap:12px;margin-bottom:16px
}
.sec-label{
  font-family:var(--mono);font-size:10px;letter-spacing:3px;
  color:var(--muted);text-transform:uppercase
}
.sec-line{flex:1;height:1px;background:linear-gradient(90deg,var(--b1),transparent)}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}

.card{
  background:var(--s1);border:1px solid var(--b1);
  border-radius:12px;padding:18px 18px 16px;
  opacity:0;transform:translateY(10px);
  transition:opacity .35s,transform .35s,border-color .2s,box-shadow .2s;
  position:relative;overflow:hidden;cursor:default
}
.card:hover{border-color:var(--b2);box-shadow:0 8px 24px rgba(0,0,0,.25)}
.card.vis{opacity:1;transform:translateY(0)}

.card.p{border-left:3px solid var(--green)}
.card.f{border-left:3px solid var(--red)}
.card.p::after{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,var(--green),transparent);opacity:.5
}

.ctop{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:12px}
.cname{font-size:13px;font-weight:600;color:#fff;line-height:1.3}

.badge{
  font-family:var(--mono);font-size:9px;font-weight:700;letter-spacing:.5px;
  padding:3px 8px;border-radius:6px;flex-shrink:0;text-transform:uppercase
}
.bp{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.bf{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border)}

.cval{font-family:var(--mono);font-size:18px;font-weight:700;margin-bottom:5px;line-height:1}
.cval.p{
  background:linear-gradient(135deg,var(--green),#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.cval.f{color:var(--red)}

.cthresh{
  font-size:11px;color:var(--muted);margin-bottom:6px;
  display:flex;align-items:center;gap:6px
}
.cthresh::before{content:'';display:inline-block;width:4px;height:4px;
  border-radius:50%;background:var(--dim);flex-shrink:0}
.cnote{font-size:11px;color:var(--dim);line-height:1.5;font-style:italic}

.cnum{
  position:absolute;top:14px;right:14px;
  font-family:var(--mono);font-size:10px;
  background:var(--grad);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
  opacity:.6
}

.cat-tag{
  display:inline-block;font-family:var(--mono);font-size:9px;letter-spacing:.5px;
  padding:2px 7px;border-radius:4px;text-transform:uppercase;margin-bottom:8px
}
.cat-F{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.2)}
.cat-V{background:rgba(6,182,212,.12);color:#67e8f9;border:1px solid rgba(6,182,212,.2)}
.cat-G{background:rgba(245,158,11,.12);color:#fcd34d;border:1px solid rgba(245,158,11,.2)}
.cat-T{background:rgba(16,185,129,.12);color:#6ee7b7;border:1px solid rgba(16,185,129,.2)}

/* ── SUMMARY BAR ── */
.summary{
  background:var(--s1);border:1px solid var(--b1);border-radius:16px;
  padding:24px 28px;display:flex;gap:0;align-items:stretch;flex-wrap:wrap;
  box-shadow:0 8px 32px rgba(0,0,0,.25);
  position:relative;overflow:hidden;margin-bottom:0
}
.summary::before{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:var(--grad);opacity:.5
}

.ss{flex:1;min-width:110px;padding:0 20px;border-right:1px solid var(--b1)}
.ss:first-child{padding-left:0}
.ss:last-child{border-right:none}
@media(max-width:500px){.ss{border-right:none;border-bottom:1px solid var(--b1);padding:12px 0}}

.ssl{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:6px}
.ssv{font-family:var(--mono);font-size:22px;font-weight:700;color:#fff;line-height:1}
.ssv.green{
  background:linear-gradient(135deg,var(--green),#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.ssv.amber{
  background:linear-gradient(135deg,var(--amber),#fcd34d);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.ssv.orange{
  background:linear-gradient(135deg,#fb923c,#fdba74);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.ssv.red{
  background:linear-gradient(135deg,var(--red),#fb7185);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.sssub{font-size:11px;color:var(--muted);margin-top:4px}

/* ── UK NOTE ── */
.uknote{
  margin-top:16px;
  background:linear-gradient(135deg,rgba(124,58,237,.08),rgba(6,182,212,.06));
  border:1px solid rgba(124,58,237,.2);border-radius:12px;
  padding:16px 20px;font-size:12.5px;color:rgba(196,181,253,.8);line-height:1.8
}
.uknote strong{color:#c4b5fd}
.uknote-icon{margin-right:6px}
</style>
</head>
<body>

<!-- SVG gradient defs for ring -->
<svg width="0" height="0" style="position:absolute">
  <defs>
    <linearGradient id="gGreen" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#34d399"/>
    </linearGradient>
    <linearGradient id="gAmber" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#fcd34d"/>
    </linearGradient>
    <linearGradient id="gOrange" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f97316"/><stop offset="100%" stop-color="#fb923c"/>
    </linearGradient>
    <linearGradient id="gRed" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f43f5e"/><stop offset="100%" stop-color="#fb7185"/>
    </linearGradient>
  </defs>
</svg>

<div class="shell">

  <!-- HEADER -->
  <div class="header">
    <div class="logo-row">
      <div class="logo-icon">📈</div>
      <span class="logo-text">Global Stock Screener</span>
    </div>
    <h1>Invest with<br/>Confidence</h1>
    <p>Type any company name or ticker — screened against 10 criteria in seconds.</p>
  </div>

  <!-- SEARCH -->
  <div class="search-outer">
    <div class="search-wrap">
      <div class="search-icon">🔍</div>
      <input id="si" type="text"
             placeholder="Apple, Unilever, Novo Nordisk, MSFT, ASML.AS ..."
             autocomplete="off" spellcheck="false"/>
      <button class="btn-screen" id="sb" onclick="go()">SCREEN →</button>
    </div>
    <div class="autocomplete" id="dd"></div>
  </div>

  <!-- EXAMPLES -->
  <div class="examples">
    <span class="ex-label">Try:</span>
    <span class="ex-chip" onclick="qs('Apple')">🍎 Apple</span>
    <span class="ex-chip" onclick="qs('Microsoft')">💻 Microsoft</span>
    <span class="ex-chip" onclick="qs('Unilever')">🇬🇧 Unilever</span>
    <span class="ex-chip" onclick="qs('ASML')">🇳🇱 ASML</span>
    <span class="ex-chip" onclick="qs('Novo Nordisk')">💊 Novo Nordisk</span>
    <span class="ex-chip" onclick="qs('Visa')">💳 Visa</span>
    <span class="ex-chip" onclick="qs('AstraZeneca')">🔬 AstraZeneca</span>
  </div>

  <!-- STATES -->
  <div class="loader" id="ld"><div class="loader-ring"></div><p>ANALYSING ...</p></div>
  <div class="err" id="eb"></div>

  <!-- RESULTS -->
  <div id="results">
    <div class="rbanner" id="rb"></div>

    <!-- Stock header -->
    <div class="shead" id="sh"></div>

    <!-- Criteria -->
    <div class="sec-header">
      <span class="sec-label">10 Screening Criteria</span>
      <div class="sec-line"></div>
    </div>
    <div class="grid" id="cg"></div>

    <!-- Summary -->
    <div class="summary" id="sm"></div>

    <!-- UK note -->
    <div class="uknote" id="uk"></div>
  </div>

</div>

<script>
var inp=document.getElementById('si'),btn=document.getElementById('sb'),
    drop=document.getElementById('dd'),ld=document.getElementById('ld'),
    eb=document.getElementById('eb'),res=document.getElementById('results');
var timer=null,items=[],idx=-1,selTicker=null;

inp.addEventListener('input',function(){
  selTicker=null;clearTimeout(timer);
  var q=inp.value.trim();
  if(q.length<2){close();return;}
  drop.innerHTML='<div class="ac-msg">Searching...</div>';
  drop.classList.add('open');
  timer=setTimeout(function(){
    fetch('/api/search?q='+encodeURIComponent(q))
      .then(function(r){return r.json();})
      .then(function(d){items=d;renderDrop();})
      .catch(close);
  },280);
});

inp.addEventListener('keydown',function(e){
  if(e.key==='ArrowDown'){e.preventDefault();mv(1);}
  else if(e.key==='ArrowUp'){e.preventDefault();mv(-1);}
  else if(e.key==='Enter'){e.preventDefault();idx>=0?pick(idx):go();}
  else if(e.key==='Escape'){close();}
});
document.addEventListener('click',function(e){if(!e.target.closest('.search-outer'))close();});

function renderDrop(){
  if(!items.length){drop.innerHTML='<div class="ac-msg">No results found</div>';return;}
  drop.innerHTML=items.map(function(it,i){
    return "<div class='ac-item' onclick='pick("+i+")'>" +
           "<span class='ac-sym'>"+it.symbol+"</span>" +
           "<span class='ac-name'>"+it.name+"</span>" +
           "<span class='ac-ex'>"+it.exchange+"</span></div>";
  }).join('');
  idx=-1;
}

function mv(dir){
  var els=drop.querySelectorAll('.ac-item');if(!els.length)return;
  els.forEach(function(e){e.classList.remove('hi');});
  idx=(idx+dir+els.length)%els.length;els[idx].classList.add('hi');
}
function pick(i){var it=items[i];if(!it)return;selTicker=it.symbol;inp.value=it.name;close();go();}
function close(){drop.classList.remove('open');drop.innerHTML='';idx=-1;}
function qs(n){inp.value=n;selTicker=null;go();}

function setLoad(on){
  btn.disabled=on;btn.innerHTML=on?'...':'SCREEN →';
  ld.classList.toggle('on',on);
  eb.classList.remove('on');
  if(on)res.classList.remove('on');
}

async function go(){
  close();
  var q=(selTicker||inp.value).trim();
  if(!q){inp.focus();return;}
  setLoad(true);
  try{
    var r=await fetch('/api/screen?q='+encodeURIComponent(q));
    var d=await r.json();
    if(!r.ok||d.error)throw new Error(d.error||'Unknown error');
    render(d);
  }catch(e){
    eb.textContent='⚠  '+e.message;eb.classList.add('on');
  }finally{setLoad(false);}
}

function fmt(n){return Number(n).toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});}

var catMap={'Fundamentals':'F','Valuation':'V','Governance':'G','Technical':'T'};

function render(d){
  var vc=d.verdict_color,r=44,ci=2*Math.PI*r,off=ci-(ci*d.score_pct/100);
  var pr=d.price?d.currency+' '+fmt(d.price):'N/A';

  // Banner
  var rb=document.getElementById('rb'),typed=inp.value.trim();
  if(typed&&typed.toUpperCase()!==d.ticker){
    rb.innerHTML="<span class='rbanner-arrow'>↳</span> Resolved <strong>\""+typed+"\"</strong> &rarr; <strong>"+d.ticker+"</strong> ("+d.name+")";
    rb.classList.add('on');
  }else{rb.classList.remove('on');}

  // Stock header
  document.getElementById('sh').innerHTML=
    "<div class='smeta'>"+
      "<div class='sticker'>"+d.ticker+"</div>"+
      "<div class='sname'>"+d.name+"</div>"+
      "<div class='stags'>"+
        (d.sector?"<span class='tag'>"+d.sector+"</span>":"")+
        (d.country?"<span class='tag'>"+d.country+"</span>":"")+
        (d.exchange?"<span class='tag'>"+d.exchange+"</span>":"")+
        "<span class='tag price'>"+pr+"</span>"+
      "</div>"+
    "</div>"+
    "<div class='srwrap'>"+
      "<div class='sring'>"+
        "<svg viewBox='0 0 96 96' width='96' height='96'>"+
          "<circle class='rbg' cx='48' cy='48' r='"+r+"'/>"+
          "<circle class='rfill "+vc+"' cx='48' cy='48' r='"+r+"' id='rf'"+
            " style='stroke-dasharray:"+ci.toFixed(1)+";stroke-dashoffset:"+ci.toFixed(1)+"'/>"+
        "</svg>"+
        "<div class='rinner'>"+
          "<div class='rscore'>"+d.passed+"/"+d.total+"</div>"+
          "<div class='rsub'>CRITERIA</div>"+
        "</div>"+
      "</div>"+
      "<div class='vpill "+vc+"'>"+d.verdict+"</div>"+
    "</div>";

  setTimeout(function(){var rf=document.getElementById('rf');if(rf)rf.style.strokeDashoffset=off;},100);

  // Cards
  var cg=document.getElementById('cg');cg.innerHTML='';
  d.criteria.forEach(function(c,i){
    var el=document.createElement('div');
    el.className='card '+(c.passed?'p':'f');
    var catKey=catMap[c.category]||'F';
    el.innerHTML=
      "<div class='cnum'>#"+(i+1)+"</div>"+
      "<div class='cat-tag cat-"+catKey+"'>"+c.category+"</div>"+
      "<div class='ctop'>"+
        "<div class='cname'>"+c.name+"</div>"+
        "<span class='badge "+(c.passed?'bp':'bf')+"'>"+(c.passed?'✓ Pass':'✗ Fail')+"</span>"+
      "</div>"+
      "<div class='cval "+(c.passed?'p':'f')+"'>"+c.value+"</div>"+
      "<div class='cthresh'>"+c.threshold+"</div>"+
      (c.note?"<div class='cnote'>"+c.note+"</div>":'');
    cg.appendChild(el);
    setTimeout(function(){el.classList.add('vis');},50+i*50);
  });

  // Summary
  var cats={};
  d.criteria.forEach(function(c){
    if(!cats[c.category])cats[c.category]={p:0,t:0};
    cats[c.category].t++;if(c.passed)cats[c.category].p++;
  });
  var catHtml=Object.entries(cats).map(function(e){
    return "<div class='ss'>"+
           "<div class='ssl'>"+e[0]+"</div>"+
           "<div class='ssv' style='font-size:18px'>"+e[1].p+"/"+e[1].t+"</div>"+
           "</div>";
  }).join('');

  document.getElementById('sm').innerHTML=
    "<div class='ss'>"+
      "<div class='ssl'>Criteria Passed</div>"+
      "<div class='ssv "+vc+"'>"+d.passed+" / "+d.total+"</div>"+
      "<div class='sssub'>"+d.score_pct+"% pass rate</div>"+
    "</div>"+
    "<div class='ss'>"+
      "<div class='ssl'>Verdict</div>"+
      "<div class='ssv "+vc+"' style='font-size:15px;line-height:1.3'>"+d.verdict+"</div>"+
    "</div>"+
    catHtml;

  // UK note
  var note="<span class='uknote-icon'>🇬🇧</span><strong>UK Investor Notes &mdash; "+d.name+"</strong><br>";
  if(d.country==='United States')
    note+="<strong>Withholding Tax:</strong> Submit <strong>W-8BEN</strong> form to your broker to reduce US dividend WHT from 30% &rarr; 15% (UK-US tax treaty). &nbsp;";
  else if(d.country==='United Kingdom')
    note+="<strong>Stamp Duty:</strong> 0.5% on UK share purchases. No WHT on UK dividends. &nbsp;";
  else
    note+="<strong>Withholding Tax:</strong> Check UK double-taxation treaty with <strong>"+d.country+"</strong> for dividend WHT rate. &nbsp;";
  note+="Hold in your <strong>Trading 212 Stocks &amp; Shares ISA</strong> (£20,000/year) to shelter gains &amp; dividends from UK tax.";
  document.getElementById('uk').innerHTML=note;

  res.classList.add('on');
  window.scrollTo({top:0,behavior:'smooth'});
}
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print(f"""
╔════════════════════════════════════════════╗
║   GLOBAL STOCK SCREENER  --  Starting  ... ║
║   Running on port: {port:<24}              ║
╚════════════════════════════════════════════╝
""", flush=True)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
