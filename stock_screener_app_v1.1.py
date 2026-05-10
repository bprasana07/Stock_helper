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

HTML = (
"<!DOCTYPE html>\n"
"<html lang='en'>\n"
"<head>\n"
"<meta charset='UTF-8'/>\n"
"<meta name='viewport' content='width=device-width,initial-scale=1.0'/>\n"
"<title>Global Stock Screener</title>\n"
"<link rel='preconnect' href='https://fonts.googleapis.com'>\n"
"<link href='https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap' rel='stylesheet'>\n"
"<style>\n"
"*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n"
":root{\n"
"  --bg:#090e1a;--surface:#0d1526;--border:#1e2d4a;--border2:#243556;\n"
"  --text:#c8d8f0;--muted:#4a6080;--dim:#1e3050;\n"
"  --gold:#f0b942;--gold-dim:#7a5a18;\n"
"  --green:#2ecc8a;--green-dim:#0d3d26;\n"
"  --red:#f05252;--red-dim:#3d0f0f;\n"
"  --amber:#f0952a;--orange:#f07a28;\n"
"  --blue:#4a9eff;--blue-dim:#0d2040;\n"
"  --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif;\n"
"}\n"
"body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}\n"
"body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;\n"
"  background-image:linear-gradient(rgba(74,158,255,.03) 1px,transparent 1px),\n"
"  linear-gradient(90deg,rgba(74,158,255,.03) 1px,transparent 1px);\n"
"  background-size:40px 40px;}\n"
".shell{position:relative;z-index:1;max-width:900px;margin:0 auto;padding:40px 24px 80px}\n"
".header{text-align:center;margin-bottom:40px}\n"
".header-tag{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:3px;\n"
"  color:var(--blue);border:1px solid var(--blue-dim);padding:4px 12px;\n"
"  border-radius:2px;margin-bottom:16px;text-transform:uppercase}\n"
".header h1{font-family:var(--mono);font-size:clamp(22px,4vw,34px);font-weight:700;\n"
"  color:#fff;letter-spacing:-1px}\n"
".header h1 span{color:var(--gold)}\n"
".header p{margin-top:10px;color:var(--muted);font-size:14px}\n"
".search-outer{position:relative;margin-bottom:12px}\n"
".search-wrap{display:flex;gap:10px;background:var(--surface);border:1px solid var(--border);\n"
"  border-radius:6px;padding:6px 6px 6px 20px;transition:border-color .2s}\n"
".search-wrap:focus-within{border-color:var(--blue)}\n"
".search-wrap input{flex:1;background:none;border:none;outline:none;\n"
"  font-family:var(--sans);font-size:17px;font-weight:500;color:#fff;}\n"
".search-wrap input::placeholder{color:var(--dim);font-size:14px;font-weight:300}\n"
".autocomplete{position:absolute;top:calc(100% + 4px);left:0;right:0;z-index:100;\n"
"  background:var(--surface);border:1px solid var(--border2);border-radius:6px;\n"
"  overflow:hidden;display:none;box-shadow:0 8px 32px rgba(0,0,0,.5)}\n"
".autocomplete.open{display:block}\n"
".ac-item{display:flex;align-items:center;gap:12px;\n"
"  padding:11px 16px;cursor:pointer;transition:background .12s;border-bottom:1px solid var(--border)}\n"
".ac-item:last-child{border-bottom:none}\n"
".ac-item:hover,.ac-item.hi{background:var(--border)}\n"
".ac-sym{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--gold);min-width:80px}\n"
".ac-name{font-size:13px;color:var(--text);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n"
".ac-ex{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:1px}\n"
".ac-msg{padding:14px 16px;font-size:13px;color:var(--muted);font-family:var(--mono)}\n"
".btn-screen{background:var(--gold);color:#0a0800;font-family:var(--mono);font-size:12px;\n"
"  font-weight:700;letter-spacing:2px;text-transform:uppercase;\n"
"  border:none;border-radius:4px;padding:14px 24px;cursor:pointer;\n"
"  transition:opacity .15s;white-space:nowrap;flex-shrink:0}\n"
".btn-screen:hover{opacity:.85}\n"
".btn-screen:disabled{opacity:.4;cursor:not-allowed}\n"
".examples{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:32px;align-items:center}\n"
".ex-label{font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:1px;text-transform:uppercase}\n"
".ex-chip{font-size:12px;color:var(--muted);border:1px solid var(--border2);\n"
"  padding:5px 12px;border-radius:3px;cursor:pointer;transition:color .15s,border-color .15s}\n"
".ex-chip:hover{color:var(--gold);border-color:var(--gold-dim)}\n"
".loader{display:none;text-align:center;padding:56px 0}\n"
".loader.on{display:block}\n"
".loader-ring{width:44px;height:44px;border:2px solid var(--border2);border-top-color:var(--gold);\n"
"  border-radius:50%;margin:0 auto 14px;animation:spin .8s linear infinite}\n"
"@keyframes spin{to{transform:rotate(360deg)}}\n"
".loader p{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:2px}\n"
".err{display:none;background:var(--red-dim);border:1px solid var(--red);\n"
"  border-radius:6px;padding:16px 20px;font-family:var(--mono);font-size:13px;color:var(--red)}\n"
".err.on{display:block}\n"
"#results{display:none}\n"
"#results.on{display:block;animation:fu .4s ease}\n"
"@keyframes fu{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}\n"
".rbanner{display:none;background:var(--blue-dim);border:1px solid rgba(74,158,255,.2);\n"
"  border-radius:5px;padding:10px 16px;margin-bottom:14px;\n"
"  font-size:12px;color:rgba(74,158,255,.85);font-family:var(--mono)}\n"
".rbanner.on{display:block}\n"
".shead{background:var(--surface);border:1px solid var(--border);border-radius:8px;\n"
"  padding:26px 30px;margin-bottom:22px;display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap}\n"
".smeta{flex:1;min-width:180px}\n"
".sticker{font-family:var(--mono);font-size:26px;font-weight:700;color:var(--gold);letter-spacing:2px}\n"
".sname{font-size:15px;color:#fff;margin:4px 0 10px;font-weight:500}\n"
".stags{display:flex;gap:8px;flex-wrap:wrap}\n"
".tag{font-family:var(--mono);font-size:10px;letter-spacing:1px;color:var(--muted);\n"
"  border:1px solid var(--border2);padding:3px 8px;border-radius:2px;text-transform:uppercase}\n"
".srwrap{display:flex;flex-direction:column;align-items:center;gap:8px}\n"
".sring{position:relative;width:96px;height:96px}\n"
".sring svg{transform:rotate(-90deg)}\n"
".sring circle{fill:none;stroke-width:6}\n"
".rbg{stroke:var(--border2)}\n"
".rfill{stroke-linecap:round;stroke-dasharray:264;stroke-dashoffset:264;\n"
"  transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)}\n"
".rfill.green{stroke:var(--green)}.rfill.amber{stroke:var(--amber)}\n"
".rfill.orange{stroke:var(--orange)}.rfill.red{stroke:var(--red)}\n"
".rinner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;\n"
"  justify-content:center;font-family:var(--mono)}\n"
".rscore{font-size:20px;font-weight:700;color:#fff;line-height:1}\n"
".rsub{font-size:9px;color:var(--muted);margin-top:2px;letter-spacing:1px}\n"
".vpill{font-family:var(--mono);font-size:10px;letter-spacing:1px;\n"
"  text-transform:uppercase;text-align:center;max-width:104px;line-height:1.4}\n"
".vpill.green{color:var(--green)}.vpill.amber{color:var(--amber)}\n"
".vpill.orange{color:var(--orange)}.vpill.red{color:var(--red)}\n"
".sec-label{font-family:var(--mono);font-size:10px;letter-spacing:3px;\n"
"  color:var(--muted);text-transform:uppercase;margin-bottom:12px}\n"
".grid{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-bottom:20px}\n"
"@media(max-width:580px){.grid{grid-template-columns:1fr}}\n"
".card{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--border);\n"
"  border-radius:6px;padding:15px 16px;opacity:0;transform:translateY(8px);\n"
"  transition:opacity .3s,transform .3s;position:relative;overflow:hidden}\n"
".card.vis{opacity:1;transform:translateY(0)}\n"
".card.p{border-left-color:var(--green)}\n"
".card.f{border-left-color:var(--red)}\n"
".card.p::before{content:'';position:absolute;inset:0;pointer-events:none;\n"
"  background:linear-gradient(135deg,rgba(46,204,138,.04),transparent 60%)}\n"
".ctop{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:9px}\n"
".cname{font-size:12px;font-weight:600;color:#fff;line-height:1.3}\n"
".badge{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.5px;\n"
"  padding:2px 7px;border-radius:2px;flex-shrink:0}\n"
".bp{background:var(--green-dim);color:var(--green)}\n"
".bf{background:var(--red-dim);color:var(--red)}\n"
".cval{font-family:var(--mono);font-size:16px;font-weight:700;margin-bottom:3px}\n"
".cval.p{color:var(--green)}.cval.f{color:var(--red)}\n"
".cthresh{font-size:11px;color:var(--muted);margin-bottom:5px}\n"
".cnote{font-size:11px;color:var(--muted);font-style:italic;line-height:1.4}\n"
".cnum{position:absolute;top:11px;right:12px;font-family:var(--mono);font-size:10px;color:var(--dim)}\n"
".summary{background:var(--surface);border:1px solid var(--border);border-radius:8px;\n"
"  padding:22px 26px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}\n"
".ss{flex:1;min-width:100px}\n"
".ssl{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--muted);\n"
"  text-transform:uppercase;margin-bottom:3px}\n"
".ssv{font-family:var(--mono);font-size:20px;font-weight:700;color:#fff}\n"
".ssv.green{color:var(--green)}.ssv.amber{color:var(--amber)}\n"
".ssv.orange{color:var(--orange)}.ssv.red{color:var(--red)}\n"
".sssub{font-size:11px;color:var(--muted);margin-top:2px}\n"
".sdiv{width:1px;background:var(--border);align-self:stretch;flex-shrink:0}\n"
".uknote{margin-top:14px;background:var(--blue-dim);border:1px solid rgba(74,158,255,.15);\n"
"  border-radius:6px;padding:13px 17px;font-size:12px;color:rgba(74,158,255,.8);line-height:1.7}\n"
".uknote strong{color:var(--blue)}\n"
"</style>\n"
"</head>\n"
"<body>\n"
"<div class='shell'>\n"
"  <div class='header'>\n"
"    <div class='header-tag'>UK GBP &middot; Global Equities</div>\n"
"    <h1>STOCK <span>SCREENER</span></h1>\n"
"    <p>Type a company name or ticker &mdash; we'll find it and screen it against 10 criteria.</p>\n"
"  </div>\n"
"  <div class='search-outer'>\n"
"    <div class='search-wrap'>\n"
"      <input id='si' type='text'\n"
"             placeholder='e.g.  Apple   Unilever   Novo Nordisk   MSFT   ASML.AS'\n"
"             autocomplete='off' spellcheck='false'/>\n"
"      <button class='btn-screen' id='sb' onclick='go()'>SCREEN &rarr;</button>\n"
"    </div>\n"
"    <div class='autocomplete' id='dd'></div>\n"
"  </div>\n"
"  <div class='examples'>\n"
"    <span class='ex-label'>Try:</span>\n"
"    <span class='ex-chip' onclick='qs(\"Apple\")'>Apple</span>\n"
"    <span class='ex-chip' onclick='qs(\"Microsoft\")'>Microsoft</span>\n"
"    <span class='ex-chip' onclick='qs(\"Unilever\")'>Unilever</span>\n"
"    <span class='ex-chip' onclick='qs(\"ASML\")'>ASML</span>\n"
"    <span class='ex-chip' onclick='qs(\"Novo Nordisk\")'>Novo Nordisk</span>\n"
"    <span class='ex-chip' onclick='qs(\"Visa\")'>Visa</span>\n"
"    <span class='ex-chip' onclick='qs(\"AstraZeneca\")'>AstraZeneca</span>\n"
"    <span class='ex-chip' onclick='qs(\"Mastercard\")'>Mastercard</span>\n"
"  </div>\n"
"  <div class='loader' id='ld'><div class='loader-ring'></div><p>ANALYSING&hellip;</p></div>\n"
"  <div class='err' id='eb'></div>\n"
"  <div id='results'>\n"
"    <div class='rbanner' id='rb'></div>\n"
"    <div class='shead' id='sh'></div>\n"
"    <div class='sec-label'>10 SCREENING CRITERIA</div>\n"
"    <div class='grid' id='cg'></div>\n"
"    <div class='summary' id='sm'></div>\n"
"    <div class='uknote' id='uk'></div>\n"
"  </div>\n"
"</div>\n"
"<script>\n"
"var inp=document.getElementById('si'),btn=document.getElementById('sb'),\n"
"    drop=document.getElementById('dd'),ld=document.getElementById('ld'),\n"
"    eb=document.getElementById('eb'),res=document.getElementById('results');\n"
"var timer=null,items=[],idx=-1,selTicker=null;\n"
"\n"
"inp.addEventListener('input',function(){\n"
"  selTicker=null;clearTimeout(timer);\n"
"  var q=inp.value.trim();\n"
"  if(q.length<2){close();return;}\n"
"  timer=setTimeout(function(){fetch('/api/search?q='+encodeURIComponent(q))\n"
"    .then(function(r){return r.json();})\n"
"    .then(function(d){items=d;renderDrop();})\n"
"    .catch(close);\n"
"    drop.innerHTML='<div class=\"ac-msg\">Searching&hellip;</div>';\n"
"    drop.classList.add('open');\n"
"  },280);\n"
"});\n"
"\n"
"inp.addEventListener('keydown',function(e){\n"
"  if(e.key==='ArrowDown'){e.preventDefault();mv(1);}\n"
"  else if(e.key==='ArrowUp'){e.preventDefault();mv(-1);}\n"
"  else if(e.key==='Enter'){e.preventDefault();idx>=0?pick(idx):go();}\n"
"  else if(e.key==='Escape'){close();}\n"
"});\n"
"\n"
"document.addEventListener('click',function(e){\n"
"  if(!e.target.closest('.search-outer'))close();\n"
"});\n"
"\n"
"function renderDrop(){\n"
"  if(!items.length){drop.innerHTML='<div class=\"ac-msg\">No results found</div>';return;}\n"
"  drop.innerHTML=items.map(function(it,i){\n"
"    return \"<div class='ac-item' onclick='pick(\"+i+\")'>\"+\n"
"           \"<span class='ac-sym'>\"+it.symbol+\"</span>\"+\n"
"           \"<span class='ac-name'>\"+it.name+\"</span>\"+\n"
"           \"<span class='ac-ex'>\"+it.exchange+\"</span></div>\";\n"
"  }).join('');\n"
"  idx=-1;\n"
"}\n"
"\n"
"function mv(dir){\n"
"  var els=drop.querySelectorAll('.ac-item');\n"
"  if(!els.length)return;\n"
"  els.forEach(function(e){e.classList.remove('hi');});\n"
"  idx=(idx+dir+els.length)%els.length;\n"
"  els[idx].classList.add('hi');\n"
"}\n"
"\n"
"function pick(i){\n"
"  var it=items[i];if(!it)return;\n"
"  selTicker=it.symbol;inp.value=it.name;close();go();\n"
"}\n"
"\n"
"function close(){drop.classList.remove('open');drop.innerHTML='';idx=-1;}\n"
"\n"
"function qs(name){inp.value=name;selTicker=null;go();}\n"
"\n"
"function setLoad(on){\n"
"  btn.disabled=on;btn.innerHTML=on?'&hellip;':'SCREEN &rarr;';\n"
"  ld.classList.toggle('on',on);\n"
"  eb.classList.remove('on');\n"
"  if(on)res.classList.remove('on');\n"
"}\n"
"\n"
"async function go(){\n"
"  close();\n"
"  var q=(selTicker||inp.value).trim();\n"
"  if(!q){inp.focus();return;}\n"
"  setLoad(true);\n"
"  try{\n"
"    var r=await fetch('/api/screen?q='+encodeURIComponent(q));\n"
"    var d=await r.json();\n"
"    if(!r.ok||d.error)throw new Error(d.error||'Unknown error');\n"
"    render(d);\n"
"  }catch(e){\n"
"    eb.textContent='Error: '+e.message;eb.classList.add('on');\n"
"  }finally{setLoad(false);}\n"
"}\n"
"\n"
"function fmt(n){return Number(n).toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});}\n"
"\n"
"function render(d){\n"
"  var vc=d.verdict_color,r=42,ci=2*Math.PI*r,off=ci-(ci*d.score_pct/100);\n"
"  var pr=d.price?d.currency+' '+fmt(d.price):'N/A';\n"
"\n"
"  var rb=document.getElementById('rb'),typed=inp.value.trim();\n"
"  if(typed&&typed.toUpperCase()!==d.ticker){\n"
"    rb.textContent='Resolved \"'+typed+'\" -> '+d.ticker+' ('+d.name+')';\n"
"    rb.classList.add('on');\n"
"  }else{rb.classList.remove('on');}\n"
"\n"
"  document.getElementById('sh').innerHTML=\n"
"    \"<div class='smeta'>\"+\n"
"      \"<div class='sticker'>\"+d.ticker+\"</div>\"+\n"
"      \"<div class='sname'>\"+d.name+\"</div>\"+\n"
"      \"<div class='stags'>\"+\n"
"        ['sector','country','exchange'].map(function(k){\n"
"          return \"<span class='tag'>\"+(d[k]||'&mdash;')+\"</span>\";\n"
"        }).join('')+\n"
"        \"<span class='tag'>Price: \"+pr+\"</span>\"+\n"
"      \"</div>\"+\n"
"    \"</div>\"+\n"
"    \"<div class='srwrap'>\"+\n"
"      \"<div class='sring'>\"+\n"
"        \"<svg viewBox='0 0 96 96' width='96' height='96'>\"+\n"
"          \"<circle class='rbg' cx='48' cy='48' r='\"+r+\"'/>\"+\n"
"          \"<circle class='rfill \"+vc+\"' cx='48' cy='48' r='\"+r+\"' id='rf'\"+\n"
"            \" style='stroke-dasharray:\"+ci.toFixed(1)+\";stroke-dashoffset:\"+ci.toFixed(1)+\"'/>\"+\n"
"        \"</svg>\"+\n"
"        \"<div class='rinner'>\"+\n"
"          \"<div class='rscore'>\"+d.passed+\"/\"+d.total+\"</div>\"+\n"
"          \"<div class='rsub'>CRITERIA</div>\"+\n"
"        \"</div>\"+\n"
"      \"</div>\"+\n"
"      \"<div class='vpill \"+vc+\"'>\"+d.verdict+\"</div>\"+\n"
"    \"</div>\";\n"
"\n"
"  setTimeout(function(){\n"
"    var rf=document.getElementById('rf');\n"
"    if(rf)rf.style.strokeDashoffset=off;\n"
"  },80);\n"
"\n"
"  var cg=document.getElementById('cg');\n"
"  cg.innerHTML='';\n"
"  d.criteria.forEach(function(c,i){\n"
"    var el=document.createElement('div');\n"
"    el.className='card '+(c.passed?'p':'f');\n"
"    el.innerHTML=\n"
"      \"<div class='cnum'>#\"+(i+1)+\"</div>\"+\n"
"      \"<div class='ctop'>\"+\n"
"        \"<div class='cname'>\"+c.name+\"</div>\"+\n"
"        \"<span class='badge \"+(c.passed?'bp':'bf')+\"'>\"+(c.passed?'&#10003; PASS':'&#10007; FAIL')+\"</span>\"+\n"
"      \"</div>\"+\n"
"      \"<div class='cval \"+(c.passed?'p':'f')+\"'>\"+c.value+\"</div>\"+\n"
"      \"<div class='cthresh'>Threshold: \"+c.threshold+\"</div>\"+\n"
"      (c.note?\"<div class='cnote'>\"+c.note+\"</div>\":'');\n"
"    cg.appendChild(el);\n"
"    setTimeout(function(){el.classList.add('vis');},55+i*55);\n"
"  });\n"
"\n"
"  var cats={};\n"
"  d.criteria.forEach(function(c){\n"
"    if(!cats[c.category])cats[c.category]={p:0,t:0};\n"
"    cats[c.category].t++;\n"
"    if(c.passed)cats[c.category].p++;\n"
"  });\n"
"  var catHtml=Object.entries(cats).map(function(e){\n"
"    return \"<div class='sdiv'></div><div class='ss'>\"+\n"
"           \"<div class='ssl'>\"+e[0]+\"</div>\"+\n"
"           \"<div class='ssv' style='font-size:17px'>\"+e[1].p+\"/\"+e[1].t+\"</div>\"+\n"
"           \"</div>\";\n"
"  }).join('');\n"
"\n"
"  document.getElementById('sm').innerHTML=\n"
"    \"<div class='ss'><div class='ssl'>Criteria Passed</div>\"+\n"
"    \"<div class='ssv \"+vc+\"'>\"+d.passed+\" / \"+d.total+\"</div>\"+\n"
"    \"<div class='sssub'>\"+d.score_pct+\"% pass rate</div></div>\"+\n"
"    \"<div class='sdiv'></div>\"+\n"
"    \"<div class='ss'><div class='ssl'>Verdict</div>\"+\n"
"    \"<div class='ssv \"+vc+\"' style='font-size:14px;line-height:1.3'>\"+d.verdict+\"</div></div>\"+\n"
"    catHtml;\n"
"\n"
"  var note=\"<strong>UK Investor Notes &mdash; \"+d.name+\"</strong><br>\";\n"
"  if(d.country==='United States')\n"
"    note+='<strong>Withholding Tax:</strong> Submit <strong>W-8BEN</strong> form to your broker '+\n"
"          'to reduce US dividend WHT from 30% to 15% under the UK-US tax treaty. &nbsp;';\n"
"  else if(d.country==='United Kingdom')\n"
"    note+='<strong>Stamp Duty:</strong> 0.5% on UK share purchases. No WHT on UK dividends. &nbsp;';\n"
"  else\n"
"    note+='<strong>Withholding Tax:</strong> Check UK double-taxation treaty with <strong>'+\n"
"          d.country+'</strong> for dividend WHT rate. &nbsp;';\n"
"  note+='Hold in your <strong>Trading 212 Stocks &amp; Shares ISA</strong> '+\n"
"        '(GBP 20,000/year) to shelter gains and dividends from UK tax.';\n"
"  document.getElementById('uk').innerHTML=note;\n"
"\n"
"  res.classList.add('on');\n"
"  window.scrollTo({top:0,behavior:'smooth'});\n"
"}\n"
"</script>\n"
"</body>\n"
"</html>"
)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  GLOBAL STOCK SCREENER -- Starting...")
    print("  Open browser at:  http://localhost:5000")
    print("  Type any company name or ticker symbol\n")
   app.run(debug=True, host="0.0.0.0", port=5000)
