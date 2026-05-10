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
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0b0f1e;color:#e2e8ff;min-height:100vh}
.bg-glow{position:fixed;top:0;left:0;right:0;bottom:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 700px 500px at 15% 10%,rgba(109,40,217,0.18),transparent),
             radial-gradient(ellipse 600px 400px at 85% 80%,rgba(6,182,212,0.12),transparent)}
.shell{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:48px 24px 100px}

/* HEADER */
.header{text-align:center;margin-bottom:44px}
.hbadge{display:inline-block;background:rgba(109,40,217,0.2);border:1px solid rgba(109,40,217,0.4);
  color:#a78bfa;font-size:11px;letter-spacing:3px;text-transform:uppercase;
  padding:5px 16px;border-radius:20px;margin-bottom:20px}
.header h1{font-size:clamp(32px,5vw,54px);font-weight:800;letter-spacing:-2px;
  line-height:1.05;margin-bottom:14px;color:#fff}
.header h1 .hi{color:#a78bfa}
.header p{color:#64748b;font-size:15px}

/* SEARCH */
.search-outer{position:relative;margin-bottom:16px}
.search-box{display:flex;align-items:center;background:#111827;border:1px solid #1e293b;
  border-radius:14px;padding:6px 6px 6px 18px;
  transition:border-color 0.2s,box-shadow 0.2s;
  box-shadow:0 4px 24px rgba(0,0,0,0.4)}
.search-box:focus-within{border-color:#7c3aed;
  box-shadow:0 0 0 3px rgba(124,58,237,0.2),0 4px 24px rgba(0,0,0,0.4)}
.search-box input{flex:1;background:none;border:none;outline:none;
  font-size:16px;font-weight:500;color:#fff;padding:10px 8px;font-family:inherit}
.search-box input::placeholder{color:#334155}
.btn-go{background:linear-gradient(135deg,#7c3aed,#06b6d4);color:#fff;border:none;
  border-radius:10px;padding:14px 28px;font-size:13px;font-weight:700;letter-spacing:1px;
  cursor:pointer;transition:opacity 0.2s,transform 0.1s;white-space:nowrap;
  box-shadow:0 4px 16px rgba(124,58,237,0.35)}
.btn-go:hover{opacity:0.88}
.btn-go:active{transform:scale(0.97)}
.btn-go:disabled{opacity:0.35;cursor:not-allowed}

/* AUTOCOMPLETE */
.ac{position:absolute;top:calc(100% + 6px);left:0;right:0;z-index:200;
  background:#111827;border:1px solid #1e293b;border-radius:12px;
  overflow:hidden;display:none;box-shadow:0 20px 60px rgba(0,0,0,0.6)}
.ac.open{display:block}
.ac-row{display:flex;align-items:center;gap:12px;padding:12px 18px;cursor:pointer;
  border-bottom:1px solid #1e293b;transition:background 0.12s}
.ac-row:last-child{border-bottom:none}
.ac-row:hover,.ac-row.hi{background:#1e293b}
.ac-sym{font-size:13px;font-weight:700;color:#a78bfa;min-width:80px;font-family:monospace}
.ac-nm{font-size:13px;color:#e2e8ff;flex:1;font-weight:500;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ac-ex{font-size:10px;color:#475569;background:#0f172a;padding:2px 8px;
  border-radius:4px;font-family:monospace}
.ac-msg{padding:14px 18px;font-size:13px;color:#475569}

/* EXAMPLES */
.examples{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:40px}
.ex-lbl{font-size:11px;color:#334155;letter-spacing:1px;text-transform:uppercase}
.chip{font-size:13px;font-weight:500;color:#64748b;border:1px solid #1e293b;
  background:#111827;padding:6px 14px;border-radius:20px;cursor:pointer;transition:all 0.18s}
.chip:hover{color:#a78bfa;border-color:rgba(124,58,237,0.4);background:rgba(124,58,237,0.1)}

/* STATES */
.loader{display:none;text-align:center;padding:60px 0}
.loader.on{display:block}
.spin{width:44px;height:44px;border:2px solid #1e293b;border-top-color:#7c3aed;
  border-radius:50%;margin:0 auto 16px;animation:spin 0.75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loader p{font-size:11px;color:#475569;letter-spacing:3px;text-transform:uppercase}
.err-box{display:none;background:rgba(244,63,94,0.1);border:1px solid rgba(244,63,94,0.3);
  border-radius:10px;padding:16px 20px;color:#f43f5e;font-size:14px}
.err-box.on{display:block}

/* RESULTS */
#results{display:none}
#results.on{display:block;animation:rise 0.4s ease}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.res-banner{display:none;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
  border-radius:10px;padding:10px 18px;margin-bottom:16px;font-size:13px;color:#7dd3fc}
.res-banner.on{display:block}

/* STOCK CARD */
.stock-card{background:#111827;border:1px solid #1e293b;border-radius:16px;
  padding:28px 32px;margin-bottom:28px;display:flex;gap:24px;
  align-items:flex-start;flex-wrap:wrap;
  box-shadow:0 8px 32px rgba(0,0,0,0.3);position:relative;overflow:hidden}
.stock-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#7c3aed,#06b6d4)}
.smeta{flex:1;min-width:200px}
.s-ticker{font-size:30px;font-weight:800;letter-spacing:1px;color:#a78bfa;margin-bottom:4px}
.s-name{font-size:16px;font-weight:600;color:#fff;margin-bottom:14px}
.s-tags{display:flex;gap:8px;flex-wrap:wrap}
.s-tag{font-size:11px;color:#64748b;border:1px solid #1e293b;padding:4px 10px;
  border-radius:6px;background:#0f172a}
.s-tag.price-tag{color:#38bdf8;border-color:rgba(56,189,248,0.25);
  background:rgba(56,189,248,0.08);font-weight:600}

/* RING */
.ring-wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.ring-rel{position:relative;width:100px;height:100px}
.ring-svg{transform:rotate(-90deg)}
.r-bg{fill:none;stroke:#1e293b;stroke-width:7}
.r-fill{fill:none;stroke-width:7;stroke-linecap:round;stroke-dasharray:283;stroke-dashoffset:283;
  transition:stroke-dashoffset 1.3s cubic-bezier(0.4,0,0.2,1)}
.r-fill.green{stroke:#10b981}.r-fill.amber{stroke:#f59e0b}
.r-fill.orange{stroke:#f97316}.r-fill.red{stroke:#f43f5e}
.ring-inner{position:absolute;top:0;left:0;right:0;bottom:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring-num{font-size:22px;font-weight:800;color:#fff;line-height:1}
.ring-lbl{font-size:9px;color:#475569;letter-spacing:2px;margin-top:3px}
.vp{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  text-align:center;padding:5px 12px;border-radius:20px;max-width:115px;line-height:1.5}
.vp.green{color:#10b981;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3)}
.vp.amber{color:#f59e0b;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3)}
.vp.orange{color:#f97316;background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.3)}
.vp.red{color:#f43f5e;background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.3)}

/* SECTION */
.sec-row{display:flex;align-items:center;gap:14px;margin-bottom:16px}
.sec-title{font-size:11px;letter-spacing:3px;color:#475569;text-transform:uppercase;white-space:nowrap}
.sec-line{flex:1;height:1px;background:linear-gradient(90deg,#1e293b,transparent)}

/* GRID */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}
@media(max-width:580px){.grid{grid-template-columns:1fr}}
.crd{background:#111827;border:1px solid #1e293b;border-radius:12px;
  padding:18px 18px 16px;position:relative;overflow:hidden;
  opacity:0;transform:translateY(10px);
  transition:opacity 0.3s ease,transform 0.3s ease,border-color 0.2s}
.crd:hover{border-color:#334155}
.crd.vis{opacity:1;transform:translateY(0)}
.crd.pass{border-left:3px solid #10b981}
.crd.fail{border-left:3px solid #f43f5e}
.crd-num{position:absolute;top:12px;right:14px;font-size:10px;color:#334155;font-family:monospace;font-weight:700}
.cat-pill{display:inline-block;font-size:9px;letter-spacing:1px;text-transform:uppercase;
  padding:2px 8px;border-radius:4px;margin-bottom:10px;font-weight:600}
.cat-F{background:rgba(124,58,237,0.15);color:#a78bfa}
.cat-V{background:rgba(56,189,248,0.12);color:#7dd3fc}
.cat-G{background:rgba(245,158,11,0.12);color:#fcd34d}
.cat-T{background:rgba(16,185,129,0.12);color:#6ee7b7}
.crd-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.crd-name{font-size:13px;font-weight:600;color:#e2e8ff;line-height:1.3}
.badge{font-size:9px;font-weight:700;letter-spacing:0.5px;padding:3px 8px;
  border-radius:6px;flex-shrink:0;text-transform:uppercase}
.b-pass{background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3)}
.b-fail{background:rgba(244,63,94,0.15);color:#f43f5e;border:1px solid rgba(244,63,94,0.3)}
.crd-val{font-size:19px;font-weight:800;margin-bottom:5px;line-height:1;font-family:monospace}
.crd-val.pass{color:#10b981}.crd-val.fail{color:#f43f5e}
.crd-thresh{font-size:11px;color:#475569;margin-bottom:6px}
.crd-note{font-size:11px;color:#334155;line-height:1.5;font-style:italic}

/* SUMMARY */
.summary{background:#111827;border:1px solid #1e293b;border-radius:16px;
  padding:24px 28px;display:flex;flex-wrap:wrap;gap:0;
  box-shadow:0 8px 32px rgba(0,0,0,0.25);position:relative;overflow:hidden}
.summary::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#7c3aed,#06b6d4)}
.ss{flex:1;min-width:110px;padding:0 20px;border-right:1px solid #1e293b}
.ss:first-child{padding-left:0}
.ss:last-child{border-right:none}
@media(max-width:500px){
  .ss{border-right:none;border-bottom:1px solid #1e293b;padding:12px 0}
  .ss:last-child{border-bottom:none}
}
.ss-lbl{font-size:9px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:6px}
.ss-val{font-size:22px;font-weight:800;color:#fff;line-height:1}
.ss-val.green{color:#10b981}.ss-val.amber{color:#f59e0b}
.ss-val.orange{color:#f97316}.ss-val.red{color:#f43f5e}
.ss-sub{font-size:11px;color:#475569;margin-top:4px}

/* UK NOTE */
.uk-note{margin-top:16px;background:rgba(56,189,248,0.06);
  border:1px solid rgba(56,189,248,0.15);border-radius:12px;
  padding:16px 20px;font-size:13px;color:#7dd3fc;line-height:1.8}
.uk-note strong{color:#38bdf8}
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="shell">

  <div class="header">
    <div class="hbadge">UK Investor &middot; Global Equities</div>
    <h1>Invest with <span class="hi">Confidence</span></h1>
    <p>Type any company name or ticker &mdash; screened against 10 criteria instantly.</p>
  </div>

  <div class="search-outer">
    <div class="search-box">
      <input id="si" type="text"
        placeholder="Apple,  Unilever,  Novo Nordisk,  MSFT,  ASML.AS ..."
        autocomplete="off" spellcheck="false"/>
      <button class="btn-go" id="sb" onclick="go()">SCREEN &rarr;</button>
    </div>
    <div class="ac" id="dd"></div>
  </div>

  <div class="examples">
    <span class="ex-lbl">Try:</span>
    <span class="chip" onclick="qs('Apple')">Apple</span>
    <span class="chip" onclick="qs('Microsoft')">Microsoft</span>
    <span class="chip" onclick="qs('Unilever')">Unilever</span>
    <span class="chip" onclick="qs('ASML')">ASML</span>
    <span class="chip" onclick="qs('Novo Nordisk')">Novo Nordisk</span>
    <span class="chip" onclick="qs('Visa')">Visa</span>
    <span class="chip" onclick="qs('AstraZeneca')">AstraZeneca</span>
    <span class="chip" onclick="qs('Mastercard')">Mastercard</span>
  </div>

  <div class="loader" id="ld"><div class="spin"></div><p>Analysing ...</p></div>
  <div class="err-box" id="eb"></div>

  <div id="results">
    <div class="res-banner" id="rb"></div>
    <div class="stock-card">
      <div class="smeta">
        <div class="s-ticker" id="r-ticker"></div>
        <div class="s-name"   id="r-name"></div>
        <div class="s-tags"   id="r-tags"></div>
      </div>
      <div class="ring-wrap">
        <div class="ring-rel">
          <svg class="ring-svg" viewBox="0 0 100 100" width="100" height="100">
            <circle class="r-bg"   cx="50" cy="50" r="45"/>
            <circle class="r-fill" cx="50" cy="50" r="45" id="rf"/>
          </svg>
          <div class="ring-inner">
            <div class="ring-num" id="r-score"></div>
            <div class="ring-lbl">CRITERIA</div>
          </div>
        </div>
        <div class="vp" id="r-verdict"></div>
      </div>
    </div>

    <div class="sec-row">
      <span class="sec-title">10 Screening Criteria</span>
      <div class="sec-line"></div>
    </div>
    <div class="grid" id="cg"></div>
    <div class="summary" id="sm"></div>
    <div class="uk-note" id="uk"></div>
  </div>

</div>
<script>
var inp=document.getElementById('si'),btn=document.getElementById('sb'),
    drop=document.getElementById('dd'),ld=document.getElementById('ld'),
    eb=document.getElementById('eb'),res=document.getElementById('results');
var timer=null,items=[],idx=-1,sel=null;

inp.addEventListener('input',function(){
  sel=null;clearTimeout(timer);
  var q=inp.value.trim();
  if(q.length<2){closeAc();return;}
  drop.innerHTML='<div class="ac-msg">Searching...</div>';
  drop.classList.add('open');
  timer=setTimeout(function(){
    fetch('/api/search?q='+encodeURIComponent(q))
      .then(function(r){return r.json();})
      .then(function(d){items=d;renderAc();})
      .catch(function(){closeAc();});
  },280);
});

inp.addEventListener('keydown',function(e){
  if(e.key==='ArrowDown'){e.preventDefault();mv(1);}
  else if(e.key==='ArrowUp'){e.preventDefault();mv(-1);}
  else if(e.key==='Enter'){e.preventDefault();idx>=0?pick(idx):go();}
  else if(e.key==='Escape'){closeAc();}
});
document.addEventListener('click',function(e){if(!e.target.closest('.search-outer'))closeAc();});

function renderAc(){
  if(!items.length){drop.innerHTML='<div class="ac-msg">No results found</div>';return;}
  drop.innerHTML=items.map(function(it,i){
    return '<div class="ac-row" onclick="pick('+i+')">'
      +'<span class="ac-sym">'+it.symbol+'</span>'
      +'<span class="ac-nm">'+it.name+'</span>'
      +'<span class="ac-ex">'+it.exchange+'</span></div>';
  }).join('');idx=-1;
}
function mv(dir){
  var els=drop.querySelectorAll('.ac-row');if(!els.length)return;
  els.forEach(function(e){e.classList.remove('hi');});
  idx=(idx+dir+els.length)%els.length;els[idx].classList.add('hi');
}
function pick(i){var it=items[i];if(!it)return;sel=it.symbol;inp.value=it.name;closeAc();go();}
function closeAc(){drop.classList.remove('open');drop.innerHTML='';idx=-1;}
function qs(name){inp.value=name;sel=null;go();}

function setLoad(on){
  btn.disabled=on;btn.innerHTML=on?'...':'SCREEN &rarr;';
  ld.classList.toggle('on',on);eb.classList.remove('on');
  if(on)res.classList.remove('on');
}

async function go(){
  closeAc();
  var q=(sel||inp.value).trim();
  if(!q){inp.focus();return;}
  setLoad(true);
  try{
    var r=await fetch('/api/screen?q='+encodeURIComponent(q));
    var d=await r.json();
    if(!r.ok||d.error)throw new Error(d.error||'Unknown error');
    render(d);
  }catch(e){
    eb.textContent='Error: '+e.message;eb.classList.add('on');
  }finally{setLoad(false);}
}

var catMap={'Fundamentals':'F','Valuation':'V','Governance':'G','Technical':'T'};
function fmt(n){return Number(n).toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});}

function render(d){
  var vc=d.verdict_color,r=45,ci=2*Math.PI*r,off=ci-(ci*d.score_pct/100);

  var rb=document.getElementById('rb'),typed=inp.value.trim();
  if(typed&&typed.toUpperCase()!==d.ticker){
    rb.innerHTML='Resolved <strong>"'+typed+'"</strong> &rarr; <strong>'+d.ticker+'</strong> ('+d.name+')';
    rb.classList.add('on');
  }else{rb.classList.remove('on');}

  document.getElementById('r-ticker').textContent=d.ticker;
  document.getElementById('r-name').textContent=d.name;

  var tags='';
  if(d.sector)   tags+='<span class="s-tag">'+d.sector+'</span>';
  if(d.country)  tags+='<span class="s-tag">'+d.country+'</span>';
  if(d.exchange) tags+='<span class="s-tag">'+d.exchange+'</span>';
  tags+='<span class="s-tag price-tag">'+(d.price?d.currency+' '+fmt(d.price):'N/A')+'</span>';
  document.getElementById('r-tags').innerHTML=tags;

  var rf=document.getElementById('rf');
  rf.setAttribute('class','r-fill '+vc);
  rf.style.strokeDasharray=ci.toFixed(1);
  rf.style.strokeDashoffset=ci.toFixed(1);
  setTimeout(function(){rf.style.strokeDashoffset=off;},100);

  document.getElementById('r-score').textContent=d.passed+'/'+d.total;
  var vEl=document.getElementById('r-verdict');
  vEl.textContent=d.verdict;vEl.className='vp '+vc;

  var cg=document.getElementById('cg');cg.innerHTML='';
  d.criteria.forEach(function(c,i){
    var el=document.createElement('div');
    el.className='crd '+(c.passed?'pass':'fail');
    var ck=catMap[c.category]||'F';
    el.innerHTML=
      '<div class="crd-num">#'+(i+1)+'</div>'+
      '<span class="cat-pill cat-'+ck+'">'+c.category+'</span>'+
      '<div class="crd-top">'+
        '<div class="crd-name">'+c.name+'</div>'+
        '<span class="badge '+(c.passed?'b-pass':'b-fail')+'">'+(c.passed?'&#10003; Pass':'&#10007; Fail')+'</span>'+
      '</div>'+
      '<div class="crd-val '+(c.passed?'pass':'fail')+'">'+c.value+'</div>'+
      '<div class="crd-thresh">Threshold: '+c.threshold+'</div>'+
      (c.note?'<div class="crd-note">'+c.note+'</div>':'');
    cg.appendChild(el);
    setTimeout(function(){el.classList.add('vis');},50+i*50);
  });

  var cats={};
  d.criteria.forEach(function(c){
    if(!cats[c.category])cats[c.category]={p:0,t:0};
    cats[c.category].t++;if(c.passed)cats[c.category].p++;
  });
  var smHTML=
    '<div class="ss"><div class="ss-lbl">Criteria Passed</div>'+
    '<div class="ss-val '+vc+'">'+d.passed+' / '+d.total+'</div>'+
    '<div class="ss-sub">'+d.score_pct+'% pass rate</div></div>'+
    '<div class="ss"><div class="ss-lbl">Verdict</div>'+
    '<div class="ss-val '+vc+'" style="font-size:15px;line-height:1.3">'+d.verdict+'</div></div>';
  Object.entries(cats).forEach(function(e){
    smHTML+='<div class="ss"><div class="ss-lbl">'+e[0]+'</div>'+
            '<div class="ss-val" style="font-size:18px">'+e[1].p+'/'+e[1].t+'</div></div>';
  });
  document.getElementById('sm').innerHTML=smHTML;

  var note='<strong>UK Investor Note &mdash; '+d.name+'</strong><br>';
  if(d.country==='United States')
    note+='<strong>Withholding Tax:</strong> Submit a <strong>W-8BEN</strong> form to your broker to reduce US dividend WHT from 30% to 15% (UK-US tax treaty). &nbsp;';
  else if(d.country==='United Kingdom')
    note+='<strong>Stamp Duty:</strong> 0.5% on UK share purchases. No WHT on UK dividends. &nbsp;';
  else
    note+='<strong>Withholding Tax:</strong> Check UK double-taxation treaty with <strong>'+d.country+'</strong> for WHT rate. &nbsp;';
  note+='Consider holding in your <strong>Trading 212 Stocks &amp; Shares ISA</strong> (&pound;20,000/year) to shelter gains and dividends from UK tax.';
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

if __name__ == "__main__":
    print("\n  GLOBAL STOCK SCREENER -- Starting...")
    print("  Open browser at:  http://localhost:5000")
    print("  Type any company name or ticker symbol\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
