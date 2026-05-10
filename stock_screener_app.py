"""
═══════════════════════════════════════════════════════════
  GLOBAL STOCK SCREENER — Single File App
  Run:  python stock_screener_app.py
  Open: http://localhost:5000
═══════════════════════════════════════════════════════════
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# HELPERS
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
# SCREENER
# ─────────────────────────────────────────────

def screen_stock(ticker):
    ticker = ticker.strip().upper()
    yf_ticker = yf.Ticker(ticker)

    info     = yf_ticker.info
    h10      = yf_ticker.history(period="10y", auto_adjust=True)
    h5       = yf_ticker.history(period="5y",  auto_adjust=True)
    h1       = yf_ticker.history(period="1y",  auto_adjust=True)
    income   = yf_ticker.financials
    balance  = yf_ticker.balance_sheet
    cashflow = yf_ticker.cashflow

    name     = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency", "USD")
    sector   = info.get("sector",   "—")
    country  = info.get("country",  "—")
    exchange = info.get("exchange", "—")
    price    = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    criteria = []

    # 1 — 10-Year Price Trend
    try:
        prices = h10["Close"].dropna()
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
                "< 10 years data", "> 10% p.a. CAGR", "Insufficient price history"))
    except Exception as e:
        criteria.append(criterion("10-Year Price Trend", "Technical", False, "Error", "> 10% p.a.", str(e)))

    # 2 — 5-Year Revenue Growth
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
                "> 10% p.a. CAGR + upward trend",
                f"Trend: {'Upward' if trend else 'Declining'}"
            ))
        else:
            criteria.append(criterion("5-Year Revenue Growth", "Fundamentals", False,
                "No data", "> 10% p.a. CAGR"))
    except Exception as e:
        criteria.append(criterion("5-Year Revenue Growth", "Fundamentals", False, "Error", "> 10% p.a.", str(e)))

    # 3 — 5-Year Profit Growth
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
                "> 10% p.a. CAGR + upward trend",
                f"Profitable throughout: {'Yes' if all_pos else 'No'} | Trend: {'Up' if trend else 'Down'}"
            ))
        else:
            criteria.append(criterion("5-Year Profit Growth", "Fundamentals", False,
                "No data", "> 10% p.a. CAGR"))
    except Exception as e:
        criteria.append(criterion("5-Year Profit Growth", "Fundamentals", False, "Error", "> 10% p.a.", str(e)))

    # 4 — ROE
    try:
        roe = info.get("returnOnEquity")
        if roe is not None:
            pct = round(roe * 100, 2)
            criteria.append(criterion(
                "Return on Equity (ROE)", "Fundamentals", pct >= 15.0,
                f"{pct}%", "> 15%",
                "Measures profitability per pound of shareholder equity"
            ))
        else:
            criteria.append(criterion("Return on Equity (ROE)", "Fundamentals", False, "N/A", "> 15%"))
    except Exception as e:
        criteria.append(criterion("Return on Equity (ROE)", "Fundamentals", False, "Error", "> 15%", str(e)))

    # 5 — ROCE
    try:
        ebit = row_vals(income, 1, "EBIT", "Operating Income") if not income.empty else []
        ta   = row_vals(balance, 1, "Total Assets") if not balance.empty else []
        cl   = row_vals(balance, 1, "Current Liabilities",
                        "Total Current Liabilities Net Minority Interest") if not balance.empty else []
        if ebit and ta and cl:
            ce   = ta[0] - cl[0]
            roce = round((ebit[0] / ce) * 100, 2) if ce > 0 else None
            criteria.append(criterion(
                "Return on Capital Employed (ROCE)", "Fundamentals",
                roce is not None and roce >= 15.0,
                f"{roce}%" if roce else "N/A", "> 15%",
                "EBIT divided by capital employed (Total Assets - Current Liabilities)"
            ))
        else:
            criteria.append(criterion("Return on Capital Employed (ROCE)", "Fundamentals", False,
                "No data", "> 15%"))
    except Exception as e:
        criteria.append(criterion("Return on Capital Employed (ROCE)", "Fundamentals", False, "Error", "> 15%", str(e)))

    # 6 — Debt-to-Equity
    try:
        de = info.get("debtToEquity")
        if de is not None:
            de_val = round(de / 100, 2)
            criteria.append(criterion(
                "Debt-to-Equity Ratio", "Fundamentals", de_val <= 0.8,
                f"{de_val}x", "< 0.8x",
                "Indicates manageable financial leverage"
            ))
        else:
            criteria.append(criterion("Debt-to-Equity Ratio", "Fundamentals", False, "N/A", "< 0.8x"))
    except Exception as e:
        criteria.append(criterion("Debt-to-Equity Ratio", "Fundamentals", False, "Error", "< 0.8x", str(e)))

    # 7 — Free Cash Flow
    ocf_vals   = []
    capex_vals = []
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
                "Free Cash Flow Trend", "Fundamentals", pos >= 4,
                f"{pos}/{n} positive years", ">= 4 of 5 years positive",
                "Profit must be backed by actual cash generation"
            ))
        else:
            criteria.append(criterion("Free Cash Flow Trend", "Fundamentals", False,
                "No data", ">= 4/5 years positive"))
    except Exception as e:
        criteria.append(criterion("Free Cash Flow Trend", "Fundamentals", False, "Error", ">= 4/5 years positive", str(e)))

    # 8 — Management Ownership
    try:
        ins = info.get("heldPercentInsiders")
        if ins is not None:
            pct = round(ins * 100, 2)
            criteria.append(criterion(
                "Management Ownership", "Governance", pct >= 5.0,
                f"{pct}%", "> 5% held by insiders",
                "Executives and directors with skin in the game"
            ))
        else:
            criteria.append(criterion("Management Ownership", "Governance", False, "N/A", "> 5%"))
    except Exception as e:
        criteria.append(criterion("Management Ownership", "Governance", False, "Error", "> 5%", str(e)))

    # 9 — Interest Coverage
    try:
        ebit_ic  = row_vals(income, 1, "EBIT", "Operating Income") if not income.empty else []
        interest = row_vals(income, 1, "Interest Expense", "Net Interest Income") if not income.empty else []
        if ebit_ic and interest and interest[0] != 0:
            ic = round(abs(ebit_ic[0] / interest[0]), 2)
            criteria.append(criterion(
                "Interest Coverage", "Fundamentals", ic >= 5.0,
                f"{ic}x", "> 5x (EBIT / Interest)",
                "Ability to service debt comfortably from operating profit"
            ))
        else:
            criteria.append(criterion(
                "Interest Coverage", "Fundamentals", True,
                "Debt-free / negligible interest", "> 5x",
                "No material interest expense detected"
            ))
    except Exception as e:
        criteria.append(criterion("Interest Coverage", "Fundamentals", False, "Error", "> 5x", str(e)))

    # 10 — Valuation (PEG or P/FCF)
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
            "PEG < 1.5 OR P/FCF < 25x",
            "At least one valuation metric must be reasonable"
        ))
    except Exception as e:
        criteria.append(criterion("Valuation", "Valuation", False, "Error", "PEG < 1.5 or P/FCF < 25x", str(e)))

    # ── Verdict ───────────────────────────────
    passed_count = sum(1 for c in criteria if c["passed"])
    total        = len(criteria)
    score_pct    = round((passed_count / total) * 100)

    if   score_pct >= 80: verdict, vc = "Strong Buy Candidate",      "green"
    elif score_pct >= 60: verdict, vc = "Potential Opportunity",      "amber"
    elif score_pct >= 40: verdict, vc = "Needs Further Research",     "orange"
    else:                 verdict, vc = "Does Not Meet Criteria",     "red"

    return {
        "ticker": ticker, "name": name, "sector": sector,
        "country": country, "exchange": exchange,
        "currency": currency, "price": price,
        "criteria": criteria,
        "passed": passed_count, "total": total,
        "score_pct": score_pct,
        "verdict": verdict, "verdict_color": vc,
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/api/screen")
def api_screen():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol"}), 400
    try:
        return jsonify(screen_stock(ticker))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to screen {ticker}: {str(e)}"}), 500

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
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Global Stock Screener</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#090e1a;--surface:#0d1526;--surface2:#111d35;
    --border:#1e2d4a;--border2:#243556;
    --text:#c8d8f0;--text-muted:#4a6080;--text-dim:#1e3050;
    --gold:#f0b942;--gold-dim:#7a5a18;
    --green:#2ecc8a;--green-dim:#0d3d26;
    --red:#f05252;--red-dim:#3d0f0f;
    --amber:#f0952a;--amber-dim:#3d2208;
    --blue:#4a9eff;--blue-dim:#0d2040;
    --orange:#f07a28;
    --mono:'Space Mono',monospace;
    --sans:'DM Sans',sans-serif;
  }
  html{scroll-behavior:smooth}
  body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
  body::before{
    content:'';position:fixed;inset:0;z-index:0;
    background-image:linear-gradient(rgba(74,158,255,.03) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(74,158,255,.03) 1px,transparent 1px);
    background-size:40px 40px;pointer-events:none
  }
  .shell{position:relative;z-index:1;max-width:900px;margin:0 auto;padding:40px 24px 80px}

  /* Header */
  .header{text-align:center;margin-bottom:44px}
  .header-tag{
    display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:3px;
    color:var(--blue);text-transform:uppercase;border:1px solid var(--blue-dim);
    padding:4px 12px;border-radius:2px;margin-bottom:18px
  }
  .header h1{font-family:var(--mono);font-size:clamp(22px,4vw,36px);font-weight:700;
    color:#fff;letter-spacing:-1px;line-height:1.1}
  .header h1 span{color:var(--gold)}
  .header p{margin-top:12px;color:var(--text-muted);font-size:14px;font-weight:300}

  /* Search */
  .search-wrap{
    display:flex;gap:12px;margin-bottom:16px;
    background:var(--surface);border:1px solid var(--border);
    border-radius:6px;padding:6px 6px 6px 20px;transition:border-color .2s
  }
  .search-wrap:focus-within{border-color:var(--blue)}
  .search-wrap input{
    flex:1;background:none;border:none;outline:none;
    font-family:var(--mono);font-size:18px;color:#fff;
    text-transform:uppercase;letter-spacing:2px
  }
  .search-wrap input::placeholder{color:var(--text-dim);letter-spacing:1px;font-size:13px}
  .btn-screen{
    background:var(--gold);color:#0a0800;font-family:var(--mono);
    font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
    border:none;border-radius:4px;padding:14px 28px;cursor:pointer;
    transition:opacity .15s,transform .1s;white-space:nowrap
  }
  .btn-screen:hover{opacity:.85}
  .btn-screen:active{transform:scale(.98)}
  .btn-screen:disabled{opacity:.4;cursor:not-allowed}

  /* Examples */
  .examples{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:36px;justify-content:center}
  .ex-chip{
    font-family:var(--mono);font-size:11px;color:var(--text-muted);
    border:1px solid var(--border2);padding:5px 12px;border-radius:3px;
    cursor:pointer;transition:color .15s,border-color .15s
  }
  .ex-chip:hover{color:var(--gold);border-color:var(--gold-dim)}

  /* Loader */
  .loader{display:none;text-align:center;padding:60px 0}
  .loader.active{display:block}
  .loader-ring{
    width:48px;height:48px;border:2px solid var(--border2);
    border-top-color:var(--gold);border-radius:50%;
    margin:0 auto 16px;animation:spin .8s linear infinite
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  .loader p{font-family:var(--mono);font-size:11px;color:var(--text-muted);letter-spacing:2px}

  /* Error */
  .error-box{
    display:none;background:var(--red-dim);border:1px solid var(--red);
    border-radius:6px;padding:18px 22px;font-family:var(--mono);
    font-size:13px;color:var(--red)
  }
  .error-box.active{display:block}

  /* Results */
  #results{display:none}
  #results.active{display:block;animation:fadeUp .4s ease}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

  /* Stock header */
  .stock-header{
    background:var(--surface);border:1px solid var(--border);
    border-radius:8px;padding:28px 32px;margin-bottom:24px;
    display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap
  }
  .stock-meta{flex:1;min-width:200px}
  .stock-ticker{font-family:var(--mono);font-size:28px;font-weight:700;color:var(--gold);letter-spacing:2px}
  .stock-name{font-size:15px;color:#fff;margin:4px 0 10px;font-weight:500}
  .stock-tags{display:flex;gap:8px;flex-wrap:wrap}
  .tag{
    font-family:var(--mono);font-size:10px;letter-spacing:1px;
    color:var(--text-muted);border:1px solid var(--border2);
    padding:3px 8px;border-radius:2px;text-transform:uppercase
  }

  /* Score ring */
  .score-ring-wrap{display:flex;flex-direction:column;align-items:center;gap:8px}
  .score-ring{position:relative;width:100px;height:100px}
  .score-ring svg{transform:rotate(-90deg)}
  .score-ring circle{fill:none;stroke-width:6}
  .ring-bg{stroke:var(--border2)}
  .ring-fill{stroke:var(--gold);stroke-linecap:round;stroke-dasharray:283;stroke-dashoffset:283;transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)}
  .ring-fill.green{stroke:var(--green)}
  .ring-fill.amber{stroke:var(--amber)}
  .ring-fill.orange{stroke:var(--orange)}
  .ring-fill.red{stroke:var(--red)}
  .ring-text{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:var(--mono)}
  .ring-score{font-size:22px;font-weight:700;color:#fff;line-height:1}
  .ring-label{font-size:9px;color:var(--text-muted);margin-top:2px;letter-spacing:1px}
  .verdict-text{font-family:var(--mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;text-align:center;max-width:110px}
  .verdict-text.green{color:var(--green)}
  .verdict-text.amber{color:var(--amber)}
  .verdict-text.orange{color:var(--orange)}
  .verdict-text.red{color:var(--red)}

  /* Criteria */
  .section-label{font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--text-muted);text-transform:uppercase;margin-bottom:14px}
  .criteria-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}
  @media(max-width:600px){.criteria-grid{grid-template-columns:1fr}}

  .criterion{
    background:var(--surface);border:1px solid var(--border);
    border-left:3px solid var(--border);border-radius:6px;padding:16px 18px;
    opacity:0;transform:translateY(8px);
    transition:opacity .3s ease,transform .3s ease;
    position:relative;overflow:hidden
  }
  .criterion.visible{opacity:1;transform:translateY(0)}
  .criterion.pass{border-left-color:var(--green)}
  .criterion.fail{border-left-color:var(--red)}
  .criterion.pass::before{
    content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(46,204,138,.04) 0%,transparent 60%);
    pointer-events:none
  }

  .crit-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
  .crit-name{font-size:13px;font-weight:600;color:#fff;line-height:1.3}
  .crit-badge{
    flex-shrink:0;font-family:var(--mono);font-size:10px;font-weight:700;
    letter-spacing:1px;padding:3px 8px;border-radius:2px;text-transform:uppercase
  }
  .badge-pass{background:var(--green-dim);color:var(--green)}
  .badge-fail{background:var(--red-dim);color:var(--red)}
  .crit-value{font-family:var(--mono);font-size:18px;font-weight:700;margin-bottom:4px}
  .crit-value.pass{color:var(--green)}
  .crit-value.fail{color:var(--red)}
  .crit-threshold{font-size:11px;color:var(--text-muted);margin-bottom:6px}
  .crit-note{font-size:11px;color:var(--text-muted);font-style:italic;line-height:1.4}
  .crit-num{position:absolute;top:12px;right:14px;font-family:var(--mono);font-size:11px;color:var(--text-dim)}

  /* Summary */
  .summary-bar{
    background:var(--surface);border:1px solid var(--border);
    border-radius:8px;padding:24px 28px;
    display:flex;gap:20px;align-items:center;flex-wrap:wrap
  }
  .summary-stat{flex:1;min-width:110px}
  .summary-stat .s-label{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--text-muted);text-transform:uppercase;margin-bottom:4px}
  .summary-stat .s-value{font-family:var(--mono);font-size:22px;font-weight:700;color:#fff}
  .summary-stat .s-value.green{color:var(--green)}
  .summary-stat .s-value.amber{color:var(--amber)}
  .summary-stat .s-value.orange{color:var(--orange)}
  .summary-stat .s-value.red{color:var(--red)}
  .summary-stat .s-sub{font-size:12px;color:var(--text-muted);margin-top:2px}
  .s-divider{width:1px;background:var(--border);align-self:stretch;flex-shrink:0}

  /* UK Note */
  .uk-note{
    margin-top:16px;background:var(--blue-dim);
    border:1px solid rgba(74,158,255,.15);border-radius:6px;
    padding:14px 18px;font-size:12px;color:rgba(74,158,255,.8);line-height:1.7
  }
  .uk-note strong{color:var(--blue)}
</style>
</head>
<body>
<div class="shell">

  <div class="header">
    <div class="header-tag">UK GBP &middot; Global Equities</div>
    <h1>STOCK <span>SCREENER</span></h1>
    <p>Enter any ticker &mdash; assessed against 10 criteria &mdash; make your call.</p>
  </div>

  <div class="search-wrap">
    <input id="tickerInput" type="text"
           placeholder="Ticker e.g. AAPL  ULVR.L  ASML.AS  NVO  MSFT"
           autocomplete="off" spellcheck="false" maxlength="12" />
    <button class="btn-screen" id="screenBtn" onclick="runScreen()">SCREEN &rarr;</button>
  </div>

  <div class="examples">
    <span class="ex-chip" onclick="quickLoad('MSFT')">MSFT</span>
    <span class="ex-chip" onclick="quickLoad('AAPL')">AAPL</span>
    <span class="ex-chip" onclick="quickLoad('ULVR.L')">ULVR.L</span>
    <span class="ex-chip" onclick="quickLoad('ASML.AS')">ASML.AS</span>
    <span class="ex-chip" onclick="quickLoad('NVO')">NVO</span>
    <span class="ex-chip" onclick="quickLoad('V')">V</span>
    <span class="ex-chip" onclick="quickLoad('MA')">MA</span>
    <span class="ex-chip" onclick="quickLoad('AZN.L')">AZN.L</span>
  </div>

  <div class="loader" id="loader">
    <div class="loader-ring"></div>
    <p>ANALYSING&hellip;</p>
  </div>

  <div class="error-box" id="errorBox"></div>

  <div id="results">
    <div class="stock-header" id="stockHeader"></div>
    <div class="section-label" style="margin-top:8px">10 SCREENING CRITERIA</div>
    <div class="criteria-grid" id="criteriaGrid"></div>
    <div class="summary-bar" id="summaryBar"></div>
    <div class="uk-note" id="ukNote"></div>
  </div>

</div>
<script>
  const input   = document.getElementById('tickerInput');
  const btn     = document.getElementById('screenBtn');
  const loader  = document.getElementById('loader');
  const errBox  = document.getElementById('errorBox');
  const results = document.getElementById('results');

  input.addEventListener('keydown', e => { if (e.key === 'Enter') runScreen(); });

  function quickLoad(t) { input.value = t; runScreen(); }

  function setLoading(on) {
    btn.disabled = on;
    btn.innerHTML = on ? '&hellip;' : 'SCREEN &rarr;';
    loader.classList.toggle('active', on);
    errBox.classList.remove('active');
    if (on) results.classList.remove('active');
  }

  async function runScreen() {
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) { input.focus(); return; }
    setLoading(true);
    try {
      const res  = await fetch('/api/screen?ticker=' + encodeURIComponent(ticker));
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Unknown error');
      renderResults(data);
    } catch(err) {
      errBox.textContent = 'Error: ' + err.message;
      errBox.classList.add('active');
    } finally {
      setLoading(false);
    }
  }

  function fmt(n, decimals=2) {
    return Number(n).toLocaleString('en-GB', {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  }

  function renderResults(d) {
    const vc     = d.verdict_color;
    const offset = 283 - (283 * d.score_pct / 100);
    const priceStr = d.price
      ? d.currency + ' ' + fmt(d.price)
      : 'N/A';

    // Stock header
    document.getElementById('stockHeader').innerHTML =
      '<div class="stock-meta">' +
        '<div class="stock-ticker">' + d.ticker + '</div>' +
        '<div class="stock-name">' + d.name + '</div>' +
        '<div class="stock-tags">' +
          '<span class="tag">' + d.sector + '</span>' +
          '<span class="tag">' + d.country + '</span>' +
          '<span class="tag">' + d.exchange + '</span>' +
          '<span class="tag">Price: ' + priceStr + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="score-ring-wrap">' +
        '<div class="score-ring">' +
          '<svg viewBox="0 0 100 100" width="100" height="100">' +
            '<circle class="ring-bg" cx="50" cy="50" r="45"/>' +
            '<circle class="ring-fill ' + vc + '" cx="50" cy="50" r="45" id="ringFill" style="stroke-dashoffset:283"/>' +
          '</svg>' +
          '<div class="ring-text">' +
            '<div class="ring-score">' + d.passed + '/' + d.total + '</div>' +
            '<div class="ring-label">CRITERIA</div>' +
          '</div>' +
        '</div>' +
        '<div class="verdict-text ' + vc + '">' + d.verdict + '</div>' +
      '</div>';

    setTimeout(function() {
      var fill = document.getElementById('ringFill');
      if (fill) fill.style.strokeDashoffset = offset;
    }, 100);

    // Criteria cards
    const grid = document.getElementById('criteriaGrid');
    grid.innerHTML = '';
    d.criteria.forEach(function(c, i) {
      const card = document.createElement('div');
      card.className = 'criterion ' + (c.passed ? 'pass' : 'fail');
      card.innerHTML =
        '<div class="crit-num">#' + (i+1) + '</div>' +
        '<div class="crit-top">' +
          '<div class="crit-name">' + c.name + '</div>' +
          '<span class="crit-badge ' + (c.passed ? 'badge-pass' : 'badge-fail') + '">' +
            (c.passed ? '&#10003; PASS' : '&#10007; FAIL') +
          '</span>' +
        '</div>' +
        '<div class="crit-value ' + (c.passed ? 'pass' : 'fail') + '">' + c.value + '</div>' +
        '<div class="crit-threshold">Threshold: ' + c.threshold + '</div>' +
        (c.note ? '<div class="crit-note">' + c.note + '</div>' : '');
      grid.appendChild(card);
      setTimeout(function() { card.classList.add('visible'); }, 60 + i * 55);
    });

    // Summary
    const cats = {};
    d.criteria.forEach(function(c) {
      if (!cats[c.category]) cats[c.category] = {pass:0,total:0};
      cats[c.category].total++;
      if (c.passed) cats[c.category].pass++;
    });
    let catHtml = '';
    Object.entries(cats).forEach(function(entry) {
      const cat = entry[0], g = entry[1];
      catHtml += '<div class="s-divider"></div><div class="summary-stat">' +
        '<div class="s-label">' + cat + '</div>' +
        '<div class="s-value" style="font-size:18px">' + g.pass + '/' + g.total + '</div>' +
        '</div>';
    });

    document.getElementById('summaryBar').innerHTML =
      '<div class="summary-stat">' +
        '<div class="s-label">Criteria Passed</div>' +
        '<div class="s-value ' + vc + '">' + d.passed + ' / ' + d.total + '</div>' +
        '<div class="s-sub">' + d.score_pct + '% pass rate</div>' +
      '</div>' +
      '<div class="s-divider"></div>' +
      '<div class="summary-stat">' +
        '<div class="s-label">Verdict</div>' +
        '<div class="s-value ' + vc + '" style="font-size:14px;line-height:1.3">' + d.verdict + '</div>' +
      '</div>' +
      catHtml;

    // UK note
    let note = '<strong>UK Investor Notes</strong><br>';
    if (d.country === 'United States') {
      note += '<strong>Withholding Tax:</strong> Submit <strong>W-8BEN</strong> form to your broker to reduce US dividend WHT from 30% to 15% (UK-US tax treaty). &nbsp;';
    } else if (d.country === 'United Kingdom') {
      note += '<strong>Stamp Duty:</strong> 0.5% applies on UK share purchases. No WHT on UK dividends. &nbsp;';
    } else {
      note += '<strong>Withholding Tax:</strong> Check UK double-taxation treaty with <strong>' + d.country + '</strong> for dividend WHT rate. &nbsp;';
    }
    note += 'Consider holding in your <strong>Trading 212 Stocks &amp; Shares ISA</strong> (GBP 20k/year allowance) to shelter capital gains and dividends from UK tax.';
    document.getElementById('ukNote').innerHTML = note;

    results.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

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
