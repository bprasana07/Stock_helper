"""
=============================================================
  screener.py — Five screening layers
  Layer 1: Fundamentals      (ROE, ROCE, D/E, FCF, etc.)
  Layer 2: Valuation         (P/E, P/FCF, PEG, EV/EBITDA, DCF)
  Layer 3: Governance        (Insider ownership, buybacks, dividends)
  Layer 4: Technical         (10Y trend, 200DMA, RSI, Beta)
  Layer 5: UK Investor Risk  (WHT, FX, ISA, stamp duty, country risk)
=============================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from config import THRESHOLDS, WHT_RATES, LOW_RISK_COUNTRIES, FX_DRAG_VS_GBP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def safe_cagr(start: float, end: float, years: float) -> Optional[float]:
    """Compute compound annual growth rate (%)."""
    try:
        if start <= 0 or end <= 0 or years <= 0:
            return None
        return ((end / start) ** (1.0 / years) - 1) * 100
    except Exception:
        return None


def get_row(df: pd.DataFrame, *candidates) -> Optional[pd.Series]:
    """Return the first matching row from a DataFrame by trying multiple label candidates."""
    for label in candidates:
        if label in df.index:
            return df.loc[label]
    return None


def row_values(df: pd.DataFrame, n: int, *candidates) -> list:
    """Extract the last n numeric values from a financial statement row."""
    row = get_row(df, *candidates)
    if row is None:
        return []
    return [float(v) for v in row.iloc[:n] if pd.notna(v) and v != 0]


def is_upward_trend(values: list) -> bool:
    """
    Return True if the linear regression slope through the values is positive.
    Values should be in chronological order (oldest first).
    """
    if len(values) < 2:
        return False
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return float(slope) > 0


def calc_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Compute 14-day RSI from a price series."""
    try:
        delta = prices.diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        return float(rsi_series.dropna().iloc[-1])
    except Exception:
        return None


def _pass(value, passes: bool, note: str = "") -> dict:
    """Standardised metric result dict."""
    return {"value": value, "pass": passes, "note": note}


# ─────────────────────────────────────────────────────────────
# LAYER 1 — FUNDAMENTALS
# ─────────────────────────────────────────────────────────────

def screen_fundamentals(data: dict) -> dict:
    """
    Checks: ROE, ROCE, ROIC, D/E, Revenue CAGR, Profit CAGR,
            FCF trend, OCF/NI ratio, Interest Coverage,
            Gross Margin trend.
    """
    scores = {}
    info    = data.get("info", {})
    income  = data.get("income_stmt",   pd.DataFrame())
    balance = data.get("balance_sheet", pd.DataFrame())
    cashflow= data.get("cashflow",      pd.DataFrame())

    # ── ROE ────────────────────────────────────────────────
    roe = info.get("returnOnEquity")
    if roe is not None:
        roe_pct = roe * 100
        scores["roe"] = _pass(f"{roe_pct:.1f}%", roe_pct >= THRESHOLDS["roe_min"])

    # ── ROCE (EBIT / Capital Employed) ──────────────────────
    if not income.empty and not balance.empty:
        try:
            ebit_row   = row_values(income, 1, "EBIT", "Operating Income")
            ta_row     = row_values(balance, 1, "Total Assets")
            cl_row     = row_values(balance, 1, "Current Liabilities",
                                    "Total Current Liabilities Net Minority Interest")
            if ebit_row and ta_row and cl_row:
                cap_employed = ta_row[0] - cl_row[0]
                if cap_employed > 0:
                    roce = (ebit_row[0] / cap_employed) * 100
                    scores["roce"] = _pass(f"{roce:.1f}%", roce >= THRESHOLDS["roce_min"])
        except Exception as e:
            logger.debug(f"ROCE error: {e}")

    # ── ROIC (from FMP ratios) ───────────────────────────────
    fmp_ratios = data.get("fmp_ratios")
    if fmp_ratios and isinstance(fmp_ratios, list) and len(fmp_ratios) > 0:
        roic = fmp_ratios[0].get("returnOnCapitalEmployed") or fmp_ratios[0].get("roic")
        if roic:
            roic_pct = roic * 100
            scores["roic"] = _pass(f"{roic_pct:.1f}%", roic_pct >= THRESHOLDS["roic_min"])

    # ── DEBT-TO-EQUITY ───────────────────────────────────────
    de = info.get("debtToEquity")
    if de is not None:
        de_val = de / 100  # yfinance gives as percentage
        scores["debt_to_equity"] = _pass(f"{de_val:.2f}x", de_val <= THRESHOLDS["de_max"])

    # ── REVENUE CAGR (5Y) ────────────────────────────────────
    if not income.empty:
        rev = row_values(income, 5, "Total Revenue")
        if len(rev) >= 2:
            rev_chron = list(reversed(rev))  # oldest → newest
            cagr = safe_cagr(rev_chron[0], rev_chron[-1], len(rev_chron) - 1)
            if cagr is not None:
                trend_ok = is_upward_trend(rev_chron)
                scores["revenue_cagr_5y"] = _pass(
                    f"{cagr:.1f}%",
                    cagr >= THRESHOLDS["revenue_cagr_5y_min"] and trend_ok,
                    "Trend check included"
                )

    # ── NET PROFIT CAGR (5Y) ─────────────────────────────────
    if not income.empty:
        ni = row_values(income, 5, "Net Income")
        if len(ni) >= 2:
            ni_chron = list(reversed(ni))
            if all(v > 0 for v in ni_chron):  # must be profitable throughout
                cagr = safe_cagr(ni_chron[0], ni_chron[-1], len(ni_chron) - 1)
                if cagr is not None:
                    scores["profit_cagr_5y"] = _pass(
                        f"{cagr:.1f}%",
                        cagr >= THRESHOLDS["profit_cagr_5y_min"] and is_upward_trend(ni_chron)
                    )

    # ── FREE CASH FLOW TREND ─────────────────────────────────
    if not cashflow.empty:
        try:
            ocf_row   = row_values(cashflow, 5, "Operating Cash Flow",
                                   "Cash Flow From Continuing Operating Activities")
            capex_row = row_values(cashflow, 5, "Capital Expenditure",
                                   "Purchase Of Property Plant And Equipment")
            if ocf_row and capex_row:
                n       = min(len(ocf_row), len(capex_row))
                fcf     = [ocf_row[i] + capex_row[i] for i in range(n)]  # capex is negative
                pos_yrs = sum(1 for f in fcf if f > 0)
                fcf_chron = list(reversed(fcf))
                scores["fcf_trend"] = _pass(
                    f"{pos_yrs}/{n} positive years",
                    pos_yrs >= THRESHOLDS["fcf_positive_years_min"] and is_upward_trend(fcf_chron)
                )

                # OCF / Net Income (earnings quality)
                ni = row_values(income, 1, "Net Income") if not income.empty else []
                if ni and ni[0] > 0 and ocf_row:
                    ratio = ocf_row[0] / ni[0]
                    scores["ocf_ni_ratio"] = _pass(
                        f"{ratio:.2f}x",
                        ratio >= THRESHOLDS["ocf_ni_ratio_min"],
                        "> 1.0 means cash earnings match reported profit"
                    )
        except Exception as e:
            logger.debug(f"FCF error: {e}")

    # ── INTEREST COVERAGE ────────────────────────────────────
    if not income.empty:
        try:
            ebit     = row_values(income, 1, "EBIT", "Operating Income")
            interest = row_values(income, 1, "Interest Expense",
                                  "Net Interest Income")
            if ebit and interest and interest[0] != 0:
                ic = abs(ebit[0] / interest[0])
                scores["interest_coverage"] = _pass(
                    f"{ic:.1f}x",
                    ic >= THRESHOLDS["interest_coverage_min"]
                )
        except Exception as e:
            logger.debug(f"Interest coverage error: {e}")

    # ── GROSS MARGIN TREND ───────────────────────────────────
    if not income.empty:
        try:
            gp  = row_values(income, 5, "Gross Profit")
            rev = row_values(income, 5, "Total Revenue")
            if gp and rev and len(gp) == len(rev):
                margins = [g / r for g, r in zip(gp, rev)]
                latest_gm = margins[0] * 100
                scores["gross_margin_trend"] = _pass(
                    f"{latest_gm:.1f}% (latest)",
                    is_upward_trend(list(reversed(margins)))
                )
        except Exception as e:
            logger.debug(f"Gross margin error: {e}")

    # ── RECEIVABLES CHECK (earnings quality) ─────────────────
    if not income.empty and not balance.empty:
        try:
            rev  = row_values(income,  2, "Total Revenue")
            recv = row_values(balance, 2, "Receivables",
                              "Net Receivables", "Accounts Receivable")
            if len(rev) == 2 and len(recv) == 2:
                rev_growth  = (rev[0]  - rev[1])  / abs(rev[1])
                recv_growth = (recv[0] - recv[1]) / abs(recv[1])
                scores["receivables_quality"] = _pass(
                    f"Rev +{rev_growth*100:.1f}% | Recv +{recv_growth*100:.1f}%",
                    recv_growth <= rev_growth + 0.05,  # receivables not growing faster than revenue
                    "Red flag if receivables grow faster than revenue"
                )
        except Exception as e:
            logger.debug(f"Receivables check error: {e}")

    return scores


# ─────────────────────────────────────────────────────────────
# LAYER 2 — VALUATION
# ─────────────────────────────────────────────────────────────

def screen_valuation(data: dict) -> dict:
    """
    Checks: Trailing P/E, Forward P/E, P/FCF, PEG, EV/EBITDA,
            Price-to-Book, DCF margin of safety.
    """
    scores   = {}
    info     = data.get("info", {})
    cashflow = data.get("cashflow", pd.DataFrame())

    # ── TRAILING P/E ─────────────────────────────────────────
    pe = info.get("trailingPE")
    if pe and pe > 0:
        scores["trailing_pe"] = _pass(f"{pe:.1f}x", pe <= THRESHOLDS["pe_max"])

    # ── FORWARD P/E ──────────────────────────────────────────
    fpe = info.get("forwardPE")
    if fpe and fpe > 0:
        scores["forward_pe"] = _pass(f"{fpe:.1f}x", fpe <= THRESHOLDS["forward_pe_max"])

    # ── PRICE / FREE CASH FLOW ───────────────────────────────
    mktcap = info.get("marketCap")
    if mktcap and not cashflow.empty:
        try:
            ocf   = row_values(cashflow, 1, "Operating Cash Flow",
                               "Cash Flow From Continuing Operating Activities")
            capex = row_values(cashflow, 1, "Capital Expenditure",
                               "Purchase Of Property Plant And Equipment")
            if ocf and capex:
                fcf = ocf[0] + capex[0]
                if fcf > 0:
                    pfcf = mktcap / fcf
                    scores["price_to_fcf"] = _pass(f"{pfcf:.1f}x", pfcf <= THRESHOLDS["pfcf_max"])
        except Exception as e:
            logger.debug(f"P/FCF error: {e}")

    # ── PEG RATIO ────────────────────────────────────────────
    peg = info.get("pegRatio")
    if peg and peg > 0:
        scores["peg_ratio"] = _pass(f"{peg:.2f}", peg <= THRESHOLDS["peg_max"])

    # ── EV / EBITDA ──────────────────────────────────────────
    ev_ebitda = info.get("enterpriseToEbitda")
    if ev_ebitda and ev_ebitda > 0:
        scores["ev_ebitda"] = _pass(f"{ev_ebitda:.1f}x", ev_ebitda <= THRESHOLDS["ev_ebitda_max"])

    # ── PRICE TO BOOK ────────────────────────────────────────
    pb = info.get("priceToBook")
    if pb and pb > 0:
        scores["price_to_book"] = _pass(f"{pb:.2f}x", pb <= THRESHOLDS["pb_max"])

    # ── DCF INTRINSIC VALUE (from FMP) ───────────────────────
    dcf_data = data.get("fmp_dcf")
    if dcf_data and isinstance(dcf_data, list) and len(dcf_data) > 0:
        try:
            dcf_val = float(dcf_data[0].get("dcf", 0))
            price   = float(dcf_data[0].get("Stock Price", 0))
            if dcf_val > 0 and price > 0:
                mos = (dcf_val - price) / price * 100
                scores["dcf_margin_of_safety"] = _pass(
                    f"{mos:.1f}% {'below' if mos < 0 else 'above'} DCF",
                    mos >= THRESHOLDS["dcf_margin_of_safety_min"]
                )
        except Exception as e:
            logger.debug(f"DCF error: {e}")

    # ── EPS CONSISTENCY (informational) ──────────────────────
    eps_ttm = info.get("trailingEps")
    eps_fwd = info.get("forwardEps")
    if eps_ttm and eps_fwd and eps_ttm > 0:
        eps_growth_implied = (eps_fwd - eps_ttm) / eps_ttm * 100
        scores["eps_growth_expected"] = _pass(
            f"{eps_growth_implied:.1f}% YoY",
            eps_growth_implied >= 5
        )

    return scores


# ─────────────────────────────────────────────────────────────
# LAYER 3 — GOVERNANCE
# ─────────────────────────────────────────────────────────────

def screen_governance(data: dict) -> dict:
    """
    Checks: Insider ownership %, institutional ownership,
            net insider buying, share buybacks, dividend consistency,
            short interest.
    """
    scores   = {}
    info     = data.get("info", {})
    cashflow = data.get("cashflow", pd.DataFrame())

    # ── INSIDER OWNERSHIP ─────────────────────────────────────
    insider_pct = info.get("heldPercentInsiders")
    if insider_pct is not None:
        pct = insider_pct * 100
        scores["insider_ownership"] = _pass(
            f"{pct:.2f}%",
            pct >= THRESHOLDS["insider_ownership_min"],
            "Skin in the game — executives & directors holding shares"
        )

    # ── INSTITUTIONAL OWNERSHIP ───────────────────────────────
    inst_pct = info.get("heldPercentInstitutions")
    if inst_pct is not None:
        pct = inst_pct * 100
        scores["institutional_ownership"] = _pass(
            f"{pct:.1f}%",
            THRESHOLDS["inst_ownership_min"] <= pct <= THRESHOLDS["inst_ownership_max"],
            "40–85% = healthy; < 40% suggests lack of confidence, > 85% = overcrowded"
        )

    # ── SHORT INTEREST ────────────────────────────────────────
    short_ratio = info.get("shortRatio")
    if short_ratio is not None:
        scores["short_ratio"] = _pass(
            f"{short_ratio:.1f} days",
            short_ratio <= THRESHOLDS["short_ratio_max"],
            "Days to cover short positions; > 5 days = elevated short interest"
        )

    # ── INSIDER TRANSACTIONS (FMP) ────────────────────────────
    insider_data = data.get("fmp_insider")
    if insider_data and isinstance(insider_data, list):
        buys  = sum(1 for t in insider_data
                    if str(t.get("transactionType", "")).upper() in ["P-PURCHASE", "BUY", "ACQUISITION"])
        sells = sum(1 for t in insider_data
                    if str(t.get("transactionType", "")).upper() in ["S-SALE", "SELL", "DISPOSITION"])
        net   = buys - sells
        scores["insider_net_buying"] = _pass(
            f"↑{buys} buys / ↓{sells} sells (net: {net:+d})",
            net >= THRESHOLDS["net_insider_buys_min"]
        )

    # ── SHARE BUYBACKS ────────────────────────────────────────
    if not cashflow.empty:
        try:
            buyback_row = row_values(cashflow, 5,
                                     "Repurchase Of Capital Stock",
                                     "Common Stock Repurchased",
                                     "Purchase Of Business")
            if buyback_row:
                buyback_years = sum(1 for v in buyback_row if v < 0)  # negative = cash outflow
                scores["buyback_history"] = _pass(
                    f"{buyback_years}/{len(buyback_row)} years with buybacks",
                    buyback_years >= THRESHOLDS["buyback_years_min"]
                )
        except Exception as e:
            logger.debug(f"Buyback error: {e}")

    # ── DIVIDEND CONSISTENCY ─────────────────────────────────
    div_yield     = info.get("dividendYield", 0) or 0
    five_yr_div   = info.get("fiveYearAvgDividendYield", 0) or 0
    payout_ratio  = info.get("payoutRatio", 0) or 0
    if five_yr_div > 0:
        scores["dividend_consistency"] = _pass(
            f"Yield: {div_yield*100:.2f}% | 5Y avg: {five_yr_div:.2f}% | Payout: {payout_ratio*100:.0f}%",
            five_yr_div > 0 and (payout_ratio == 0 or payout_ratio < 0.75),
            "5Y dividend history with sustainable payout ratio (< 75%)"
        )

    return scores


# ─────────────────────────────────────────────────────────────
# LAYER 4 — TECHNICAL
# ─────────────────────────────────────────────────────────────

def screen_technical(data: dict) -> dict:
    """
    Checks: 10Y price CAGR, 5Y price CAGR, 200-DMA (price above + slope),
            RSI, Beta, 52-week proximity, volume trend.
    """
    scores  = {}
    info    = data.get("info", {})
    h10     = data.get("hist_10y", pd.DataFrame())
    h5      = data.get("hist_5y",  pd.DataFrame())
    h1      = data.get("hist_1y",  pd.DataFrame())

    if h10.empty:
        return scores

    prices_10 = h10["Close"]
    prices_1  = h1["Close"] if not h1.empty else pd.Series(dtype=float)

    # ── 10Y PRICE CAGR ───────────────────────────────────────
    if len(prices_10) >= 252 * 8:
        years = len(prices_10) / 252
        cagr  = safe_cagr(float(prices_10.iloc[0]), float(prices_10.iloc[-1]), years)
        if cagr is not None:
            scores["price_cagr_10y"] = _pass(
                f"{cagr:.1f}% p.a.",
                cagr >= THRESHOLDS["price_cagr_10y_min"]
            )

    # ── 5Y PRICE CAGR ────────────────────────────────────────
    if not h5.empty:
        prices_5 = h5["Close"]
        if len(prices_5) >= 252 * 4:
            cagr5 = safe_cagr(float(prices_5.iloc[0]), float(prices_5.iloc[-1]), 5)
            if cagr5 is not None:
                scores["price_cagr_5y"] = _pass(
                    f"{cagr5:.1f}% p.a.",
                    cagr5 >= THRESHOLDS["price_cagr_5y_min"]
                )

    # ── 200-DAY MOVING AVERAGE ───────────────────────────────
    if len(prices_1) >= 200:
        ma200         = prices_1.rolling(200).mean()
        current_price = float(prices_1.iloc[-1])
        ma200_now     = float(ma200.iloc[-1])
        ma200_20d_ago = float(ma200.iloc[-21]) if len(ma200) >= 21 else ma200_now

        pct_above = (current_price / ma200_now - 1) * 100
        scores["price_above_200dma"] = _pass(
            f"{pct_above:+.1f}% vs 200DMA",
            current_price > ma200_now
        )
        scores["200dma_slope"] = _pass(
            f"200DMA {'↑ rising' if ma200_now > ma200_20d_ago else '↓ falling'}",
            ma200_now > ma200_20d_ago,
            "Positive slope = established uptrend"
        )

    # ── RSI (14-day) ─────────────────────────────────────────
    if len(prices_1) >= 20:
        rsi_val = calc_rsi(prices_1)
        if rsi_val is not None:
            scores["rsi_14d"] = _pass(
                f"{rsi_val:.1f}",
                rsi_val < THRESHOLDS["rsi_max"],
                "< 70 = not overbought at entry"
            )

    # ── BETA ─────────────────────────────────────────────────
    beta = info.get("beta")
    if beta is not None:
        scores["beta"] = _pass(f"{beta:.2f}", beta <= THRESHOLDS["beta_max"])

    # ── 52-WEEK HIGH PROXIMITY ───────────────────────────────
    w52_high = info.get("fiftyTwoWeekHigh")
    current  = info.get("currentPrice") or info.get("regularMarketPrice")
    if w52_high and current:
        pct_of_high = (current / w52_high) * 100
        scores["proximity_52w_high"] = _pass(
            f"{pct_of_high:.0f}% of 52W high",
            pct_of_high >= THRESHOLDS["pct_of_52w_high_min"],
            "Price should be within 25% of 52-week high"
        )

    # ── VOLUME TREND (30-day avg vs 90-day avg) ───────────────
    if not h1.empty and "Volume" in h1.columns:
        vol      = h1["Volume"].dropna()
        vol_30d  = vol.iloc[-30:].mean() if len(vol) >= 30 else None
        vol_90d  = vol.iloc[-90:].mean() if len(vol) >= 90 else None
        if vol_30d and vol_90d and vol_90d > 0:
            vol_ratio = vol_30d / vol_90d
            scores["volume_trend"] = _pass(
                f"30D avg {vol_ratio:.2f}x vs 90D",
                vol_ratio >= 0.9,
                "Healthy if recent volume not significantly below trend"
            )

    return scores


# ─────────────────────────────────────────────────────────────
# LAYER 5 — UK INVESTOR RISK
# ─────────────────────────────────────────────────────────────

def screen_uk_risk(data: dict) -> dict:
    """
    Checks: Withholding tax, GBP-adjusted return estimate,
            ISA eligibility, stamp duty, currency risk,
            country risk.
    """
    scores   = {}
    info     = data.get("info", {})
    country  = info.get("country", "Unknown")
    currency = info.get("currency", "USD")
    exchange = info.get("exchange", "")
    h10      = data.get("hist_10y", pd.DataFrame())

    # ── WITHHOLDING TAX ───────────────────────────────────────
    wht = WHT_RATES.get(country, 15.0)
    notes = {
        "United States":  "Submit W-8BEN to your broker to claim 15% treaty rate",
        "Switzerland":    "Reclaim excess WHT via HMRC — complex but possible",
        "Germany":        "UK-Germany tax treaty reduces rate; reclaim via broker",
        "France":         "12.8% treaty rate — confirm with broker",
    }
    scores["withholding_tax"] = _pass(
        f"{wht}%",
        wht <= THRESHOLDS["wht_max"],
        notes.get(country, "Check UK double-taxation treaty with this country")
    )

    # ── GBP-ADJUSTED 10Y CAGR (estimate) ─────────────────────
    if not h10.empty:
        prices = h10["Close"]
        if len(prices) >= 252 * 8:
            years  = len(prices) / 252
            lcl_cagr = safe_cagr(float(prices.iloc[0]), float(prices.iloc[-1]), years) or 0
            fx_drag  = FX_DRAG_VS_GBP.get(currency, 0)
            gbp_cagr = lcl_cagr + fx_drag
            scores["gbp_adjusted_cagr_10y"] = _pass(
                f"~{gbp_cagr:.1f}% p.a. (local {lcl_cagr:.1f}% + FX adj {fx_drag:+.1f}%)",
                gbp_cagr >= THRESHOLDS["gbp_cagr_10y_min"]
            )

    # ── ISA ELIGIBILITY ───────────────────────────────────────
    # OTC/pink sheet stocks are NOT ISA-eligible
    isa_eligible = exchange not in ["PNK", "PINK", "GREY", "OTC"]
    scores["isa_eligible"] = _pass(
        "Yes" if isa_eligible else "No — OTC/unlisted",
        isa_eligible,
        "Confirm with your broker (Trading 212, HL, etc.) for specific stocks"
    )

    # ── STAMP DUTY ────────────────────────────────────────────
    # 0.5% on purchases of UK-listed shares; 0% for overseas stocks
    stamp = 0.5 if country == "United Kingdom" else 0.0
    scores["stamp_duty"] = _pass(
        f"{stamp}% on purchase",
        True,  # informational — not a pass/fail
        "0.5% applies to UK shares only; no stamp duty on non-UK stocks"
    )

    # ── CURRENCY RISK ─────────────────────────────────────────
    major_currencies = {"GBP", "USD", "EUR", "CHF", "JPY", "AUD", "CAD"}
    high_fx_risk = currency not in major_currencies
    fx_drag_est  = FX_DRAG_VS_GBP.get(currency, 0)
    scores["currency_risk"] = _pass(
        f"{currency} | Est. annual FX impact: {fx_drag_est:+.1f}%",
        not high_fx_risk,
        "Minor currencies carry higher volatility vs GBP"
    )

    # ── COUNTRY RISK ──────────────────────────────────────────
    in_low_risk = country in LOW_RISK_COUNTRIES
    scores["country_risk"] = _pass(
        country,
        in_low_risk,
        "Based on political stability, rule of law, regulatory transparency"
    )

    return scores
