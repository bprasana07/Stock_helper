"""
=============================================================
  fetcher.py — Data acquisition layer
  Sources: yfinance (free), FMP free tier, ExchangeRate-API
=============================================================
"""

import time
import logging
import requests
import yfinance as yf
import pandas as pd
from typing import Optional
from config import FMP_API_KEY, EXCHANGE_RATE_API_KEY, FX_DRAG_VS_GBP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GBP CONVERTER
# ─────────────────────────────────────────────────────────────
class GBPConverter:
    """Fetches live GBP exchange rates via ExchangeRate-API free tier."""

    def __init__(self):
        self._rates = {}
        self._fetch_rates()

    def _fetch_rates(self):
        try:
            url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/GBP"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data.get("result") == "success":
                self._rates = data["conversion_rates"]
                logger.info("✅ Live GBP exchange rates loaded")
                return
        except Exception as e:
            logger.warning(f"Live FX fetch failed: {e}")

        # Fallback hardcoded rates (update periodically)
        logger.warning("⚠️  Using hardcoded fallback FX rates")
        self._rates = {
            "USD": 1.27, "EUR": 1.17, "CHF": 1.13, "JPY": 192.0,
            "AUD": 1.96, "CAD": 1.73, "DKK": 8.73, "SEK": 13.1,
            "TWD": 41.0, "SGD": 1.71, "HKD": 9.93,
        }

    def to_gbp(self, amount: float, from_currency: str) -> float:
        """Convert an amount in foreign currency to GBP."""
        if from_currency == "GBP":
            return amount
        rate = self._rates.get(from_currency)
        if rate:
            return amount / rate
        return amount  # unknown currency — return as-is

    def fx_drag(self, currency: str) -> float:
        """Historical annual FX drag vs GBP (%)."""
        return FX_DRAG_VS_GBP.get(currency, 0.0)


# ─────────────────────────────────────────────────────────────
# STOCK DATA FETCHER
# ─────────────────────────────────────────────────────────────
class StockDataFetcher:
    """
    Fetches all data needed for screening from yfinance + FMP.
    yfinance  → price history, financials, info (free, no key)
    FMP       → insider transactions, DCF, ratios (free tier: 250 req/day)
    """

    FMP_BASE = "https://financialmodelingprep.com/api/v3"

    def __init__(self, gbp_converter: GBPConverter):
        self.gbp = gbp_converter

    def _fmp(self, endpoint: str, params: dict = {}) -> Optional[list]:
        """Generic FMP API call with error handling."""
        try:
            params = {**params, "apikey": FMP_API_KEY}
            resp = requests.get(f"{self.FMP_BASE}/{endpoint}", params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            logger.debug(f"FMP {endpoint} returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"FMP error [{endpoint}]: {e}")
        return None

    def _fmp_ticker(self, ticker: str) -> str:
        """Normalise ticker for FMP (remove exchange suffixes)."""
        return (ticker
                .replace(".L", "")
                .replace(".AS", "")
                .replace(".DE", "")
                .replace(".PA", "")
                .replace(".SW", "")
                .replace(".CO", ""))

    def fetch(self, ticker: str) -> dict:
        """
        Returns a unified data dict for a given ticker.
        Keys: ticker, name, sector, country, currency, info,
              hist_10y, hist_5y, hist_1y, income_stmt,
              balance_sheet, cashflow, fmp_ratios,
              fmp_insider, fmp_dcf, error
        """
        result = {
            "ticker":        ticker,
            "name":          ticker,
            "sector":        "Unknown",
            "country":       "Unknown",
            "currency":      "USD",
            "exchange":      "",
            "info":          {},
            "hist_10y":      __import__("pandas").DataFrame(),
            "hist_5y":       __import__("pandas").DataFrame(),
            "hist_1y":       __import__("pandas").DataFrame(),
            "income_stmt":   __import__("pandas").DataFrame(),
            "balance_sheet": __import__("pandas").DataFrame(),
            "cashflow":      __import__("pandas").DataFrame(),
            "fmp_ratios":    None,
            "fmp_insider":   None,
            "fmp_dcf":       None,
            "error":         None,
        }

        logger.info(f"  → Fetching {ticker}")

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.fast_info

            # Basic info from yfinance
            full_info = yf_ticker.info
            result["info"]     = full_info
            result["name"]     = full_info.get("longName", ticker)
            result["sector"]   = full_info.get("sector", "Unknown")
            result["country"]  = full_info.get("country", "Unknown")
            result["currency"] = full_info.get("currency", "USD")
            result["exchange"] = full_info.get("exchange", "")

            # Price history
            result["hist_10y"] = yf_ticker.history(period="10y", auto_adjust=True)
            result["hist_5y"]  = yf_ticker.history(period="5y",  auto_adjust=True)
            result["hist_1y"]  = yf_ticker.history(period="1y",  auto_adjust=True)

            # Annual financials (4 columns = last 4 fiscal years)
            result["income_stmt"]   = yf_ticker.financials
            result["balance_sheet"] = yf_ticker.balance_sheet
            result["cashflow"]      = yf_ticker.cashflow

            # FMP supplementary data (uses normalised ticker)
            fmp_t = self._fmp_ticker(ticker)
            result["fmp_ratios"]  = self._fmp(f"ratios/{fmp_t}", {"limit": 5})
            result["fmp_insider"] = self._fmp("insider-trading", {"symbol": fmp_t, "limit": 20})
            result["fmp_dcf"]     = self._fmp(f"discounted-cash-flow/{fmp_t}")

            # Ensure financial DataFrames are never None
            for key in ("income_stmt", "balance_sheet", "cashflow"):
                if result.get(key) is None:
                    result[key] = pd.DataFrame()

            time.sleep(0.4)  # Respect rate limits

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"  ✗ {ticker}: {e}")
            # Ensure history keys exist even on error
            for key in ("hist_10y", "hist_5y", "hist_1y", "income_stmt", "balance_sheet", "cashflow"):
                result.setdefault(key, pd.DataFrame())

        return result
