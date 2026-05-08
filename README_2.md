# Global Stock Screening Agent — UK GBP Investor

A fully automated 5-layer stock screening agent built for UK investors
investing globally in GBP. Screens fundamentals, valuation, governance,
technicals, and UK-specific risk across major global exchanges.

---

## 📁 Project Structure

```
stock_agent/
├── config.py          ← All thresholds, weights, universe, constants
├── fetcher.py         ← Data acquisition (yfinance + FMP + FX rates)
├── screener.py        ← Five screening layers
├── agent.py           ← Scorer, reporter & main entry point
└── requirements.txt   ← Dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free API keys
| Service | URL | Cost | Usage |
|---|---|---|---|
| Financial Modeling Prep | https://financialmodelingprep.com | Free | 250 req/day — insider data, DCF, ratios |
| ExchangeRate-API | https://exchangerate-api.com | Free | 1,500 req/month — live GBP FX rates |

> **yfinance** requires no key — uses Yahoo Finance data directly.

### 3. Add your API keys to `config.py`
```python
FMP_API_KEY           = "your_key_here"
EXCHANGE_RATE_API_KEY = "your_key_here"
```

### 4. Run the agent
```bash
python agent.py
```

---

## 📊 Screening Criteria (5 Layers)

### Layer 1 — Fundamentals (30% weight)
| Metric | Threshold | Why |
|---|---|---|
| ROE | > 15% | Efficient use of shareholders' equity |
| ROCE | > 15% | Efficiency across all capital |
| ROIC | > 12% | Must exceed cost of capital |
| Debt/Equity | < 0.8x | Manageable leverage |
| Revenue CAGR (5Y) | > 10% | Consistent top-line growth |
| Net Profit CAGR (5Y) | > 10% | Growing bottom line |
| FCF | Positive ≥ 4 of 5 years | Earnings backed by cash |
| OCF / Net Income | > 1.0x | Cash earnings quality check |
| Interest Coverage | > 5x | Debt comfortably serviceable |
| Gross Margin | Upward trend | Pricing power signal |
| Receivables vs Revenue | Receivables ≤ Revenue growth | Earnings inflation check |

### Layer 2 — Valuation (20% weight)
| Metric | Threshold | Why |
|---|---|---|
| Trailing P/E | < 45x | Broad entry price filter |
| Forward P/E | < 35x | Analyst earnings expectations |
| Price / FCF | < 25x | Cash-based valuation |
| PEG Ratio | < 1.5 | P/E relative to growth |
| EV / EBITDA | < 20x | Enterprise value vs earnings |
| Price / Book | < 10x | Asset-based sanity check |
| DCF Margin of Safety | > 10% below intrinsic value | Buying with a discount |
| EPS Growth (fwd) | > 5% | Positive earnings trajectory |

### Layer 3 — Governance (20% weight)
| Metric | Threshold | Why |
|---|---|---|
| Insider Ownership | > 5% | Skin in the game |
| Institutional Ownership | 40–85% | Healthy institutional confidence |
| Short Interest Ratio | < 5 days | Low short-seller conviction |
| Insider Net Buying | Net positive | Smart money buying |
| Buyback History | ≥ 3 of 5 years | Shareholder-friendly capital return |
| Dividend Consistency | 5Y history, payout < 75% | Sustainable income signal |

### Layer 4 — Technical (15% weight)
| Metric | Threshold | Why |
|---|---|---|
| 10Y Price CAGR | > 10% p.a. | Proven long-term compounder |
| 5Y Price CAGR | > 8% p.a. | Recent trend confirmation |
| Price vs 200-DMA | Above | Established uptrend |
| 200-DMA Slope | Rising | Trend sustainability |
| RSI (14-day) | < 70 | Not overbought at entry |
| Beta | < 1.3 | Manageable volatility |
| 52-Week Proximity | Within 25% of high | Not in structural downtrend |
| Volume Trend | 30D ≥ 90D avg | Conviction in price moves |

### Layer 5 — UK Investor Risk (15% weight)
| Metric | Threshold | Why |
|---|---|---|
| Withholding Tax | ≤ 15% | Dividend income protection |
| GBP-Adjusted CAGR | > 10% p.a. | True GBP return after FX |
| ISA Eligible | Yes | Tax shelter eligibility |
| Stamp Duty | Informational | 0.5% UK shares; 0% overseas |
| Currency Risk | Major currencies only | FX volatility management |
| Country Risk | Low-risk country | Political/regulatory stability |

---

## ⚙️ Customisation

### Change your stock universe (`config.py`)
```python
UNIVERSE = {
    "MY_PICKS": ["AAPL", "MSFT", "ULVR.L", "ASML.AS"],
}
```

### Adjust thresholds (`config.py`)
```python
THRESHOLDS["roe_min"] = 20.0        # Stricter ROE
THRESHOLDS["de_max"]  = 0.5         # More conservative D/E
THRESHOLDS["peg_max"] = 1.0         # Only buy at fair/cheap valuation
```

### Change scoring weights (`config.py`)
```python
WEIGHTS = {
    "fundamentals": 0.40,   # More weight on quality
    "valuation":    0.25,
    "governance":   0.15,
    "technical":    0.10,
    "uk_risk":      0.10,
}
```

### Run a quick single-stock check
```python
from agent import run_agent
df = run_agent(["MSFT"], verbose=True)
```

---

## 📤 Output

The agent outputs:
1. **Terminal leaderboard** — top 15 stocks ranked by score
2. **Per-stock breakdown** — pass/fail for every metric
3. **CSV file** — `stock_screener_results_YYYYMMDD_HHMMSS.csv`

---

## ⚠️ UK Investor Reminders

- **W-8BEN form**: Submit to your broker for US stocks to reduce WHT from 30% → 15%
- **ISA allowance**: £20,000/year in Stocks & Shares ISA (CGT and dividend tax free)
- **CGT exempt amount**: Only £3,000/year outside ISA — hold winners in ISA first
- **Stamp duty**: 0.5% on UK shares only; no stamp duty on US, EU, or other stocks
- **Dividend tax**: Basic rate (8.75%), Higher (33.75%), Additional (39.35%) outside ISA

---

## 🆓 Data Sources

| Source | API Key | Limit | Data Provided |
|---|---|---|---|
| yfinance | None needed | Unlimited* | Price history, financials, ratios |
| FMP Free | Required | 250 req/day | Insider trades, DCF, ROIC, ratios |
| ExchangeRate-API | Required | 1,500/month | Live GBP FX conversion rates |

*yfinance is an unofficial Yahoo Finance wrapper — use responsibly and add delays.
