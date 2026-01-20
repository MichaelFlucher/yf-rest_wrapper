from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

app = Flask(__name__)

# API keys from environment
TIINGO_API_KEY = os.environ.get('TIINGO_API_KEY')
FMP_API_KEY = os.environ.get('FMP_API_KEY')

pd.options.mode.use_inf_as_na = True

def clean_json(df):
    # Replace literal strings "NaN" or "nan" with actual np.nan
    df = df.replace(["NaN", "nan"], np.nan)
    # Replace infinities with np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop any rows that contain NaN
    df = df.dropna(how='any')
    # Reset index and return a list of records
    return df.reset_index().to_dict(orient='records')

# Route for searching symbols by ISIN or query
@app.route('/search', methods=['GET'])
def search_symbol():
    try:
        query = request.args.get('q')
        region = request.args.get('region', 'US')

        if not query:
            return jsonify({"error": "Missing query parameter 'q'"}), 400

        # Build Yahoo Finance search URL
        yahoo_url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 10,
            'newsCount': 0,
            'enableFuzzyQuery': False,
            'quotesQueryId': 'tss_match_phrase_query',
            'region': region
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        response = requests.get(yahoo_url, params=params, headers=headers, timeout=10)

        if not response.ok:
            return jsonify({
                "error": f"Yahoo Finance returned {response.status_code}",
                "details": response.text[:200]
            }), response.status_code

        return jsonify(response.json()), 200

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request to Yahoo Finance timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for ticker basic info
@app.route('/stock/<ticker>', methods=['GET'])
def get_ticker_info(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        return jsonify(ticker_obj.info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/growth_estimates', methods=['GET'])
def get_growth_estimates(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        growth = ticker_obj.growth_estimates
        if growth is None or growth.empty:
            return jsonify({"error": "No growth estimates available"}), 404
        return jsonify(clean_json(growth)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/earnings', methods=['GET'])
def get_earnings(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        earnings = ticker_obj.earnings
        if earnings is None or earnings.empty:
            return jsonify({"error": "No earnings data available"}), 404
        return jsonify(clean_json(earnings)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/financials', methods=['GET'])
def get_financials(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        financials = ticker_obj.financials
        if financials is None or financials.empty:
            return jsonify({"error": "No financials data available"}), 404
        return jsonify(clean_json(financials)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/balance_sheet', methods=['GET'])
def get_balance_sheet(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        balance_sheet = ticker_obj.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return jsonify({"error": "No balance sheet data available"}), 404
        return jsonify(clean_json(balance_sheet)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/cashflow', methods=['GET'])
def get_cashflow(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        cashflow = ticker_obj.cashflow
        if cashflow is None or cashflow.empty:
            return jsonify({"error": "No cashflow data available"}), 404
        return jsonify(clean_json(cashflow)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/sustainability', methods=['GET'])
def get_sustainability(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        sustainability = ticker_obj.sustainability
        if sustainability is None or sustainability.empty:
            return jsonify({"error": "No sustainability data available"}), 404
        return jsonify(clean_json(sustainability)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock/<ticker>/recommendations', methods=['GET'])
def get_recommendations(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        recommendations = ticker_obj.recommendations
        if recommendations is None or recommendations.empty:
            return jsonify({"error": "No recommendations available"}), 404
        return jsonify(clean_json(recommendations)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_holdings_from_yahoo_api(ticker):
    """
    Fetch ETF holdings directly from Yahoo Finance quoteSummary API.
    This works for more ETFs than yfinance's funds_data property.
    """
    print(f"[{ticker}] Fetching from Yahoo Finance quoteSummary API...")
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {
        'modules': 'topHoldings,assetProfile'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if not response.ok:
            print(f"[{ticker}] Yahoo Finance API returned status {response.status_code}")
            return None

        data = response.json()
        result = data.get('quoteSummary', {}).get('result', [{}])[0]

        if result and 'topHoldings' in result:
            holdings_count = len(result['topHoldings'].get('holdings', []))
            sectors_count = len(result['topHoldings'].get('sectorWeightings', []))
            print(f"[{ticker}] Yahoo Finance API: Found {holdings_count} holdings, {sectors_count} sector weightings")
            return result
        else:
            print(f"[{ticker}] Yahoo Finance API: No topHoldings data in response")
            return None
    except Exception as e:
        print(f"[{ticker}] Yahoo Finance API error: {e}")
        return None


def fetch_holdings_from_tiingo(ticker):
    """
    Fetch ETF holdings from Tiingo API as a fallback.
    Requires TIINGO_API_KEY environment variable.
    """
    print(f"[{ticker}] Fetching from Tiingo API...")

    if not TIINGO_API_KEY:
        print(f"[{ticker}] Tiingo API: No API key configured (TIINGO_API_KEY not set)")
        return None

    # Strip exchange suffix for Tiingo (e.g., CNYAN.MX -> CNYAN)
    base_ticker = ticker.split('.')[0]
    print(f"[{ticker}] Tiingo API: Using base ticker '{base_ticker}'")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {TIINGO_API_KEY}'
    }

    result = {
        "topHoldings": [],
        "sectorWeights": []
    }

    # Try to get holdings
    try:
        holdings_url = f"https://api.tiingo.com/tiingo/funds/{base_ticker}/holdings"
        response = requests.get(holdings_url, headers=headers, timeout=10)
        print(f"[{ticker}] Tiingo holdings API: Status {response.status_code}")
        if response.ok:
            holdings_data = response.json()
            if isinstance(holdings_data, list) and len(holdings_data) > 0:
                latest = holdings_data[0] if holdings_data else {}
                holdings = latest.get('holdings', [])
                print(f"[{ticker}] Tiingo API: Found {len(holdings)} holdings")
                for holding in holdings[:20]:
                    result["topHoldings"].append({
                        "symbol": holding.get('ticker', ''),
                        "holdingName": holding.get('name', ''),
                        "holdingPercent": float(holding.get('weight', 0)) * 100
                    })
            else:
                print(f"[{ticker}] Tiingo API: Empty or invalid holdings response")
        else:
            print(f"[{ticker}] Tiingo holdings API: Failed - {response.text[:200]}")
    except Exception as e:
        print(f"[{ticker}] Tiingo holdings API error: {e}")

    # Try to get sector/exposure data
    try:
        exposure_url = f"https://api.tiingo.com/tiingo/funds/{base_ticker}/metrics"
        response = requests.get(exposure_url, headers=headers, timeout=10)
        print(f"[{ticker}] Tiingo metrics API: Status {response.status_code}")
        if response.ok:
            metrics_data = response.json()
            if isinstance(metrics_data, list) and len(metrics_data) > 0:
                latest = metrics_data[0] if metrics_data else {}
                sector_fields = [
                    ('basicMaterials', 'Basic Materials'),
                    ('communicationServices', 'Communication Services'),
                    ('consumerCyclical', 'Consumer Cyclical'),
                    ('consumerDefensive', 'Consumer Defensive'),
                    ('energy', 'Energy'),
                    ('financialServices', 'Financial Services'),
                    ('healthcare', 'Healthcare'),
                    ('industrials', 'Industrials'),
                    ('realEstate', 'Real Estate'),
                    ('technology', 'Technology'),
                    ('utilities', 'Utilities'),
                ]
                sectors_found = 0
                for field, display_name in sector_fields:
                    weight = latest.get(field)
                    if weight and float(weight) > 0:
                        result["sectorWeights"].append({
                            "sector": display_name,
                            "weight": float(weight) * 100
                        })
                        sectors_found += 1
                print(f"[{ticker}] Tiingo API: Found {sectors_found} sector weightings")
            else:
                print(f"[{ticker}] Tiingo API: Empty or invalid metrics response")
        else:
            print(f"[{ticker}] Tiingo metrics API: Failed - {response.text[:200]}")
    except Exception as e:
        print(f"[{ticker}] Tiingo metrics API error: {e}")

    if result["topHoldings"] or result["sectorWeights"]:
        print(f"[{ticker}] Tiingo API: SUCCESS - {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors")
        return result

    print(f"[{ticker}] Tiingo API: No data found")
    return None


def fetch_holdings_from_fmp(ticker):
    """
    Fetch ETF holdings from Financial Modeling Prep API as a fallback.
    Requires FMP_API_KEY environment variable.
    """
    print(f"[{ticker}] Fetching from Financial Modeling Prep API...")

    if not FMP_API_KEY:
        print(f"[{ticker}] FMP API: No API key configured (FMP_API_KEY not set)")
        return None

    # Strip exchange suffix for FMP (e.g., CNYAN.MX -> CNYAN)
    base_ticker = ticker.split('.')[0]
    print(f"[{ticker}] FMP API: Using base ticker '{base_ticker}'")

    result = {
        "topHoldings": [],
        "sectorWeights": []
    }

    # Try to get ETF holdings
    try:
        holdings_url = f"https://financialmodelingprep.com/api/v3/etf-holder/{base_ticker}?apikey={FMP_API_KEY}"
        response = requests.get(holdings_url, timeout=10)
        print(f"[{ticker}] FMP holdings API: Status {response.status_code}")
        if response.ok:
            holdings_data = response.json()
            if isinstance(holdings_data, list) and len(holdings_data) > 0:
                print(f"[{ticker}] FMP API: Found {len(holdings_data)} holdings")
                for holding in holdings_data[:20]:
                    weight = holding.get('weightPercentage', '')
                    if isinstance(weight, str):
                        weight = float(weight.replace('%', '')) if weight else 0
                    else:
                        weight = float(weight) if weight else 0
                    result["topHoldings"].append({
                        "symbol": holding.get('asset', ''),
                        "holdingName": holding.get('name', ''),
                        "holdingPercent": weight
                    })
            else:
                print(f"[{ticker}] FMP API: Empty or invalid holdings response - {str(holdings_data)[:200]}")
        else:
            print(f"[{ticker}] FMP holdings API: Failed - {response.text[:200]}")
    except Exception as e:
        print(f"[{ticker}] FMP holdings API error: {e}")

    # Try to get sector weightings
    try:
        sector_url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{base_ticker}?apikey={FMP_API_KEY}"
        response = requests.get(sector_url, timeout=10)
        print(f"[{ticker}] FMP sector API: Status {response.status_code}")
        if response.ok:
            sector_data = response.json()
            if isinstance(sector_data, list) and len(sector_data) > 0:
                sectors_found = 0
                for sector_item in sector_data:
                    weight = sector_item.get('weightPercentage', '')
                    if isinstance(weight, str):
                        weight = float(weight.replace('%', '')) if weight else 0
                    else:
                        weight = float(weight) if weight else 0
                    if weight > 0:
                        result["sectorWeights"].append({
                            "sector": sector_item.get('sector', ''),
                            "weight": weight
                        })
                        sectors_found += 1
                print(f"[{ticker}] FMP API: Found {sectors_found} sector weightings")
            else:
                print(f"[{ticker}] FMP API: Empty or invalid sector response - {str(sector_data)[:200]}")
        else:
            print(f"[{ticker}] FMP sector API: Failed - {response.text[:200]}")
    except Exception as e:
        print(f"[{ticker}] FMP sector API error: {e}")

    if result["topHoldings"] or result["sectorWeights"]:
        print(f"[{ticker}] FMP API: SUCCESS - {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors")
        return result

    print(f"[{ticker}] FMP API: No data found")
    return None


@app.route('/stock/<ticker>/holdings', methods=['GET'])
def get_holdings(ticker):
    """
    Get ETF holdings and sector weights.
    Returns top holdings and sector breakdown for ETFs/Funds.
    Uses Yahoo Finance quoteSummary API directly for better coverage.
    """
    print(f"\n{'='*60}")
    print(f"[{ticker}] Starting holdings lookup...")
    print(f"{'='*60}")

    try:
        ticker_obj = yf.Ticker(ticker)

        # Check if this is an ETF/Fund
        info = ticker_obj.info
        quote_type = info.get('quoteType', '')
        print(f"[{ticker}] Quote type: {quote_type}")

        if quote_type not in ['ETF', 'MUTUALFUND']:
            print(f"[{ticker}] REJECTED: Not an ETF or Fund")
            return jsonify({"error": f"Not an ETF or Fund (quoteType: {quote_type})"}), 400

        result = {
            "topHoldings": [],
            "sectorWeights": [],
            "fetchedAt": datetime.now().isoformat()
        }

        # Try Yahoo Finance quoteSummary API directly (better coverage)
        yahoo_data = fetch_holdings_from_yahoo_api(ticker)

        if yahoo_data and 'topHoldings' in yahoo_data:
            top_holdings_data = yahoo_data['topHoldings']

            # Get holdings
            holdings = top_holdings_data.get('holdings', [])
            for holding in holdings:
                result["topHoldings"].append({
                    "symbol": holding.get('symbol', ''),
                    "holdingName": holding.get('holdingName', ''),
                    "holdingPercent": holding.get('holdingPercent', {}).get('raw', 0) * 100
                })

            # Get sector weightings
            sector_weights = top_holdings_data.get('sectorWeightings', [])
            for sector_dict in sector_weights:
                for sector, weight_data in sector_dict.items():
                    weight = weight_data.get('raw', 0) if isinstance(weight_data, dict) else weight_data
                    result["sectorWeights"].append({
                        "sector": sector,
                        "weight": float(weight) * 100
                    })

            if result["topHoldings"] or result["sectorWeights"]:
                result["source"] = "yahoo"
                print(f"[{ticker}] SUCCESS via Yahoo Finance API")

        # Fallback to yfinance funds_data if Yahoo API didn't return data
        if not result["topHoldings"] and not result["sectorWeights"]:
            print(f"[{ticker}] Fetching from yfinance funds_data...")
            try:
                funds_data = ticker_obj.funds_data
                if funds_data:
                    # Get top holdings from yfinance
                    try:
                        top_holdings = funds_data.top_holdings
                        if top_holdings is not None and hasattr(top_holdings, 'empty') and not top_holdings.empty:
                            print(f"[{ticker}] yfinance: Found {len(top_holdings)} holdings")
                            for idx, row in top_holdings.iterrows():
                                result["topHoldings"].append({
                                    "symbol": str(idx) if idx else "",
                                    "holdingName": row.get('Name', row.get('holdingName', '')),
                                    "holdingPercent": float(row.get('Holding Percent', row.get('holdingPercent', 0))) * 100
                                })
                        else:
                            print(f"[{ticker}] yfinance: No holdings data")
                    except Exception as e:
                        print(f"[{ticker}] yfinance holdings error: {e}")

                    # Get sector weights from yfinance
                    try:
                        sector_weights = funds_data.sector_weightings
                        if sector_weights:
                            print(f"[{ticker}] yfinance: Found sector weightings")
                            if isinstance(sector_weights, list):
                                for item in sector_weights:
                                    if isinstance(item, dict):
                                        for sector, weight in item.items():
                                            result["sectorWeights"].append({
                                                "sector": sector,
                                                "weight": float(weight) * 100
                                            })
                            elif isinstance(sector_weights, dict):
                                for sector, weight in sector_weights.items():
                                    result["sectorWeights"].append({
                                        "sector": sector,
                                        "weight": float(weight) * 100
                                    })
                        else:
                            print(f"[{ticker}] yfinance: No sector weightings")
                    except Exception as e:
                        print(f"[{ticker}] yfinance sector error: {e}")

                    if result["topHoldings"] or result["sectorWeights"]:
                        result["source"] = "yfinance"
                        print(f"[{ticker}] SUCCESS via yfinance")
                else:
                    print(f"[{ticker}] yfinance: No funds_data available")
            except Exception as e:
                print(f"[{ticker}] yfinance funds_data error: {e}")

        # Fallback to Tiingo API if still no data
        if not result["topHoldings"] and not result["sectorWeights"]:
            tiingo_data = fetch_holdings_from_tiingo(ticker)
            if tiingo_data:
                result["topHoldings"] = tiingo_data.get("topHoldings", [])
                result["sectorWeights"] = tiingo_data.get("sectorWeights", [])
                result["source"] = "tiingo"
                print(f"[{ticker}] SUCCESS via Tiingo API")

        # Fallback to Financial Modeling Prep API if still no data
        if not result["topHoldings"] and not result["sectorWeights"]:
            fmp_data = fetch_holdings_from_fmp(ticker)
            if fmp_data:
                result["topHoldings"] = fmp_data.get("topHoldings", [])
                result["sectorWeights"] = fmp_data.get("sectorWeights", [])
                result["source"] = "fmp"
                print(f"[{ticker}] SUCCESS via FMP API")

        # Return error if no data available from any source
        if not result["topHoldings"] and not result["sectorWeights"]:
            print(f"[{ticker}] FAILED: No data from any source")
            print(f"{'='*60}\n")
            return jsonify({"error": "No holdings data available"}), 404

        print(f"[{ticker}] Final result: {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors (source: {result.get('source', 'unknown')})")
        print(f"{'='*60}\n")
        return jsonify(result), 200
    except Exception as e:
        print(f"[{ticker}] EXCEPTION: {e}")
        print(f"{'='*60}\n")
        return jsonify({"error": str(e)}), 500

@app.route('/stocks/close_prices', methods=['POST'])
def get_close_prices():
    try:
        data = request.get_json()
        tickers = data.get("tickers", [])
        if not tickers or not isinstance(tickers, list):
            return jsonify({"error": "Invalid or missing tickers list"}), 400

        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)

        df = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if df.empty:
            return jsonify({"error": "No data available for the given tickers"}), 404

        # Convert to structured format
        response = {
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "prices": {ticker: df[ticker].tolist() for ticker in tickers}
        }

        return jsonify(clean_json(response)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stocks/close_prices_range', methods=['GET'])
def get_close_prices_range():
    try:
        tickers = request.args.getlist("ticker")
        start_date = request.args.get("start")
        end_date = request.args.get("end")

        # Validate input
        if not tickers:
            return jsonify({"error": "Missing tickers"}), 400
        if not start_date or not end_date:
            return jsonify({"error": "Missing start or end date"}), 400

        # Parse dates
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        df = yf.download(tickers, start=start, end=end, auto_adjust=False)["Close"]
        if df.empty:
            return jsonify({"error": "No data available for the given parameters"}), 404

        # Ensure all tickers are in the DataFrame (even if missing)
        for ticker in tickers:
            if ticker not in df.columns:
                df[ticker] = float('nan')

        response = {
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "prices": {ticker: [None if pd.isna(v) else v for v in df[ticker].tolist()] for ticker in tickers}
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
