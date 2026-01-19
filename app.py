from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

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

@app.route('/stock/<ticker>/holdings', methods=['GET'])
def get_holdings(ticker):
    """
    Get ETF holdings and sector weights.
    Returns top holdings and sector breakdown for ETFs/Funds.
    """
    try:
        ticker_obj = yf.Ticker(ticker)

        # Check if this is an ETF/Fund
        info = ticker_obj.info
        quote_type = info.get('quoteType', '')
        if quote_type not in ['ETF', 'MUTUALFUND']:
            return jsonify({"error": f"Not an ETF or Fund (quoteType: {quote_type})"}), 400

        # Get funds data
        funds_data = ticker_obj.funds_data

        result = {
            "topHoldings": [],
            "sectorWeights": [],
            "fetchedAt": datetime.now().isoformat()
        }

        # Get top holdings
        try:
            top_holdings = funds_data.top_holdings
            if top_holdings is not None and not top_holdings.empty:
                for idx, row in top_holdings.iterrows():
                    holding = {
                        "symbol": str(idx) if idx else "",
                        "holdingName": row.get('Name', row.get('holdingName', '')),
                        "holdingPercent": float(row.get('Holding Percent', row.get('holdingPercent', 0))) * 100
                    }
                    result["topHoldings"].append(holding)
        except Exception as e:
            print(f"Error getting top holdings: {e}")

        # Get sector weights
        try:
            sector_weights = funds_data.sector_weightings
            if sector_weights:
                for sector_dict in sector_weights:
                    for sector, weight in sector_dict.items():
                        result["sectorWeights"].append({
                            "sector": sector,
                            "weight": float(weight) * 100
                        })
        except Exception as e:
            print(f"Error getting sector weights: {e}")

        # Return error if no data available
        if not result["topHoldings"] and not result["sectorWeights"]:
            return jsonify({"error": "No holdings data available"}), 404

        return jsonify(result), 200
    except Exception as e:
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
