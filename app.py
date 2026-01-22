from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import logging
import sys
import hashlib
from cachetools import TTLCache
from threading import Lock

# Configure logging to flush immediately and show in Docker logs
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API keys from environment
TIINGO_API_KEY = os.environ.get('TIINGO_API_KEY')
FMP_API_KEY = os.environ.get('FMP_API_KEY')

# Cache configuration
# TTLCache(maxsize, ttl_in_seconds)
search_cache = TTLCache(maxsize=500, ttl=86400)  # 24 hours for ISIN searches
ticker_info_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour for ticker info
holdings_cache = TTLCache(maxsize=100, ttl=21600)  # 6 hours for ETF holdings
cache_lock = Lock()

def get_cache_key(*args):
    """Generate a cache key from arguments."""
    key_str = ":".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()

# Note: pd.options.mode.use_inf_as_na was removed in pandas 2.1.0
# Inf values are now handled in clean_json() function instead

def clean_json(df):
    # Replace literal strings "NaN" or "nan" with actual np.nan
    df = df.replace(["NaN", "nan"], np.nan)
    # Replace infinities with np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop any rows that contain NaN
    df = df.dropna(how='any')
    # Reset index and return a list of records
    return df.reset_index().to_dict(orient='records')


def enrich_holdings_with_country(holdings, etf_ticker):
    """
    Enriches ETF holdings with country information for each holding.
    Looks up each holding's ticker to get its country.

    Args:
        holdings: List of holding dicts with 'symbol' key
        etf_ticker: The ETF ticker (for logging)

    Returns:
        The same holdings list with 'country' field added to each holding
    """
    if not holdings:
        return holdings

    symbols = [h.get('symbol', '') for h in holdings if h.get('symbol')]
    if not symbols:
        return holdings

    logger.info(f"[{etf_ticker}] Enriching {len(symbols)} holdings with country data...")

    # Create a map of symbol -> country
    country_map = {}

    for symbol in symbols:
        if not symbol:
            continue
        try:
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            country = info.get('country', '')
            if country:
                country_map[symbol] = country
                logger.info(f"[{etf_ticker}]   {symbol} -> {country}")
            else:
                # Try to infer country from exchange
                exchange = info.get('exchange', '')
                country_map[symbol] = infer_country_from_exchange(exchange)
                logger.info(f"[{etf_ticker}]   {symbol} -> {country_map[symbol]} (inferred from {exchange})")
        except Exception as e:
            logger.warning(f"[{etf_ticker}]   {symbol} -> failed to get country: {e}")
            country_map[symbol] = ''

    # Add country to each holding
    for holding in holdings:
        symbol = holding.get('symbol', '')
        holding['country'] = country_map.get(symbol, '')

    logger.info(f"[{etf_ticker}] Country enrichment complete")
    return holdings


def infer_country_from_exchange(exchange):
    """
    Infers country from exchange code when direct country info is not available.
    """
    exchange_country_map = {
        'NYQ': 'United States',
        'NMS': 'United States',
        'NGM': 'United States',
        'PCX': 'United States',
        'BTS': 'United States',
        'NYS': 'United States',
        'NAS': 'United States',
        'LSE': 'United Kingdom',
        'GER': 'Germany',
        'FRA': 'Germany',
        'STU': 'Germany',
        'DUS': 'Germany',
        'MUN': 'Germany',
        'HAM': 'Germany',
        'BER': 'Germany',
        'PAR': 'France',
        'AMS': 'Netherlands',
        'BRU': 'Belgium',
        'LIS': 'Portugal',
        'MIL': 'Italy',
        'MCE': 'Spain',
        'SWX': 'Switzerland',
        'VIE': 'Austria',
        'HKG': 'Hong Kong',
        'TYO': 'Japan',
        'SHH': 'China',
        'SHZ': 'China',
        'TSE': 'Canada',
        'ASX': 'Australia',
        'KSC': 'South Korea',
        'KOE': 'South Korea',
        'TAI': 'Taiwan',
        'NSE': 'India',
        'BSE': 'India',
    }
    return exchange_country_map.get(exchange, '')

# Route for searching symbols by ISIN or query
@app.route('/search', methods=['GET'])
def search_symbol():
    try:
        query = request.args.get('q')
        region = request.args.get('region', 'US')

        if not query:
            return jsonify({"error": "Missing query parameter 'q'"}), 400

        # Check cache first
        cache_key = get_cache_key('search', query, region)
        with cache_lock:
            if cache_key in search_cache:
                logger.info(f"[search] Cache HIT for query '{query}'")
                return jsonify(search_cache[cache_key]), 200

        logger.info(f"[search] Cache MISS for query '{query}', fetching from Yahoo...")

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

        result = response.json()

        # Cache the successful result
        with cache_lock:
            search_cache[cache_key] = result

        return jsonify(result), 200

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
        # Check cache first
        cache_key = get_cache_key('ticker_info', ticker)
        with cache_lock:
            if cache_key in ticker_info_cache:
                logger.info(f"[{ticker}] Ticker info cache HIT")
                return jsonify(ticker_info_cache[cache_key]), 200

        logger.info(f"[{ticker}] Ticker info cache MISS, fetching from Yahoo...")

        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # Cache the result
        with cache_lock:
            ticker_info_cache[cache_key] = info

        return jsonify(info), 200
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

def find_alternative_ticker_by_isin(isin, original_ticker):
    """
    Search for alternative tickers using ISIN.
    Returns a US-listed ticker if available, or None.
    Prioritizes US exchanges for better data coverage.
    """
    if not isin:
        return None

    logger.info(f"[{original_ticker}] Searching for alternative tickers with ISIN: {isin}")

    try:
        yahoo_url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': isin,
            'quotesCount': 10,
            'newsCount': 0,
            'enableFuzzyQuery': False,
            'quotesQueryId': 'tss_match_phrase_query'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        response = requests.get(yahoo_url, params=params, headers=headers, timeout=10)
        if not response.ok:
            logger.warning(f"[{original_ticker}] ISIN search failed: {response.status_code}")
            return None

        data = response.json()
        quotes = data.get('quotes', [])

        if not quotes:
            logger.info(f"[{original_ticker}] No alternative tickers found for ISIN {isin}")
            return None

        logger.info(f"[{original_ticker}] Found {len(quotes)} tickers for ISIN {isin}")

        # Prioritize exchanges: US first, then major exchanges
        priority_exchanges = ['NYQ', 'NMS', 'NGM', 'PCX', 'BTS', 'LSE', 'GER', 'PAR']

        for exchange in priority_exchanges:
            for quote in quotes:
                symbol = quote.get('symbol', '')
                quote_exchange = quote.get('exchange', '')
                quote_type = quote.get('quoteType', '')

                # Skip if it's the same ticker we started with
                if symbol == original_ticker:
                    continue

                # Only consider ETFs
                if quote_type not in ['ETF', 'MUTUALFUND']:
                    continue

                if quote_exchange == exchange:
                    logger.info(f"[{original_ticker}] Found alternative: {symbol} on {quote_exchange}")
                    return symbol

        # If no priority exchange found, return first ETF that's not the original
        for quote in quotes:
            symbol = quote.get('symbol', '')
            quote_type = quote.get('quoteType', '')
            if symbol != original_ticker and quote_type in ['ETF', 'MUTUALFUND']:
                logger.info(f"[{original_ticker}] Using alternative: {symbol} on {quote.get('exchange', 'unknown')}")
                return symbol

        logger.info(f"[{original_ticker}] No suitable alternative ticker found")
        return None

    except Exception as e:
        logger.error(f"[{original_ticker}] ISIN search error: {e}")
        return None


def fetch_holdings_from_yahoo_api(ticker):
    """
    Fetch ETF holdings directly from Yahoo Finance quoteSummary API.
    This works for more ETFs than yfinance's funds_data property.
    """
    logger.info(f"[{ticker}] Fetching from Yahoo Finance quoteSummary API...")
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
            logger.warning(f"[{ticker}] Yahoo Finance API returned status {response.status_code}")
            return None

        data = response.json()
        result = data.get('quoteSummary', {}).get('result', [{}])[0]

        if result and 'topHoldings' in result:
            holdings_count = len(result['topHoldings'].get('holdings', []))
            sectors_count = len(result['topHoldings'].get('sectorWeightings', []))
            logger.info(f"[{ticker}] Yahoo Finance API: Found {holdings_count} holdings, {sectors_count} sector weightings")
            return result
        else:
            logger.info(f"[{ticker}] Yahoo Finance API: No topHoldings data in response")
            return None
    except Exception as e:
        logger.error(f"[{ticker}] Yahoo Finance API error: {e}")
        return None


def fetch_holdings_from_tiingo(ticker):
    """
    Fetch ETF holdings from Tiingo API as a fallback.
    Requires TIINGO_API_KEY environment variable.
    """
    logger.info(f"[{ticker}] Fetching from Tiingo API...")

    if not TIINGO_API_KEY:
        logger.warning(f"[{ticker}] Tiingo API: No API key configured (TIINGO_API_KEY not set)")
        return None

    # Strip exchange suffix for Tiingo (e.g., CNYAN.MX -> CNYAN)
    base_ticker = ticker.split('.')[0]
    logger.info(f"[{ticker}] Tiingo API: Using base ticker '{base_ticker}'")

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
        logger.info(f"[{ticker}] Tiingo holdings API: Status {response.status_code}")
        if response.ok:
            holdings_data = response.json()
            if isinstance(holdings_data, list) and len(holdings_data) > 0:
                latest = holdings_data[0] if holdings_data else {}
                holdings = latest.get('holdings', [])
                logger.info(f"[{ticker}] Tiingo API: Found {len(holdings)} holdings")
                for holding in holdings[:20]:
                    result["topHoldings"].append({
                        "symbol": holding.get('ticker', ''),
                        "holdingName": holding.get('name', ''),
                        "holdingPercent": float(holding.get('weight', 0)) * 100
                    })
            else:
                logger.info(f"[{ticker}] Tiingo API: Empty or invalid holdings response")
        else:
            logger.warning(f"[{ticker}] Tiingo holdings API: Failed - {response.text[:200]}")
    except Exception as e:
        logger.error(f"[{ticker}] Tiingo holdings API error: {e}")

    # Try to get sector/exposure data
    try:
        exposure_url = f"https://api.tiingo.com/tiingo/funds/{base_ticker}/metrics"
        response = requests.get(exposure_url, headers=headers, timeout=10)
        logger.info(f"[{ticker}] Tiingo metrics API: Status {response.status_code}")
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
                logger.info(f"[{ticker}] Tiingo API: Found {sectors_found} sector weightings")
            else:
                logger.info(f"[{ticker}] Tiingo API: Empty or invalid metrics response")
        else:
            logger.warning(f"[{ticker}] Tiingo metrics API: Failed - {response.text[:200]}")
    except Exception as e:
        logger.error(f"[{ticker}] Tiingo metrics API error: {e}")

    if result["topHoldings"] or result["sectorWeights"]:
        logger.info(f"[{ticker}] Tiingo API: SUCCESS - {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors")
        return result

    logger.info(f"[{ticker}] Tiingo API: No data found")
    return None


def fetch_holdings_from_fmp(ticker):
    """
    Fetch ETF holdings from Financial Modeling Prep API as a fallback.
    Requires FMP_API_KEY environment variable.
    Uses the stable API endpoints (v3 legacy endpoints deprecated Aug 2025).
    """
    logger.info(f"[{ticker}] Fetching from Financial Modeling Prep API...")

    if not FMP_API_KEY:
        logger.warning(f"[{ticker}] FMP API: No API key configured (FMP_API_KEY not set)")
        return None

    # Strip exchange suffix for FMP (e.g., CNYAN.MX -> CNYAN)
    base_ticker = ticker.split('.')[0]
    logger.info(f"[{ticker}] FMP API: Using base ticker '{base_ticker}'")

    result = {
        "topHoldings": [],
        "sectorWeights": []
    }

    # Try to get ETF holdings (stable API)
    try:
        holdings_url = f"https://financialmodelingprep.com/stable/etf/holdings?symbol={base_ticker}&apikey={FMP_API_KEY}"
        response = requests.get(holdings_url, timeout=10)
        logger.info(f"[{ticker}] FMP holdings API: Status {response.status_code}")
        if response.ok:
            holdings_data = response.json()
            if isinstance(holdings_data, list) and len(holdings_data) > 0:
                logger.info(f"[{ticker}] FMP API: Found {len(holdings_data)} holdings")
                # Log first holding structure for debugging
                if holdings_data:
                    logger.info(f"[{ticker}] FMP API: Sample holding keys: {list(holdings_data[0].keys())}")
                for holding in holdings_data[:20]:
                    # Handle both old and new field names
                    weight = holding.get('weightPercentage', holding.get('weight', ''))
                    if isinstance(weight, str):
                        weight = float(weight.replace('%', '')) if weight else 0
                    else:
                        weight = float(weight) if weight else 0
                    result["topHoldings"].append({
                        "symbol": holding.get('asset', holding.get('symbol', '')),
                        "holdingName": holding.get('name', ''),
                        "holdingPercent": weight
                    })
            else:
                logger.info(f"[{ticker}] FMP API: Empty or invalid holdings response - {str(holdings_data)[:200]}")
        else:
            logger.warning(f"[{ticker}] FMP holdings API: Failed - {response.text[:200]}")
    except Exception as e:
        logger.error(f"[{ticker}] FMP holdings API error: {e}")

    # Try to get sector weightings (stable API)
    try:
        sector_url = f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={base_ticker}&apikey={FMP_API_KEY}"
        response = requests.get(sector_url, timeout=10)
        logger.info(f"[{ticker}] FMP sector API: Status {response.status_code}")
        if response.ok:
            sector_data = response.json()
            if isinstance(sector_data, list) and len(sector_data) > 0:
                # Log first sector structure for debugging
                if sector_data:
                    logger.info(f"[{ticker}] FMP API: Sample sector keys: {list(sector_data[0].keys())}")
                sectors_found = 0
                for sector_item in sector_data:
                    weight = sector_item.get('weightPercentage', sector_item.get('weight', ''))
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
                logger.info(f"[{ticker}] FMP API: Found {sectors_found} sector weightings")
            else:
                logger.info(f"[{ticker}] FMP API: Empty or invalid sector response - {str(sector_data)[:200]}")
        else:
            logger.warning(f"[{ticker}] FMP sector API: Failed - {response.text[:200]}")
    except Exception as e:
        logger.error(f"[{ticker}] FMP sector API error: {e}")

    if result["topHoldings"] or result["sectorWeights"]:
        logger.info(f"[{ticker}] FMP API: SUCCESS - {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors")
        return result

    logger.info(f"[{ticker}] FMP API: No data found")
    return None


@app.route('/stock/<ticker>/holdings', methods=['GET'])
def get_holdings(ticker):
    """
    Get ETF holdings and sector weights.
    Returns top holdings and sector breakdown for ETFs/Funds.
    Uses Yahoo Finance quoteSummary API directly for better coverage.

    Optional query param:
        isin: If provided, skips ISIN lookup for alternative ticker fallback
    """
    # Get optional ISIN from query params (avoids lookup if provided by caller)
    provided_isin = request.args.get('isin')

    # Check cache first
    cache_key = get_cache_key('holdings', ticker, provided_isin or '')
    with cache_lock:
        if cache_key in holdings_cache:
            logger.info(f"[{ticker}] Holdings cache HIT")
            return jsonify(holdings_cache[cache_key]), 200

    logger.info(f"{'='*60}")
    logger.info(f"[{ticker}] Holdings cache MISS, starting lookup...")
    if provided_isin:
        logger.info(f"[{ticker}] ISIN provided by caller: {provided_isin}")
    logger.info(f"{'='*60}")

    try:
        ticker_obj = yf.Ticker(ticker)

        # Check if this is an ETF/Fund
        info = ticker_obj.info
        quote_type = info.get('quoteType', '')
        logger.info(f"[{ticker}] Quote type: {quote_type}")

        if quote_type not in ['ETF', 'MUTUALFUND']:
            logger.warning(f"[{ticker}] REJECTED: Not an ETF or Fund")
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
                logger.info(f"[{ticker}] SUCCESS via Yahoo Finance API")

        # Fallback to yfinance funds_data if Yahoo API didn't return data
        if not result["topHoldings"] and not result["sectorWeights"]:
            logger.info(f"[{ticker}] Fetching from yfinance funds_data...")
            try:
                funds_data = ticker_obj.funds_data
                if funds_data:
                    # Get top holdings from yfinance
                    try:
                        top_holdings = funds_data.top_holdings
                        if top_holdings is not None and hasattr(top_holdings, 'empty') and not top_holdings.empty:
                            logger.info(f"[{ticker}] yfinance: Found {len(top_holdings)} holdings")
                            for idx, row in top_holdings.iterrows():
                                result["topHoldings"].append({
                                    "symbol": str(idx) if idx else "",
                                    "holdingName": row.get('Name', row.get('holdingName', '')),
                                    "holdingPercent": float(row.get('Holding Percent', row.get('holdingPercent', 0))) * 100
                                })
                        else:
                            logger.info(f"[{ticker}] yfinance: No holdings data")
                    except Exception as e:
                        logger.error(f"[{ticker}] yfinance holdings error: {e}")

                    # Get sector weights from yfinance
                    try:
                        sector_weights = funds_data.sector_weightings
                        if sector_weights:
                            logger.info(f"[{ticker}] yfinance: Found sector weightings")
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
                            logger.info(f"[{ticker}] yfinance: No sector weightings")
                    except Exception as e:
                        logger.error(f"[{ticker}] yfinance sector error: {e}")

                    if result["topHoldings"] or result["sectorWeights"]:
                        result["source"] = "yfinance"
                        logger.info(f"[{ticker}] SUCCESS via yfinance")
                else:
                    logger.info(f"[{ticker}] yfinance: No funds_data available")
            except Exception as e:
                logger.error(f"[{ticker}] yfinance funds_data error: {e}")

        # Fallback to Tiingo API if still no data
        if not result["topHoldings"] and not result["sectorWeights"]:
            tiingo_data = fetch_holdings_from_tiingo(ticker)
            if tiingo_data:
                result["topHoldings"] = tiingo_data.get("topHoldings", [])
                result["sectorWeights"] = tiingo_data.get("sectorWeights", [])
                result["source"] = "tiingo"
                logger.info(f"[{ticker}] SUCCESS via Tiingo API")

        # Fallback to Financial Modeling Prep API if still no data
        if not result["topHoldings"] and not result["sectorWeights"]:
            fmp_data = fetch_holdings_from_fmp(ticker)
            if fmp_data:
                result["topHoldings"] = fmp_data.get("topHoldings", [])
                result["sectorWeights"] = fmp_data.get("sectorWeights", [])
                result["source"] = "fmp"
                logger.info(f"[{ticker}] SUCCESS via FMP API")

        # ISIN Fallback: Try to find an alternative ticker (e.g., US-listed version)
        if not result["topHoldings"] and not result["sectorWeights"]:
            # Use provided ISIN if available, otherwise try to get from ticker info
            isin = provided_isin or info.get('isin')
            if isin:
                alt_ticker = find_alternative_ticker_by_isin(isin, ticker)
                if alt_ticker:
                    logger.info(f"[{ticker}] Trying alternative ticker: {alt_ticker}")

                    # Try Yahoo Finance with alternative ticker
                    alt_yahoo_data = fetch_holdings_from_yahoo_api(alt_ticker)
                    if alt_yahoo_data and 'topHoldings' in alt_yahoo_data:
                        top_holdings_data = alt_yahoo_data['topHoldings']
                        holdings = top_holdings_data.get('holdings', [])
                        for holding in holdings:
                            result["topHoldings"].append({
                                "symbol": holding.get('symbol', ''),
                                "holdingName": holding.get('holdingName', ''),
                                "holdingPercent": holding.get('holdingPercent', {}).get('raw', 0) * 100
                            })
                        sector_weights = top_holdings_data.get('sectorWeightings', [])
                        for sector_dict in sector_weights:
                            for sector, weight_data in sector_dict.items():
                                weight = weight_data.get('raw', 0) if isinstance(weight_data, dict) else weight_data
                                result["sectorWeights"].append({
                                    "sector": sector,
                                    "weight": float(weight) * 100
                                })
                        if result["topHoldings"] or result["sectorWeights"]:
                            result["source"] = f"yahoo (via {alt_ticker})"
                            result["alternativeTicker"] = alt_ticker
                            logger.info(f"[{ticker}] SUCCESS via Yahoo Finance using {alt_ticker}")

                    # Try yfinance with alternative ticker if Yahoo didn't work
                    if not result["topHoldings"] and not result["sectorWeights"]:
                        try:
                            alt_ticker_obj = yf.Ticker(alt_ticker)
                            alt_funds_data = alt_ticker_obj.funds_data
                            if alt_funds_data:
                                try:
                                    top_holdings = alt_funds_data.top_holdings
                                    if top_holdings is not None and hasattr(top_holdings, 'empty') and not top_holdings.empty:
                                        logger.info(f"[{ticker}] yfinance ({alt_ticker}): Found {len(top_holdings)} holdings")
                                        for idx, row in top_holdings.iterrows():
                                            result["topHoldings"].append({
                                                "symbol": str(idx) if idx else "",
                                                "holdingName": row.get('Name', row.get('holdingName', '')),
                                                "holdingPercent": float(row.get('Holding Percent', row.get('holdingPercent', 0))) * 100
                                            })
                                except Exception as e:
                                    logger.error(f"[{ticker}] yfinance ({alt_ticker}) holdings error: {e}")

                                try:
                                    sector_weights = alt_funds_data.sector_weightings
                                    if sector_weights:
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
                                except Exception as e:
                                    logger.error(f"[{ticker}] yfinance ({alt_ticker}) sector error: {e}")

                                if result["topHoldings"] or result["sectorWeights"]:
                                    result["source"] = f"yfinance (via {alt_ticker})"
                                    result["alternativeTicker"] = alt_ticker
                                    logger.info(f"[{ticker}] SUCCESS via yfinance using {alt_ticker}")
                        except Exception as e:
                            logger.error(f"[{ticker}] yfinance ({alt_ticker}) error: {e}")
            else:
                logger.info(f"[{ticker}] No ISIN available for alternative ticker lookup")

        # Return error if no data available from any source
        if not result["topHoldings"] and not result["sectorWeights"]:
            logger.warning(f"[{ticker}] FAILED: No data from any source")
            logger.info(f"{'='*60}")
            return jsonify({"error": "No holdings data available"}), 404

        # Enrich holdings with country data
        if result["topHoldings"]:
            result["topHoldings"] = enrich_holdings_with_country(result["topHoldings"], ticker)

        # Cache the successful result
        with cache_lock:
            holdings_cache[cache_key] = result

        logger.info(f"[{ticker}] Final result: {len(result['topHoldings'])} holdings, {len(result['sectorWeights'])} sectors (source: {result.get('source', 'unknown')})")
        logger.info(f"{'='*60}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[{ticker}] EXCEPTION: {e}")
        logger.info(f"{'='*60}")
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

        # yfinance end date is exclusive, so add 1 day to include the end date
        end_inclusive = end + timedelta(days=1)

        df = yf.download(tickers, start=start, end=end_inclusive, auto_adjust=False)["Close"]
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


@app.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get current cache statistics."""
    with cache_lock:
        return jsonify({
            "search_cache": {
                "size": len(search_cache),
                "maxsize": search_cache.maxsize,
                "ttl_seconds": 86400
            },
            "ticker_info_cache": {
                "size": len(ticker_info_cache),
                "maxsize": ticker_info_cache.maxsize,
                "ttl_seconds": 3600
            },
            "holdings_cache": {
                "size": len(holdings_cache),
                "maxsize": holdings_cache.maxsize,
                "ttl_seconds": 21600
            }
        }), 200


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches."""
    with cache_lock:
        search_cache.clear()
        ticker_info_cache.clear()
        holdings_cache.clear()
    logger.info("All caches cleared")
    return jsonify({"message": "All caches cleared"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
