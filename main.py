#!/usr/bin/env python3
"""
COMPLETE MARKET MONITOR - v9.1 (TRADINGVIEW CLEAN EDITION)
- CLEAN CHARTS: TradingView-style clear charts without technical indicators
- ENHANCED: Improved data fetching with retry mechanisms
- COMPREHENSIVE: Complete option chain data with all metrics
- OPTIMIZED: Better performance and reliability
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Optional, Tuple

# CONFIG
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# INDICES - ALL 4
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1},
    "NSE_INDEX|Nifty Bank": {"name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|Nifty Fin Service": {"name": "FIN NIFTY", "expiry_day": 1},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCAP NIFTY", "expiry_day": 0}
}

# COMPLETE NIFTY 50 STOCKS
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE", "NSE_EQ|INE467B01029": "TATAMOTORS",
    "NSE_EQ|INE040A01034": "HDFCBANK", "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN", "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE854D01024": "TCS", "NSE_EQ|INE030A01027": "BHARTIARTL",
    "NSE_EQ|INE238A01034": "AXISBANK", "NSE_EQ|INE237A01028": "KOTAKBANK",
    "NSE_EQ|INE155A01022": "TATASTEEL", "NSE_EQ|INE047A01021": "HCLTECH",
    "NSE_EQ|INE423A01024": "ADANIENT", "NSE_EQ|INE075A01022": "WIPRO",
    "NSE_EQ|INE018A01030": "LT", "NSE_EQ|INE019A01038": "ASIANPAINT",
    "NSE_EQ|INE585B01010": "MARUTI", "NSE_EQ|INE742F01042": "ADANIPORTS",
    "NSE_EQ|INE001A01036": "ULTRACEMCO", "NSE_EQ|INE101A01026": "M&M",
    "NSE_EQ|INE044A01036": "SUNPHARMA", "NSE_EQ|INE280A01028": "TITAN",
    "NSE_EQ|INE669C01036": "TECHM", "NSE_EQ|INE522F01014": "COALINDIA",
    "NSE_EQ|INE066F01012": "JSWSTEEL", "NSE_EQ|INE733E01010": "NTPC",
    "NSE_EQ|INE752E01010": "POWERGRID", "NSE_EQ|INE239A01016": "NESTLEIND",
    "NSE_EQ|INE296A01024": "BAJFINANCE", "NSE_EQ|INE213A01029": "ONGC",
    "NSE_EQ|INE205A01025": "HINDALCO", "NSE_EQ|INE154A01025": "ITC",
    "NSE_EQ|INE860A01027": "HDFCLIFE", "NSE_EQ|INE123W01016": "SBILIFE",
    "NSE_EQ|INE114A01011": "EICHERMOT", "NSE_EQ|INE047A01021": "GRASIM",
    "NSE_EQ|INE095A01012": "INDUSINDBK", "NSE_EQ|INE918I01018": "BAJAJFINSV",
    "NSE_EQ|INE158A01026": "HEROMOTOCO", "NSE_EQ|INE361B01024": "DIVISLAB",
    "NSE_EQ|INE059A01026": "CIPLA", "NSE_EQ|INE437A01024": "APOLLOHOSP",
    "NSE_EQ|INE364U01010": "ADANIGREEN", "NSE_EQ|INE029A01011": "BPCL",
    "NSE_EQ|INE216A01030": "BRITANNIA", "NSE_EQ|INE214T01019": "LTIM",
    "NSE_EQ|INE849A01020": "TRENT", "NSE_EQ|INE721A01013": "SHRIRAMFIN",
    "NSE_EQ|INE263A01024": "BEL", "NSE_EQ|INE511C01022": "POONAWALLA",
    "NSE_EQ|INE594E01019": "HINDUNILVR",
}

# Global tracking
DAILY_STATS = {
    "total_alerts": 0, 
    "indices_count": 0, 
    "stocks_count": 0, 
    "start_time": None,
    "api_calls": 0
}

print("="*70)
print("🚀 COMPLETE MARKET MONITOR - v9.1 (TRADINGVIEW CLEAN EDITION)")
print("="*70)

def make_api_request(url: str, headers: dict, max_retries: int = 3, timeout: int = 15) -> Optional[dict]:
    """Enhanced API request with retry mechanism and better error handling"""
    for attempt in range(max_retries):
        try:
            DAILY_STATS["api_calls"] += 1
            resp = requests.get(url, headers=headers, timeout=timeout)
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                print(f"  ⚠️ Rate limited. Waiting {wait_time}s...")
                time_sleep.sleep(wait_time)
            else:
                print(f"  ⚠️ API error {resp.status_code} on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time_sleep.sleep(2)
                    
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Request error: {e} on attempt {attempt + 1}")
        
        if attempt < max_retries - 1:
            time_sleep.sleep(1)
    
    print(f"  ❌ Failed after {max_retries} attempts")
    return None

def get_expiries(instrument_key: str) -> List[str]:
    """Get available expiries with enhanced error handling"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
    
    data = make_api_request(url, headers)
    if data and 'data' in data:
        contracts = data['data']
        expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
        return expiries
    
    return []

def get_next_expiry(instrument_key: str, expiry_day: int = 1) -> str:
    """Get next expiry with fallback calculation"""
    expiries = get_expiries(instrument_key)
    
    if not expiries:
        today = datetime.now(IST)
        days_ahead = expiry_day - today.weekday()
        if days_ahead <= 0: 
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    today = datetime.now(IST).date()
    future_expiries = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    return min(future_expiries) if future_expiries else expiries[0]

def get_option_chain(instrument_key: str, expiry: str) -> List[dict]:
    """Get complete option chain data"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    
    data = make_api_request(url, headers, timeout=20)
    if data and 'data' in data:
        strikes = data['data']
        return sorted(strikes, key=lambda x: x.get('strike_price', 0))
    
    return []

def get_spot_price(instrument_key: str) -> float:
    """Get spot price with enhanced reliability"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    data = make_api_request(url, headers)
    if data and 'data' in data:
        quote_data = data['data']
        if quote_data:
            first_key = list(quote_data.keys())[0]
            ltp = quote_data[first_key].get('last_price', 0)
            if ltp:
                return float(ltp)
    
    return 0.0

def get_historical_data(instrument_key: str, symbol: str) -> Tuple[List, int]:
    """Get combined historical and intraday data with improved processing"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    all_candles = []
    
    # Get historical data (15 days)
    try:
        to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
        from_date = (datetime.now(IST) - timedelta(days=15)).strftime('%Y-%m-%d')
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_date}/{from_date}"
        
        data = make_api_request(url, headers, timeout=20)
        if data and data.get('status') == 'success':
            for candle in data.get('data', {}).get('candles', []):
                # Convert 30min to 5min candles
                split_candles = split_30min_to_5min(candle)
                all_candles.extend(split_candles)
    except Exception as e:
        print(f"  ⚠️ Historical data error for {symbol}: {e}")
    
    # Get intraday data
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        data = make_api_request(url, headers, timeout=20)
        
        if data and data.get('status') == 'success':
            candles_1min = data.get('data', {}).get('candles', [])
            if candles_1min:
                # Convert to DataFrame for efficient resampling
                df = pd.DataFrame(candles_1min, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.set_index('ts').astype(float)
                
                # Resample to 5 minutes
                df_resampled = df.resample('5min').agg({
                    'o': 'first', 'h': 'max', 'l': 'min', 
                    'c': 'last', 'v': 'sum', 'oi': 'last'
                }).dropna()
                
                # Convert back to list format
                intraday_candles = [
                    [idx.isoformat(), r['o'], r['h'], r['l'], r['c'], r['v'], r['oi']] 
                    for idx, r in df_resampled.iterrows()
                ]
                all_candles.extend(intraday_candles)
    except Exception as e:
        print(f"  ⚠️ Intraday data error for {symbol}: {e}")
    
    # Sort all candles by timestamp
    all_candles = sorted(all_candles, key=lambda x: x[0])
    
    # Count historical candles (before today)
    today = datetime.now(IST).date()
    hist_count = len([
        c for c in all_candles 
        if datetime.fromisoformat(c[0]).astimezone(IST).date() < today
    ])
    
    return all_candles, hist_count

def split_30min_to_5min(candle_30min: list) -> List[list]:
    """Split 30-minute candle into 5-minute candles"""
    try:
        ts_str, o, h, l, c, v, oi = candle_30min
        dt_start = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(IST)
        candles_5min = []
        
        for i in range(6):
            c_time = dt_start + timedelta(minutes=i * 5)
            # Linear interpolation for OHLC
            progress = i / 6
            c_open = o + (c - o) * progress
            c_close = o + (c - o) * ((i + 1) / 6)
            c_high = max(h, c_open, c_close)
            c_low = min(l, c_open, c_close)
            
            candles_5min.append([
                c_time.isoformat(), 
                float(c_open), 
                float(c_high), 
                float(c_low), 
                float(c_close), 
                float(v) / 6, 
                float(oi)
            ])
        return candles_5min
    except Exception as e:
        print(f"  ⚠️ Candle splitting error: {e}")
        return []

def create_tradingview_chart(candles: List[list], symbol: str, spot_price: float, hist_count: int) -> Optional[io.BytesIO]:
    """Create clean TradingView-style chart without indicators"""
    if not candles or len(candles) < 10:
        return None
    
    # Convert to DataFrame
    data = []
    for c in candles:
        try:
            ts = datetime.fromisoformat(c[0].replace("Z", "+00:00")).astimezone(IST)
            if time(9, 15) <= ts.time() <= time(15, 30):
                data.append({
                    'ts': ts, 'o': float(c[1]), 'h': float(c[2]), 
                    'l': float(c[3]), 'c': float(c[4]), 'v': int(c[5])
                })
        except (ValueError, TypeError):
            continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    # Create clean figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), 
                                  gridspec_kw={'height_ratios': [3, 1]}, 
                                  facecolor='#0e1217')
    
    # Set background colors
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0e1217')
        ax.tick_params(axis='both', colors='#b2b5be', labelsize=11)
        ax.grid(True, alpha=0.2, color='#363a45')
        for spine in ax.spines.values():
            spine.set_color('#1e222d')
    
    # Plot clean candlesticks
    x_positions = range(len(df))
    for i, (idx, row) in enumerate(df.iterrows()):
        color = '#26a69a' if row['c'] >= row['o'] else '#ef5350'
        
        # High-Low line (thin wick)
        ax1.plot([i, i], [row['l'], row['h']], color=color, linewidth=0.8, zorder=1)
        
        # Open-Close body (thicker)
        body_height = abs(row['c'] - row['o'])
        body_bottom = min(row['o'], row['c'])
        
        if body_height > 0:
            # Proper candlestick body
            rect = Rectangle((i - 0.35, body_bottom), 0.7, body_height, 
                           facecolor=color, alpha=1.0, zorder=2,
                           edgecolor=color, linewidth=0.5)
            ax1.add_patch(rect)
        else:
            # Doji - just a horizontal line
            ax1.plot([i - 0.3, i + 0.3], [body_bottom, body_bottom], 
                    color=color, linewidth=1.5, zorder=2)
    
    # Current spot price line
    ax1.axhline(y=spot_price, color='#00e676', linestyle='--', 
                linewidth=2.0, alpha=0.9, label=f'Spot: ₹{spot_price:.2f}')
    
    # Volume bars
    for i, (idx, row) in enumerate(df.iterrows()):
        color = '#26a69a' if row['c'] >= row['o'] else '#ef5350'
        ax2.bar(i, row['v'], width=0.8, color=color, alpha=0.7)
    
    # Historical/intraday separator
    if 0 < hist_count < len(df):
        ax1.axvline(x=hist_count - 0.5, color='#ffa726', linestyle='--', 
                   linewidth=2.0, alpha=0.8, label='Live Data Start')
        ax2.axvline(x=hist_count - 0.5, color='#ffa726', linestyle='--', 
                   linewidth=2.0, alpha=0.8)
    
    # Clean legends
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1e222d', 
               edgecolor='#363a45', labelcolor='#b2b5be')
    
    # Professional titles and labels
    ax1.set_title(
        f'📊 {symbol} • TRADINGVIEW STYLE CHART • {datetime.now(IST).strftime("%d %b %Y • %I:%M:%S %p IST")}',
        color='#ffffff', fontsize=16, fontweight='bold', pad=20
    )
    
    ax1.set_ylabel('Price (₹)', color='#b2b5be', fontsize=12)
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=12)
    ax2.set_xlabel('Time (5min intervals)', color='#b2b5be', fontsize=12)
    
    # X-axis formatting
    tick_positions = []
    tick_labels = []
    last_date = None
    
    for i, (idx, row) in enumerate(df.iterrows()):
        current_date = row['ts'].strftime('%d-%m')
        if current_date != last_date:
            tick_positions.append(i)
            tick_labels.append(row['ts'].strftime('%d %b\n%H:%M'))
            last_date = current_date
    
    # Limit number of x-axis labels for cleanliness
    if len(tick_positions) > 8:
        step = max(1, len(tick_positions) // 6)
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='#b2b5be', fontsize=10)
        ax.set_xlim(0, len(df))
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.08)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0e1217', 
                bbox_inches='tight', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def format_option_chain_message(symbol: str, spot: float, expiry: str, strikes: List[dict]) -> Optional[str]:
    """Format comprehensive option chain data with all metrics"""
    if not strikes:
        return None
    
    # Find ATM strike
    atm_strike = min(strikes, key=lambda x: abs(x.get('strike_price', 0) - spot))
    atm_index = strikes.index(atm_strike)
    
    # Select strikes around ATM (7 on each side for comprehensive view)
    start_idx = max(0, atm_index - 7)
    end_idx = min(len(strikes), atm_index + 8)
    selected = strikes[start_idx:end_idx]
    
    # Calculate comprehensive totals and metrics
    total_ce_oi = total_pe_oi = total_ce_volume = total_pe_volume = 0
    total_ce_oi_change = total_pe_oi_change = 0
    max_pain_oi = 0
    max_pain_strike = 0
    
    for strike in strikes:
        sp = strike.get('strike_price', 0)
        ce_data = strike.get('call_options', {}).get('market_data', {})
        pe_data = strike.get('put_options', {}).get('market_data', {})
        
        ce_oi = ce_data.get('oi', 0)
        pe_oi = pe_data.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_volume += ce_data.get('volume', 0)
        total_pe_volume += pe_data.get('volume', 0)
        total_ce_oi_change += ce_data.get('oi_change', 0)
        total_pe_oi_change += pe_data.get('oi_change', 0)
        
        # Calculate max pain (simplified)
        strike_oi = ce_oi + pe_oi
        if strike_oi > max_pain_oi:
            max_pain_oi = strike_oi
            max_pain_strike = sp
    
    # Build comprehensive message
    msg = f"🎯 *{symbol} - COMPLETE OPTION CHAIN ANALYSIS*\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*Spot Price:* `₹{spot:,.2f}`\n"
    msg += f"*Expiry:* `{expiry}`\n"
    msg += f"*ATM Strike:* `₹{atm_strike.get('strike_price', 0):,.0f}`\n"
    msg += f"*Max Pain:* `₹{max_pain_strike:,.0f}`\n\n"
    
    # Comprehensive option chain table
    msg += "```\n"
    msg += "┌───────────── CALLS ─────────────┬────────── STRIKE ──────────┬───────────── PUTS ─────────────┐\n"
    msg += "│    OI      Vol    Chg     LTP   │           Price            │   LTP    Chg     Vol      OI    │\n"
    msg += "├─────────────────────────────────┼────────────────────────────┼─────────────────────────────────┤\n"
    
    for strike in selected:
        sp = strike.get('strike_price', 0)
        marker = "⚡" if sp == atm_strike.get('strike_price', 0) else "│"
        
        ce_data = strike.get('call_options', {}).get('market_data', {})
        pe_data = strike.get('put_options', {}).get('market_data', {})
        
        # Call options data
        ce_oi = fmt_val(ce_data.get('oi', 0))
        ce_vol = fmt_val(ce_data.get('volume', 0))
        ce_oi_chg = fmt_change(ce_data.get('oi_change', 0))
        ce_ltp = f"{ce_data.get('ltp', 0):6.1f}"
        
        # Put options data
        pe_oi = fmt_val(pe_data.get('oi', 0))
        pe_vol = fmt_val(pe_data.get('volume', 0))
        pe_oi_chg = fmt_change(pe_data.get('oi_change', 0))
        pe_ltp = f"{pe_data.get('ltp', 0):6.1f}"
        
        msg += f"│ {ce_oi:>7} {ce_vol:>6} {ce_oi_chg:>5} {ce_ltp} │ {marker} ₹{sp:>8,.0f} {marker} │ {pe_ltp} {pe_oi_chg:>5} {pe_vol:>6} {pe_oi:>7} │\n"
    
    msg += "└─────────────────────────────────┴────────────────────────────┴─────────────────────────────────┘\n"
    msg += "```\n\n"
    
    # Advanced market sentiment and statistics
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_volume = total_pe_volume / total_ce_volume if total_ce_volume > 0 else 0
    
    # Determine comprehensive sentiment
    if pcr_oi > 1.5:
        pcr_oi_sentiment = "🟢 STRONG BULLISH"
    elif pcr_oi > 1.2:
        pcr_oi_sentiment = "🟢 BULLISH"
    elif pcr_oi > 0.8:
        pcr_oi_sentiment = "🟡 NEUTRAL"
    elif pcr_oi > 0.5:
        pcr_oi_sentiment = "🔴 BEARISH"
    else:
        pcr_oi_sentiment = "🔴 STRONG BEARISH"
    
    # OI change sentiment
    oi_change_sentiment = ""
    if total_ce_oi_change > 0 and total_pe_oi_change > 0:
        oi_change_sentiment = "📈 Both CE/PE OI Rising"
    elif total_ce_oi_change < 0 and total_pe_oi_change < 0:
        oi_change_sentiment = "📉 Both CE/PE OI Falling"
    elif total_ce_oi_change > total_pe_oi_change:
        oi_change_sentiment = "🔴 CE OI Rising Faster"
    else:
        oi_change_sentiment = "🟢 PE OI Rising Faster"
    
    msg += "*MARKET SENTIMENT & STATISTICS:*\n"
    msg += f"• *PCR (OI):* `{pcr_oi:.3f}` {pcr_oi_sentiment}\n"
    msg += f"• *PCR (Volume):* `{pcr_volume:.3f}`\n"
    msg += f"• *Total CE OI:* `{fmt_val(total_ce_oi)}` | *Total PE OI:* `{fmt_val(total_pe_oi)}`\n"
    msg += f"• *OI Change CE:* `{fmt_change(total_ce_oi_change)}` | *OI Change PE:* `{fmt_change(total_pe_oi_change)}`\n"
    msg += f"• *Total CE Volume:* `{fmt_val(total_ce_volume)}` | *Total PE Volume:* `{fmt_val(total_pe_volume)}`\n"
    msg += f"• *OI Trend:* {oi_change_sentiment}\n\n"
    
    msg += f"🕒 *Last Updated:* {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

def fmt_val(value: float) -> str:
    """Format large numbers with K, L, Cr suffixes"""
    if value >= 10000000:
        return f"{value/10000000:.1f}Cr"
    elif value >= 100000:
        return f"{value/100000:.1f}L"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{int(value)}"

def fmt_change(change: float) -> str:
    """Format change values with +/- signs and colors"""
    if change > 0:
        return f"+{fmt_val(change)}"
    elif change < 0:
        return f"-{fmt_val(abs(change))}"
    else:
        return " 0 "

async def send_telegram_message(bot: Bot, text: str = None, photo: io.BytesIO = None, caption: str = None) -> bool:
    """Send message to Telegram with enhanced error handling"""
    try:
        if photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID, 
                photo=photo, 
                caption=caption, 
                parse_mode='Markdown'
            )
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=text, 
                parse_mode='Markdown'
            )
        DAILY_STATS["total_alerts"] += 1
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

async def process_instrument(bot: Bot, key: str, name: str, expiry_day: int, 
                           is_stock: bool = False, idx: int = 0, total: int = 0) -> bool:
    """Process a single instrument with comprehensive data"""
    prefix = f"[{idx}/{total}] STOCK:" if is_stock else "INDEX:"
    print(f"\n{prefix} {name}")
    
    try:
        # Get spot price
        spot = get_spot_price(key)
        if spot == 0:
            print(f"  ❌ Failed to get spot price for {name}")
            return False
        print(f"  ✅ Spot: ₹{spot:.2f}")
        
        # Get option chain
        expiry = get_next_expiry(key, expiry_day=expiry_day)
        strikes = get_option_chain(key, expiry)
        
        if strikes:
            msg = format_option_chain_message(name, spot, expiry, strikes)
            if msg:
                success = await send_telegram_message(bot, text=msg)
                if success:
                    print("    📤 Comprehensive option chain sent")
                else:
                    print("    ❌ Failed to send option chain")
        else:
            print("    ⚠️ No option chain data found")
        
        # Get clean TradingView chart data
        candles, hist_count = get_historical_data(key, name)
        if candles and len(candles) >= 10:
            chart = create_tradingview_chart(candles, name, spot, hist_count)
            if chart:
                caption = f"📈 *{name}* • Spot: `₹{spot:,.2f}` • {datetime.now(IST).strftime('%I:%M %p IST')}"
                success = await send_telegram_message(bot, photo=chart, caption=caption)
                if success:
                    print("    📤 TradingView chart sent")
                else:
                    print("    ❌ Failed to send chart")
        else:
            print("    ⚠️ Insufficient data for chart")
        
        # Update statistics
        if is_stock:
            DAILY_STATS["stocks_count"] += 1
        else:
            DAILY_STATS["indices_count"] += 1
            
        return True
        
    except Exception as e:
        print(f"  ❌ Processing error for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def fetch_all(bot: Bot):
    """Fetch data for all instruments"""
    now = datetime.now(IST)
    print(f"\n{'='*60}")
    print(f"🚀 MARKET DATA RUN: {now.strftime('%I:%M:%S %p IST')}")
    print(f"{'='*60}")
    
    # Send start message
    header = f"🚀 *MARKET DATA UPDATE INITIATED*\n_Time: {now.strftime('%I:%M:%S %p IST')}_\n_Processing 4 indices + {len(NIFTY50_STOCKS)} stocks..._"
    await send_telegram_message(bot, text=header)
    
    # Reset counters
    DAILY_STATS["indices_count"] = 0
    DAILY_STATS["stocks_count"] = 0
    
    # Process indices
    print(f"\n📊 PROCESSING INDICES:")
    for i, (key, info) in enumerate(INDICES.items(), 1):
        await process_instrument(bot, key, info["name"], info["expiry_day"])
        await asyncio.sleep(1.5)
    
    # Process stocks
    print(f"\n📈 PROCESSING STOCKS:")
    stock_items = list(NIFTY50_STOCKS.items())
    for i, (key, symbol) in enumerate(stock_items, 1):
        await process_instrument(
            bot, key, symbol, 3, 
            is_stock=True, idx=i, total=len(stock_items)
        )
        await asyncio.sleep(1.2)
    
    # Send completion summary
    summary = (
        f"✅ *MARKET UPDATE COMPLETE*\n\n"
        f"📊 *Indices Processed:* {DAILY_STATS['indices_count']}/4\n"
        f"📈 *Stocks Processed:* {DAILY_STATS['stocks_count']}/{len(NIFTY50_STOCKS)}\n"
        f"📡 *Total Alerts Today:* {DAILY_STATS['total_alerts']}\n"
        f"🔢 *API Calls Made:* {DAILY_STATS['api_calls']}\n\n"
        f"⏰ *Next Update:* 5 minutes\n"
        f"🕒 *Completed at:* {datetime.now(IST).strftime('%I:%M:%S %p IST')}"
    )
    await send_telegram_message(bot, text=summary)
    
    print(f"\n✅ CYCLE COMPLETED:")
    print(f"   • Indices: {DAILY_STATS['indices_count']}/4")
    print(f"   • Stocks: {DAILY_STATS['stocks_count']}/{len(NIFTY50_STOCKS)}")
    print(f"   • API Calls: {DAILY_STATS['api_calls']}")
    print(f"   • Total Alerts: {DAILY_STATS['total_alerts']}")

async def main():
    """Main application loop"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    while True:
        now = datetime.now(IST)
        is_market_hours = (now.weekday() < 5) and (time(9, 15) <= now.time() <= time(15, 35))
        
        if is_market_hours:
            if DAILY_STATS["start_time"] is None:
                DAILY_STATS["start_time"] = now
                DAILY_STATS["api_calls"] = 0
            
            await fetch_all(bot)
            print(f"\n⏳ Next run in 5 minutes...")
            await asyncio.sleep(300)
        else:
            print(f"\n💤 Market closed. Current time: {now.strftime('%I:%M %p IST')}")
            
            if now.hour >= 16 and DAILY_STATS["start_time"] is not None:
                print("🔄 Resetting daily stats for next trading day...")
                DAILY_STATS.update({
                    "total_alerts": 0,
                    "indices_count": 0, 
                    "stocks_count": 0,
                    "start_time": None,
                    "api_calls": 0
                })
            
            await asyncio.sleep(900)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
