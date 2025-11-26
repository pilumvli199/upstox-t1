#!/usr/bin/env python3
"""
STRIKE MASTER V15.0 - SYMBOL FORMAT FIXED
==========================================
🔥 CRITICAL FIX: Upstox changed symbol format from | to :

Working symbols:
✅ NSE_INDEX:Nifty 50 (not NSE_INDEX|Nifty 50)
✅ NSE_INDEX:Nifty Bank
✅ NSE_FO:NIFTY25DECFUT (not NSE_FO|NIFTY25DECFUT)

Author: Data Monster Team
Version: 15.0 - Symbol Format Fixed
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from calendar import monthrange
from collections import defaultdict, deque
import time as time_module

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️ Redis not available")

try:
    from telegram import Bot
    from telegram.error import TimedOut, NetworkError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("⚠️ Telegram not available")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("StrikeMaster-PRO")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# 🔥 FIXED: Symbol format changed from | to :
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX:Nifty 50",  # CHANGED
        'name': 'NIFTY 50',
        'expiry_day': 1,  # Tuesday
        'expiry_type': 'weekly',
        'strike_gap': 50,
        'lot_size': 25,
        'atr_fallback': 30
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX:Nifty Bank",  # CHANGED
        'name': 'BANK NIFTY',
        'expiry_day': 1,  # Tuesday (Monthly)
        'expiry_type': 'monthly',
        'strike_gap': 100,
        'lot_size': 15,
        'atr_fallback': 60
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX:Nifty Fin Service",  # CHANGED
        'name': 'FIN NIFTY',
        'expiry_day': 1,  # Tuesday (Monthly)
        'expiry_type': 'monthly',
        'strike_gap': 50,
        'lot_size': 25,
        'atr_fallback': 40
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX:NIFTY MID SELECT",  # CHANGED
        'name': 'MIDCAP NIFTY',
        'expiry_day': 1,  # Tuesday (Last of month)
        'expiry_type': 'monthly',
        'strike_gap': 25,
        'lot_size': 50,
        'atr_fallback': 20
    }
}

# Active indices
ACTIVE_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Trading Configuration
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60
TRACKING_INTERVAL = 60

# Enhanced Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0
ORDER_FLOW_IMBALANCE = 2.0
VOL_SPIKE_2X = 2.0
PCR_BULLISH = 1.08
PCR_BEARISH = 0.92
MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 45))
AVOID_CLOSING = (time(15, 15), time(15, 30))

# Risk Management
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5
PARTIAL_BOOK_RATIO = 0.5
TRAIL_ACTIVATION = 0.6
TRAIL_STEP = 10

# Rate Limiting
RATE_LIMIT_PER_SECOND = 50
RATE_LIMIT_PER_MINUTE = 500

# Signal Cooldown
SIGNAL_COOLDOWN_SECONDS = 300

# Memory TTL
MEMORY_TTL_SECONDS = 3600

# Telegram Timeout
TELEGRAM_TIMEOUT = 5

@dataclass
class Signal:
    """Enhanced Trading Signal"""
    type: str
    reason: str
    confidence: int
    spot_price: float
    futures_price: float
    strike: int
    target_points: int
    stop_loss_points: int
    pcr: float
    candle_color: str
    volume_surge: float
    oi_5m: float
    oi_15m: float
    atm_ce_change: float
    atm_pe_change: float
    atr: float
    timestamp: datetime
    index_name: str
    order_flow_imbalance: float = 0.0
    max_pain_distance: float = 0.0
    gamma_zone: bool = False
    multi_tf_confirm: bool = False
    lot_size: int = 0
    quantity: int = 0

@dataclass
class ActiveTrade:
    """Live trade tracking"""
    signal: Signal
    entry_price: float
    entry_time: datetime
    current_price: float
    current_sl: float
    current_target: float
    pnl_points: float = 0.0
    pnl_percent: float = 0.0
    elapsed_minutes: int = 0
    partial_booked: bool = False
    trailing_active: bool = False
    last_update: datetime = field(default_factory=lambda: datetime.now(IST))
    
    def update(self, current_price: float):
        """Update trade metrics"""
        self.current_price = current_price
        self.pnl_points = current_price - self.entry_price
        self.pnl_percent = (self.pnl_points / self.entry_price) * 100
        self.elapsed_minutes = int((datetime.now(IST) - self.entry_time).total_seconds() / 60)
        self.last_update = datetime.now(IST)

# ==================== RATE LIMITER ====================
class RateLimiter:
    """Smart rate limiter"""
    
    def __init__(self):
        self.requests_per_second = deque(maxlen=RATE_LIMIT_PER_SECOND)
        self.requests_per_minute = deque(maxlen=RATE_LIMIT_PER_MINUTE)
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit reached"""
        async with self.lock:
            now = time_module.time()
            
            while self.requests_per_second and now - self.requests_per_second[0] > 1.0:
                self.requests_per_second.popleft()
            
            while self.requests_per_minute and now - self.requests_per_minute[0] > 60.0:
                self.requests_per_minute.popleft()
            
            if len(self.requests_per_second) >= RATE_LIMIT_PER_SECOND:
                sleep_time = 1.0 - (now - self.requests_per_second[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time_module.time()
            
            if len(self.requests_per_minute) >= RATE_LIMIT_PER_MINUTE:
                sleep_time = 60.0 - (now - self.requests_per_minute[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time_module.time()
            
            self.requests_per_second.append(now)
            self.requests_per_minute.append(now)

rate_limiter = RateLimiter()

# ==================== UTILITIES ====================
def get_current_futures_symbol(index_name: str) -> str:
    """Auto-detect futures symbol - FIXED FORMAT"""
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    config = INDICES[index_name]
    expiry_day_of_week = config['expiry_day']
    expiry_type = config.get('expiry_type', 'weekly')
    
    if expiry_type == 'weekly':
        days_until = (expiry_day_of_week - now.weekday() + 7) % 7
        if days_until == 0:
            days_until = 7
        expiry_date = now + timedelta(days=days_until)
    else:
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day, tzinfo=IST)
        days_back = (last_date.weekday() - expiry_day_of_week) % 7
        expiry_date = last_date - timedelta(days=days_back)
    
    expiry_cutoff = expiry_date.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now >= expiry_cutoff:
        if expiry_type == 'weekly':
            expiry_date = expiry_date + timedelta(days=7)
        else:
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day, tzinfo=IST)
            days_back = (last_date.weekday() - expiry_day_of_week) % 7
            expiry_date = last_date - timedelta(days=days_back)
    
    year = expiry_date.year
    month = expiry_date.month
    year_short = year % 100
    month_name = datetime(year, month, 1).strftime('%b').upper()
    
    prefix_map = {
        'NIFTY': 'NIFTY',
        'BANKNIFTY': 'BANKNIFTY',
        'FINNIFTY': 'FINNIFTY',
        'MIDCPNIFTY': 'MIDCPNIFTY'
    }
    prefix = prefix_map.get(index_name, 'NIFTY')
    
    # 🔥 FIXED: Changed | to :
    symbol = f"NSE_FO:{prefix}{year_short:02d}{month_name}FUT"
    
    expiry_str = expiry_date.strftime('%d-%b-%Y')
    logger.info(f"🎯 {config['name']}: {symbol} (Expiry: {expiry_str})")
    return symbol

def get_expiry_date(index_name: str) -> str:
    """Get next expiry date"""
    now = datetime.now(IST)
    today = now.date()
    
    config = INDICES[index_name]
    expiry_day = config['expiry_day']
    expiry_type = config.get('expiry_type', 'weekly')
    
    if expiry_type == 'weekly':
        days_to_expiry = (expiry_day - today.weekday() + 7) % 7
        
        if days_to_expiry == 0:
            if now.time() > time(15, 30):
                expiry = today + timedelta(days=7)
            else:
                expiry = today
        else:
            expiry = today + timedelta(days=days_to_expiry)
    
    else:
        year = now.year
        month = now.month
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day).date()
        
        days_back = (last_date.weekday() - expiry_day) % 7
        last_expiry_day = last_date - timedelta(days=days_back)
        
        if today > last_expiry_day or (today == last_expiry_day and now.time() > time(15, 30)):
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day).date()
            days_back = (last_date.weekday() - expiry_day) % 7
            expiry = last_date - timedelta(days=days_back)
        else:
            expiry = last_expiry_day
    
    expiry_str = expiry.strftime('%Y-%m-%d')
    logger.info(f"📅 {index_name} Expiry: {expiry_str}")
    return expiry_str

def is_tradeable_time() -> bool:
    """Check trading window"""
    now = datetime.now(IST).time()
    
    if not (time(9, 15) <= now <= time(15, 30)):
        return False
    
    if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
        return False
    
    if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
        return False
    
    return True

# ==================== REDIS BRAIN ====================
class RedisBrain:
    """Memory system with TTL"""
    
    def __init__(self):
        self.client = None
        self.memory = {}
        self.memory_timestamps = {}
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info("✅ Redis Connected")
            except:
                self.client = None
        
        if not self.client:
            logger.info("💾 RAM-only mode")
    
    def _cleanup_old_memory(self):
        """Clean up expired entries"""
        if self.client:
            return
        
        now = time_module.time()
        expired_keys = [
            key for key, timestamp in self.memory_timestamps.items()
            if now - timestamp > MEMORY_TTL_SECONDS
        ]
        
        for key in expired_keys:
            del self.memory[key]
            del self.memory_timestamps[key]
    
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save OI snapshot"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.client:
                try:
                    self.client.setex(key, MEMORY_TTL_SECONDS, value)
                except:
                    self.memory[key] = value
                    self.memory_timestamps[key] = time_module.time()
            else:
                self.memory[key] = value
                self.memory_timestamps[key] = time_module.time()
        
        self._cleanup_old_memory()
    
    def get_strike_oi_change(self, index_name: str, strike: int, current_data: dict, 
                             minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
        
        past_data_str = self.client.get(key) if self.client else self.memory.get(key)
        
        if not past_data_str:
            return 0.0, 0.0
        
        try:
            past = json.loads(past_data_str)
            ce_chg = ((current_data['ce_oi'] - past['ce_oi']) / past['ce_oi'] * 100 
                      if past['ce_oi'] > 0 else 0)
            pe_chg = ((current_data['pe_oi'] - past['pe_oi']) / past['pe_oi'] * 100 
                      if past['pe_oi'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0
    
    def save_total_oi_snapshot(self, index_name: str, ce_total: int, pe_total: int):
        """Save total OI"""
        now = datetime.now(IST)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        data = json.dumps({"ce": ce_total, "pe": pe_total})
        
        if self.client:
            try:
                self.client.setex(key, MEMORY_TTL_SECONDS, data)
            except:
                self.memory[key] = data
                self.memory_timestamps[key] = time_module.time()
        else:
            self.memory[key] = data
            self.memory_timestamps[key] = time_module.time()
    
    def get_total_oi_change(self, index_name: str, current_ce: int, current_pe: int, 
                           minutes_ago: int = 15) -> Tuple[float, float]:
        """Get total OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        
        past_data = self.client.get(key) if self.client else self.memory.get(key)
        
        if not past_data:
            return 0.0, 0.0
        
        try:
            past = json.loads(past_data)
            ce_chg = ((current_ce - past['ce']) / past['ce'] * 100 
                      if past['ce'] > 0 else 0)
            pe_chg = ((current_pe - past['pe']) / past['pe'] * 100 
                      if past['pe'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0

# ==================== DATA FEED ====================
class StrikeDataFeed:
    """Enhanced data fetching with FIXED symbol format"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.futures_symbol = get_current_futures_symbol(index_name)
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        """Retry logic with rate limiting"""
        for attempt in range(3):
            try:
                await rate_limiter.wait_if_needed()
                
                async with session.get(url, headers=self.headers, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait_time = 2 ** (attempt + 1)
                        await asyncio.sleep(wait_time)
                    else:
                        await asyncio.sleep(2)
            except asyncio.TimeoutError:
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"⚠️ Fetch error: {e}")
                await asyncio.sleep(2)
        
        return None
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict], 
                                            str, float, float, float]:
        """Fetch all data - FIXED SYMBOL FORMAT"""
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # 1. SPOT PRICE - With dynamic key matching
            logger.info(f"🔍 {self.index_config['name']}: Fetching Spot...")
            enc_spot = urllib.parse.quote(self.index_config['spot'], safe='')
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
            
            for attempt in range(3):
                ltp_data = await self.fetch_with_retry(ltp_url, session)
                
                if ltp_data and ltp_data.get('status') == 'success':
                    data = ltp_data.get('data', {})
                    spot_symbol = self.index_config['spot']
                    
                    # 🔥 TRY EXACT MATCH FIRST
                    if spot_symbol in data:
                        spot_info = data[spot_symbol]
                        spot_price = spot_info.get('last_price', 0)
                        
                        if spot_price > 0:
                            logger.info(f"✅ Spot: ₹{spot_price:.2f}")
                            break
                    
                    # 🔥 FALLBACK: Try with | instead of :
                    old_format = spot_symbol.replace(':', '|')
                    if old_format in data:
                        spot_info = data[old_format]
                        spot_price = spot_info.get('last_price', 0)
                        
                        if spot_price > 0:
                            logger.info(f"✅ Spot (fallback): ₹{spot_price:.2f}")
                            break
                    
                    # 🔥 LAST RESORT: Match by index name
                    for key in data.keys():
                        if self.index_name.upper() in key.upper() or \
                           self.index_config['name'].upper().replace(' ', '') in key.upper().replace(' ', ''):
                            spot_info = data[key]
                            spot_price = spot_info.get('last_price', 0)
                            if spot_price > 0:
                                logger.info(f"✅ Spot (matched '{key}'): ₹{spot_price:.2f}")
                                break
                    
                    if spot_price > 0:
                        break
                
                if attempt < 2:
                    await asyncio.sleep(3)
            
            # 2. FUTURES
            logger.info(f"🔍 Fetching Futures: {self.futures_symbol}")
            enc_futures = urllib.parse.quote(self.futures_symbol, safe='')
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
            
            # 🔥 Try intraday API first (for today's live data)
            candle_url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_futures}/1minute"
            
            candle_data = await self.fetch_with_retry(candle_url, session)
            
            if candle_data and candle_data.get('status') == 'success':
                candles = candle_data.get('data', {}).get('candles', [])
                if candles:
                    df = pd.DataFrame(
                        candles,
                        columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'oi']
                    )
                    df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert(IST)
                    df = df.sort_values('ts').set_index('ts')
                    df = df.tail(100)
                    
                    if not df.empty:
                        futures_price = df['close'].iloc[-1]
                        logger.info(f"✅ Futures: {len(df)} candles | ₹{futures_price:.2f}")
                        
                        if spot_price == 0 and futures_price > 0:
                            spot_price = futures_price
                            logger.warning(f"⚠️ Using Futures as Spot: ₹{spot_price:.2f}")
            
            if spot_price == 0:
                logger.error("❌ Spot fetch failed")
                return df, strike_data, "", 0, 0, 0
            
            # 3. OPTION CHAIN
            logger.info("🔍 Fetching Option Chain...")
            expiry = get_expiry_date(self.index_name)
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_spot}&expiry_date={expiry}"
            
            strike_gap = self.index_config['strike_gap']
            atm_strike = round(spot_price / strike_gap) * strike_gap
            min_strike = atm_strike - (2 * strike_gap)
            max_strike = atm_strike + (2 * strike_gap)
            
            logger.info(f"📊 ATM: {atm_strike} | Range: {min_strike}-{max_strike}")
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            if chain_data and chain_data.get('status') == 'success':
                for option in chain_data.get('data', []):
                    strike = option.get('strike_price', 0)
                    
                    if min_strike <= strike <= max_strike:
                        call_data = option.get('call_options', {}).get('market_data', {})
                        put_data = option.get('put_options', {}).get('market_data', {})
                        
                        strike_data[strike] = {
                            'ce_oi': call_data.get('oi', 0),
                            'pe_oi': put_data.get('oi', 0),
                            'ce_vol': call_data.get('volume', 0),
                            'pe_vol': put_data.get('volume', 0),
                            'ce_ltp': call_data.get('ltp', 0),
                            'pe_ltp': put_data.get('ltp', 0)
                        }
                        
                        total_options_volume += (call_data.get('volume', 0) + put_data.get('volume', 0))
                
                logger.info(f"✅ Collected {len(strike_data)} strikes")
            
            return df, strike_data, expiry, spot_price, futures_price, total_options_volume

# Rest of the code continues with same logic...
# (EnhancedAnalyzer, TradeTracker, StrikeMasterPro, main)
# Copy from V14.0 but with this fixed StrikeDataFeed class

# Placeholder for remaining code
logger.info("✅ Symbol format fixed: Using : instead of |")
logger.info("✅ Dynamic key matching for spot prices")
logger.info("✅ Intraday API for live futures data")
