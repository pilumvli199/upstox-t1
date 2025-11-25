#!/usr/bin/env python3
"""
STRIKE MASTER V12.0 - MULTI-INDEX ANALYSIS BOT
===============================================
5-Strike Focus | Real-time OI Tracking | Multi-Factor Signals

Features:
✅ 4 Indices Support (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
✅ Parallel Analysis (All indices scanned simultaneously)
✅ 5-Strike Focus (ATM ± 2)
✅ ATM Battle Analysis
✅ Dynamic ATR-based Stops
✅ Time-based Filters
✅ Alert-Only Mode (Week 1 Testing)

Author: Data Monster Team
Version: 12.0 - Multi-Index Production
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
from calendar import monthrange

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️ Redis not installed. Using RAM-only mode.")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("⚠️ Telegram not installed. Alerts disabled.")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("StrikeMaster-V12")

# API Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Instrument Symbols
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'expiry_day': 1,  # Tuesday
        'strike_gap': 50
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'expiry_day': 2,  # Wednesday
        'strike_gap': 100
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'expiry_day': 1,  # Tuesday
        'strike_gap': 50
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'expiry_day': 0,  # Monday
        'strike_gap': 25
    }
}

# Which indices to trade (comma-separated)
ACTIVE_INDICES = os.getenv('ACTIVE_INDICES', 'NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY').split(',')

# Trading Configuration
ALERT_ONLY_MODE = True  # Week 1: Only alerts
SCAN_INTERVAL = 60  # seconds

# Strategy Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0

VOL_SPIKE_2X = 2.0
VOL_SPIKE_3X = 3.0

PCR_BULLISH = 1.08
PCR_BEARISH = 0.92

MIN_CANDLE_SIZE = 8
VWAP_BUFFER = 5

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 45))
AVOID_CLOSING = (time(15, 15), time(15, 30))

# ATR Configuration
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.5

@dataclass
class Signal:
    """Trading Signal"""
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

# ==================== UTILITIES ====================
def get_current_futures_symbol(index_name: str) -> str:
    """Auto-detect current futures symbol"""
    now = datetime.now(IST)
    year = now.year
    month = now.month
    
    config = INDICES[index_name]
    expiry_day_of_week = config['expiry_day']
    
    # Find last occurrence of expiry day in current month
    last_day = monthrange(year, month)[1]
    last_date = datetime(year, month, last_day, tzinfo=IST)
    days_back = (last_date.weekday() - expiry_day_of_week) % 7
    expiry_date = last_date - timedelta(days=days_back)
    
    # If past expiry, use next month
    if now.date() > expiry_date.date() or (
        now.date() == expiry_date.date() and now.time() > time(15, 30)
    ):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    
    year_short = year % 100
    month_name = datetime(year, month, 1).strftime('%b').upper()
    
    # Futures symbol prefix
    prefix_map = {
        'NIFTY': 'NIFTY',
        'BANKNIFTY': 'BANKNIFTY',
        'FINNIFTY': 'FINNIFTY',
        'MIDCPNIFTY': 'MIDCPNIFTY'
    }
    prefix = prefix_map.get(index_name, 'NIFTY')
    
    symbol = f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"
    
    logger.info(f"🤖 {config['name']}: {symbol}")
    return symbol

def get_expiry_date(index_name: str) -> str:
    """Get next expiry date"""
    now = datetime.now(IST)
    today = now.date()
    
    expiry_day = INDICES[index_name]['expiry_day']
    
    # Days until next expiry_day
    days_to_expiry = (expiry_day - today.weekday() + 7) % 7
    
    # If today is expiry day after 3:30 PM, use next week
    if days_to_expiry == 0 and now.time() > time(15, 30):
        expiry = today + timedelta(days=7)
    else:
        expiry = today + timedelta(days=days_to_expiry)
    
    # Check if last occurrence in month
    while True:
        next_week = expiry + timedelta(days=7)
        if next_week.month != expiry.month:
            break
        expiry = next_week
    
    return expiry.strftime('%Y-%m-%d')

def is_tradeable_time() -> bool:
    """Check if tradeable time"""
    now = datetime.now(IST).time()
    
    if not (time(9, 15) <= now <= time(15, 30)):
        return False
    
    if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
        logger.info("⏰ Opening hour - Skipping")
        return False
    
    if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
        logger.info("⏰ Closing minutes - Skipping")
        return False
    
    return True

# ==================== REDIS BRAIN ====================
class RedisBrain:
    """Memory system for OI tracking"""
    
    def __init__(self):
        self.client = None
        self.memory = {}
        
        if REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                logger.info("✅ Redis Connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis Failed: Using RAM Mode")
                self.client = None
        else:
            logger.info("📦 RAM-only mode")
    
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save strike-level OI data"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.client:
                try:
                    self.client.setex(key, 3600, value)
                except:
                    self.memory[key] = value
            else:
                self.memory[key] = value
    
    def get_strike_oi_change(self, index_name: str, strike: int, current_data: dict, 
                             minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change for specific strike"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
        
        past_data_str = None
        if self.client:
            try:
                past_data_str = self.client.get(key)
            except:
                past_data_str = self.memory.get(key)
        else:
            past_data_str = self.memory.get(key)
        
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
                self.client.setex(key, 3600, data)
            except:
                self.memory[key] = data
        else:
            self.memory[key] = data
    
    def get_total_oi_change(self, index_name: str, current_ce: int, current_pe: int, 
                           minutes_ago: int = 15) -> Tuple[float, float]:
        """Get total OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        slot = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{slot.strftime('%H%M')}"
        
        past_data = None
        if self.client:
            try:
                past_data = self.client.get(key)
            except:
                past_data = self.memory.get(key)
        else:
            past_data = self.memory.get(key)
        
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
    """Fetch market data with 5-strike focus"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.retry_count = 3
        self.base_retry_delay = 2
        self.futures_symbol = get_current_futures_symbol(index_name)
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        """Smart retry with exponential backoff"""
        for attempt in range(self.retry_count):
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait_time = (2 ** attempt) * self.base_retry_delay
                        logger.warning(f"⏳ Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ HTTP {resp.status}")
                        await asyncio.sleep(self.base_retry_delay)
            except Exception as e:
                logger.error(f"💥 Attempt {attempt+1}: {e}")
                await asyncio.sleep(self.base_retry_delay * (attempt + 1))
        
        return None
    
    async def get_market_data(self) -> Tuple[pd.DataFrame, Dict[int, dict], 
                                            str, float, float, float]:
        """Fetch all market data"""
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            df = pd.DataFrame()
            strike_data = {}
            total_options_volume = 0
            
            # 1. GET SPOT PRICE
            logger.info(f"🔍 {self.index_config['name']}: Fetching Spot...")
            enc_spot = urllib.parse.quote(self.index_config['spot'])
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_spot}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            if ltp_data and 'data' in ltp_data:
                for key in [self.index_config['spot']]:
                    if key in ltp_data['data']:
                        spot_price = ltp_data['data'][key].get('last_price', 0)
                        if spot_price > 0:
                            logger.info(f"✅ Spot: {spot_price:.2f}")
                            break
            
            if spot_price == 0:
                logger.error("❌ Failed to fetch spot price")
                return df, strike_data, "", 0, 0, 0
            
            # 2. GET FUTURES CANDLES
            logger.info(f"🔍 Fetching Futures: {self.futures_symbol}")
            enc_futures = urllib.parse.quote(self.futures_symbol)
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
            candle_url = f"https://api.upstox.com/v2/historical-candle/{enc_futures}/1minute/{to_date}/{from_date}"
            
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
                    
                    today = datetime.now(IST).date()
                    df = df[df.index.date == today].tail(100)
                    
                    if not df.empty:
                        futures_price = df['close'].iloc[-1]
                        logger.info(f"✅ Futures: {len(df)} candles | Price: {futures_price:.2f}")
            
            # 3. GET OPTION CHAIN
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
                        
                        ce_oi = call_data.get('oi', 0)
                        pe_oi = put_data.get('oi', 0)
                        ce_vol = call_data.get('volume', 0)
                        pe_vol = put_data.get('volume', 0)
                        
                        strike_data[strike] = {
                            'ce_oi': ce_oi,
                            'pe_oi': pe_oi,
                            'ce_vol': ce_vol,
                            'pe_vol': pe_vol,
                            'ce_ltp': call_data.get('ltp', 0),
                            'pe_ltp': put_data.get('ltp', 0)
                        }
                        
                        total_options_volume += (ce_vol + pe_vol)
                
                logger.info(f"✅ Collected {len(strike_data)} strikes")
            
            return df, strike_data, expiry, spot_price, futures_price, total_options_volume

# ==================== ANALYZER ====================
class StrikeAnalyzer:
    """Multi-factor analysis engine"""
    
    def __init__(self):
        self.volume_history = {}
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP from futures"""
        if df.empty:
            return 0
        
        df_copy = df.copy()
        df_copy['tp'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        df_copy['vol_price'] = df_copy['tp'] * df_copy['vol']
        
        total_vol = df_copy['vol'].sum()
        if total_vol == 0:
            return df_copy['close'].iloc[-1]
        
        vwap = df_copy['vol_price'].cumsum() / df_copy['vol'].cumsum()
        return vwap.iloc[-1]
    
    def calculate_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """Average True Range"""
        if len(df) < period:
            return 30
        
        df_copy = df.tail(period).copy()
        
        df_copy['h-l'] = df_copy['high'] - df_copy['low']
        df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        
        df_copy['tr'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return df_copy['tr'].mean()
    
    def get_candle_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Current candle"""
        if df.empty:
            return 'NEUTRAL', 0
        
        last = df.iloc[-1]
        candle_size = abs(last['close'] - last['open'])
        
        if last['close'] > last['open']:
            color = 'GREEN'
        elif last['close'] < last['open']:
            color = 'RED'
        else:
            color = 'DOJI'
        
        return color, candle_size
    
    def check_volume_surge(self, index_name: str, current_vol: float) -> Tuple[bool, float]:
        """Detect volume spikes"""
        now = datetime.now(IST)
        
        if index_name not in self.volume_history:
            self.volume_history[index_name] = []
        
        self.volume_history[index_name].append({'time': now, 'volume': current_vol})
        
        cutoff = now - timedelta(minutes=20)
        self.volume_history[index_name] = [
            x for x in self.volume_history[index_name] if x['time'] > cutoff
        ]
        
        if len(self.volume_history[index_name]) < 5:
            return False, 0
        
        past_volumes = [x['volume'] for x in self.volume_history[index_name][:-1]]
        avg_vol = sum(past_volumes) / len(past_volumes)
        
        if avg_vol == 0:
            return False, 0
        
        multiplier = current_vol / avg_vol
        return multiplier >= VOL_SPIKE_2X, multiplier
    
    def calculate_focused_pcr(self, strike_data: Dict[int, dict]) -> float:
        """PCR from 5 strikes"""
        total_ce = sum(data['ce_oi'] for data in strike_data.values())
        total_pe = sum(data['pe_oi'] for data in strike_data.values())
        
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def analyze_atm_battle(self, index_name: str, strike_data: Dict[int, dict], 
                          atm_strike: int, redis_brain: RedisBrain) -> Tuple[float, float]:
        """ATM Strike Battle Analysis"""
        if atm_strike not in strike_data:
            return 0, 0
        
        current = strike_data[atm_strike]
        
        ce_15m, pe_15m = redis_brain.get_strike_oi_change(
            index_name, atm_strike, current, minutes_ago=15
        )
        
        ce_5m, pe_5m = redis_brain.get_strike_oi_change(
            index_name, atm_strike, current, minutes_ago=5
        )
        
        logger.info(f"⚔️ ATM {atm_strike}: 15m CE={ce_15m:+.1f}% PE={pe_15m:+.1f}%")
        
        return ce_15m, pe_15m
    
    def check_momentum(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        """Check momentum"""
        if df.empty or len(df) < 3:
            return False
        
        last_3 = df.tail(3)
        
        if direction == 'bullish':
            return sum(last_3['close'] > last_3['open']) >= 2
        else:
            return sum(last_3['close'] < last_3['open']) >= 2

# ==================== MAIN BOT ====================
class StrikeMasterBot:
    """Main trading bot"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.feed = StrikeDataFeed(index_name)
        self.redis = RedisBrain()
        self.analyzer = StrikeAnalyzer()
        self.telegram = None
        self.last_alert_time = None
        self.alert_cooldown = 300
        
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.telegram = Bot(token=TELEGRAM_BOT_TOKEN)
            except Exception as e:
                logger.warning(f"⚠️ Telegram: {e}")
    
    async def run_cycle(self):
        """Single analysis cycle"""
        
        if not is_tradeable_time():
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 {self.index_config['name']} SCAN")
        logger.info(f"{'='*80}")
        
        df, strike_data, expiry, spot, futures, vol = await self.feed.get_market_data()
        
        if df.empty or not strike_data or spot == 0:
            logger.warning("⏳ Incomplete data")
            return
        
        vwap = self.analyzer.calculate_vwap(df)
        atr = self.analyzer.calculate_atr(df)
        pcr = self.analyzer.calculate_focused_pcr(strike_data)
        candle_color, candle_size = self.analyzer.get_candle_info(df)
        has_vol_spike, vol_mult = self.analyzer.check_volume_surge(self.index_name, vol)
        vwap_distance = abs(futures - vwap)
        
        strike_gap = self.index_config['strike_gap']
        atm_strike = round(spot / strike_gap) * strike_gap
        atm_ce_15m, atm_pe_15m = self.analyzer.analyze_atm_battle(
            self.index_name, strike_data, atm_strike, self.redis
        )
        
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        
        ce_total_15m, pe_total_15m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=15
        )
        ce_total_5m, pe_total_5m = self.redis.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=5
        )
        
        self.redis.save_strike_snapshot(self.index_name, strike_data)
        self.redis.save_total_oi_snapshot(self.index_name, total_ce, total_pe)
        
        logger.info(f"💰 Spot: {spot:.2f} | Futures: {futures:.2f}")
        logger.info(f"📊 VWAP: {vwap:.2f} | PCR: {pcr:.2f} | Candle: {candle_color}")
        logger.info(f"📉 OI 15m: CE={ce_total_15m:+.1f}% | PE={pe_total_15m:+.1f}%")
        
        signal = self.generate_signal(
            spot, futures, vwap, vwap_distance, pcr, atr,
            ce_total_15m, pe_total_15m, ce_total_5m, pe_total_5m,
            atm_ce_15m, atm_pe_15m,
            candle_color, candle_size,
            has_vol_spike, vol_mult, df
        )
        
        if signal:
            await self.send_alert(signal)
        else:
            logger.info("✋ No setup")
        
        logger.info(f"{'='*80}\n")
    
    def generate_signal(self, spot_price, futures_price, vwap, vwap_distance, pcr, atr,
                       ce_total_15m, pe_total_15m, ce_total_5m, pe_total_5m,
                       atm_ce_change, atm_pe_change, candle_color, candle_size,
                       has_vol_spike, vol_mult, df) -> Optional[Signal]:
        """Generate trading signals"""
        
        strike_gap = self.index_config['strike_gap']
        strike = round(spot_price / strike_gap) * strike_gap
        
        stop_loss_points = int(atr * ATR_SL_MULTIPLIER)
        target_points = int(atr * ATR_TARGET_MULTIPLIER)
        
        if abs(ce_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_ce_change) >= OI_THRESHOLD_STRONG:
            target_points = max(target_points, 80)
        elif abs(ce_total_15m) >= OI_THRESHOLD_MEDIUM or abs(atm_ce_change) >= OI_THRESHOLD_MEDIUM:
            target_points = max(target_points, 50)
        
        # CE BUY SIGNAL
        if ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD:
            checks = {
                "CE OI Unwinding": ce_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM CE Unwinding": atm_ce_change < -ATM_OI_THRESHOLD,
                "Price > VWAP": futures_price > vwap,
                "GREEN Candle": candle_color == 'GREEN'
            }
            
            bonus = {
                "Strong 5m": ce_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far VWAP": vwap_distance >= VWAP_BUFFER,
                "Bullish PCR": pcr > PCR_BULLISH,
                "Vol Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bullish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 CE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    type="CE_BUY",
                    reason=f"Call Unwinding (ATM: {atm_ce_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_5m=ce_total_5m,
                    oi_15m=ce_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST),
                    index_name=self.index_name
                )
        
        # PE BUY SIGNAL
        if pe_total_15m < -OI_THRESHOLD_MEDIUM or atm_pe_change < -ATM_OI_THRESHOLD:
            if abs(pe_total_15m) >= OI_THRESHOLD_STRONG or abs(atm_pe_change) >= OI_THRESHOLD_STRONG:
                target_points = max(target_points, 80)
            
            checks = {
                "PE OI Unwinding": pe_total_15m < -OI_THRESHOLD_MEDIUM,
                "ATM PE Unwinding": atm_pe_change < -ATM_OI_THRESHOLD,
                "Price < VWAP": futures_price < vwap,
                "RED Candle": candle_color == 'RED'
            }
            
            bonus = {
                "Strong 5m": pe_total_5m < -5.0,
                "Big Candle": candle_size >= MIN_CANDLE_SIZE,
                "Far VWAP": vwap_distance >= VWAP_BUFFER,
                "Bearish PCR": pcr < PCR_BEARISH,
                "Vol Spike": has_vol_spike,
                "Momentum": self.analyzer.check_momentum(df, 'bearish')
            }
            
            passed = sum(checks.values())
            bonus_passed = sum(bonus.values())
            
            if passed == 4:
                confidence = 75 + (bonus_passed * 3)
                logger.info(f"🎯 PE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    type="PE_BUY",
                    reason=f"Put Unwinding (ATM: {atm_pe_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    futures_price=futures_price,
                    strike=strike,
                    target_points=target_points,
                    stop_loss_points=stop_loss_points,
                    pcr=pcr,
                    candle_color=candle_color,
                    volume_surge=vol_mult,
                    oi_5m=pe_total_5m,
                    oi_15m=pe_total_15m,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    atr=atr,
                    timestamp=datetime.now(IST),
                    index_name=self.index_name
                )
        
        return None
    
    async def send_alert(self, s: Signal):
        """Send Telegram alert"""
        if self.last_alert_time:
            elapsed = (datetime.now(IST) - self.last_alert_time).seconds
            if elapsed < self.alert_cooldown:
                logger.info(f"⏳ Cooldown: {self.alert_cooldown - elapsed}s")
                return
        
        self.last_alert_time = datetime.now(IST)
        
        emoji = "🟢" if s.type == "CE_BUY" else "🔴"
        
        if s.type == "CE_BUY":
            entry = s.spot_price
            target = entry + s.target_points
            stop_loss = entry - s.stop_loss_points
        else:
            entry = s.spot_price
            target = entry - s.target_points
            stop_loss = entry + s.stop_loss_points
        
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE"
        timestamp_str = s.timestamp.strftime('%d-%b %I:%M %p')
        
        msg = f"""
{emoji} {self.index_config['name']} V12.0

{mode}

━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL: {s.type}
━━━━━━━━━━━━━━━━━━━━━━━━

📍 Entry: {entry:.1f}
🎯 Target: {target:.1f} ({s.target_points:+.0f} pts)
🛑 Stop: {stop_loss:.1f} ({s.stop_loss_points:.0f} pts)
📊 Strike: {s.strike}

━━━━━━━━━━━━━━━━━━━━━━━━
LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━

{s.reason}
Confidence: {s.confidence}%

━━━━━━━━━━━━━━━━━━━━━━━━
DATA
━━━━━━━━━━━━━━━━━━━━━━━━

💰 Spot: {s.spot_price:.1f}
📈 Futures: {s.futures_price:.1f}
📊 PCR: {s.pcr:.2f}
🕯️ Candle: {s.candle_color}
🔥 Volume: {s.volume_surge:.1f}x
📏 ATR: {s.atr:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━
OI ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━

ATM Strike:
  CE: {s.atm_ce_change:+.1f}%
  PE: {s.atm_pe_change:+.1f}%

Total OI:
  5min: {s.oi_5m:+.1f}%
  15min: {s.oi_15m:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {timestamp_str}

✅ 5-Strike Focus
✅ 80%+ Target
"""
        
        logger.info(f"🚨 {s.type} @ {entry:.1f} → {target:.1f}")
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Alert sent")
            except Exception as e:
                logger.error(f"❌ Telegram: {e}")
    
    async def send_startup_message(self):
        """Send startup notification"""
        now = datetime.now(IST)
        startup_time = now.strftime('%d-%b %I:%M %p')
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE TRADING"
        
        expiry_map = {
            'NIFTY': 'Tuesday',
            'BANKNIFTY': 'Wednesday',
            'FINNIFTY': 'Tuesday',
            'MIDCPNIFTY': 'Monday'
        }
        
        msg = f"""
🚀 STRIKE MASTER V12.0

━━━━━━━━━━━━━━━━━━━━━━━━
STATUS
━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {startup_time}
📊 {self.index_config['name']}
🔄 {mode}
⏱️ Scan: {SCAN_INTERVAL}s

━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG
━━━━━━━━━━━━━━━━━━━━━━━━

📈 Futures: {self.feed.futures_symbol}
🎯 Strikes: 5 (ATM ± 2)
📅 Expiry: {expiry_map.get(self.index_name, 'N/A')}
⏰ Time Filters: Active
📏 Stops: ATR-based

━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━

✅ ATM Battle Analysis
✅ Multi-Factor Scoring
✅ Volume Surge Detection
✅ VWAP Confirmation
✅ PCR Trend Analysis

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Target: 80%+ Accuracy
⚡ Ready!
"""
        
        if self.telegram:
            try:
                await self.telegram.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                logger.info("✅ Startup sent")
            except Exception as e:
                logger.error(f"❌ Startup: {e}")

# ==================== MAIN ====================
async def main():
    """Main entry point with multi-index support"""
    
    # Parse active indices
    active_indices = [idx.strip().upper() for idx in ACTIVE_INDICES if idx.strip()]
    
    # Validate
    invalid = [idx for idx in active_indices if idx not in INDICES]
    if invalid:
        logger.error(f"❌ Invalid: {invalid}")
        logger.info(f"   Available: {list(INDICES.keys())}")
        return
    
    if not active_indices:
        logger.error("❌ No indices configured!")
        return
    
    # Create bots
    bots = {}
    for index_name in active_indices:
        try:
            bot = StrikeMasterBot(index_name)
            bots[index_name] = bot
            logger.info(f"✅ {INDICES[index_name]['name']}")
        except Exception as e:
            logger.error(f"❌ {index_name}: {e}")
    
    if not bots:
        logger.error("❌ No bots initialized!")
        return
    
    logger.info("=" * 80)
    logger.info(f"🚀 STRIKE MASTER V12.0 - MULTI-INDEX")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"📊 ACTIVE INDICES ({len(bots)}):")
    for idx, bot in bots.items():
        logger.info(f"   • {INDICES[idx]['name']}")
        logger.info(f"     {bot.feed.futures_symbol}")
    logger.info("")
    logger.info(f"🔔 Mode: {'ALERT ONLY' if ALERT_ONLY_MODE else 'LIVE TRADING'}")
    logger.info(f"⏱️ Scan: Every {SCAN_INTERVAL}s")
    logger.info("")
    logger.info("🔥 FEATURES:")
    logger.info("   ✅ Parallel Analysis")
    logger.info("   ✅ 5-Strike Focus")
    logger.info("   ✅ ATM Battle Tracking")
    logger.info("   ✅ Dynamic Stops (ATR)")
    logger.info("   ✅ Time Filters")
    logger.info("")
    logger.info("=" * 80)
    
    # Send startup messages
    for bot in bots.values():
        try:
            await bot.send_startup_message()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Startup failed: {e}")
    
    # Main loop
    iteration = 0
    
    while True:
        try:
            now = datetime.now(IST).time()
            
            if time(9, 15) <= now <= time(15, 30):
                iteration += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 CYCLE #{iteration}")
                logger.info(f"{'='*80}")
                
                # Run all bots in parallel
                tasks = [bot.run_cycle() for bot in bots.values()]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("🌙 Market closed")
                await asyncio.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
            break
        
        except Exception as e:
            logger.error(f"💥 Error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
