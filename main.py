#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V13.3 - COMPLETE PRODUCTION VERSION
======================================================
✅ All 4 Indices: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
✅ Real-time Option Chain Analysis
✅ OI Change Tracking (5min & 15min)
✅ PCR Analysis
✅ Volume Surge Detection
✅ Telegram Alerts
✅ Memory System (RAM/Redis)

Author: Complete Production Version
Date: November 24, 2025
"""

import os
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta, time
import pytz
from calendar import monthrange
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# Optional imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available, using RAM mode")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Telegram not available, alerts disabled")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("MultiIndexBot-V13.3")

# Environment Variables
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# ✅ ALL 4 INDICES CONFIGURATION
INDICES = {
    'NIFTY': {
        'spot': "NSE_INDEX|Nifty 50",
        'name': 'NIFTY 50',
        'strike_gap': 50,
        'has_weekly': True,
        'expiry_day': 1,
        'futures_prefix': 'NIFTY'
    },
    'BANKNIFTY': {
        'spot': "NSE_INDEX|Nifty Bank",
        'name': 'BANK NIFTY',
        'strike_gap': 100,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'BANKNIFTY'
    },
    'FINNIFTY': {
        'spot': "NSE_INDEX|Nifty Fin Service",
        'name': 'FIN NIFTY',
        'strike_gap': 50,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'FINNIFTY'
    },
    'MIDCPNIFTY': {
        'spot': "NSE_INDEX|NIFTY MID SELECT",
        'name': 'MIDCAP NIFTY',
        'strike_gap': 25,
        'has_weekly': False,
        'expiry_day': 1,
        'futures_prefix': 'MIDCPNIFTY'
    }
}

# 🔥 ALL INDICES ACTIVE
ACTIVE_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# Trading Configuration
ALERT_ONLY_MODE = True
SCAN_INTERVAL = 60  # seconds between full scans
INDEX_SCAN_DELAY = 2  # delay between indices

# Strategy Thresholds
OI_THRESHOLD_STRONG = 8.0
OI_THRESHOLD_MEDIUM = 5.0
ATM_OI_THRESHOLD = 5.0

VOL_SPIKE_2X = 2.0
VOL_SPIKE_3X = 3.0

PCR_BULLISH = 1.08
PCR_BEARISH = 0.92

# Time Filters
AVOID_OPENING = (time(9, 15), time(9, 45))
AVOID_CLOSING = (time(15, 15), time(15, 30))

# ==================== DATA STRUCTURES ====================
@dataclass
class Signal:
    """Trading Signal"""
    index_name: str
    type: str  # CE_BUY or PE_BUY
    reason: str
    confidence: int
    spot_price: float
    strike: int
    target_points: int
    stop_loss_points: int
    pcr: float
    atm_ce_change: float
    atm_pe_change: float
    ce_total_15m: float
    pe_total_15m: float
    ce_total_5m: float
    pe_total_5m: float
    volume_surge: float
    timestamp: datetime

# ==================== MEMORY SYSTEM ====================
class MemorySystem:
    """Unified memory system (Redis or RAM)"""
    
    def __init__(self):
        self.use_redis = False
        self.redis_client = None
        self.ram_storage = {
            'strike_snapshots': {},  # {index: {strike: {timestamp: data}}}
            'total_oi': {},  # {index: {timestamp: {ce, pe}}}
            'volume_history': {}  # {index: [{time, volume}]}
        }
        
        # Try Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                self.redis_client.ping()
                self.use_redis = True
                logger.info("✅ Redis Connected")
            except:
                logger.info("📦 Using RAM mode")
        else:
            logger.info("📦 RAM-only mode")
    
    def _clean_old_data(self, data_dict: dict, minutes: int = 30):
        """Remove data older than specified minutes"""
        cutoff = datetime.now(IST) - timedelta(minutes=minutes)
        return {
            ts: d for ts, d in data_dict.items()
            if isinstance(ts, datetime) and ts > cutoff
        }
    
    # ===== STRIKE SNAPSHOTS =====
    def save_strike_snapshot(self, index_name: str, strike_data: Dict[int, dict]):
        """Save strike-level OI data"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        
        for strike, data in strike_data.items():
            key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
            value = json.dumps(data)
            
            if self.use_redis:
                try:
                    self.redis_client.setex(key, 3600, value)
                except:
                    # Fallback to RAM
                    if index_name not in self.ram_storage['strike_snapshots']:
                        self.ram_storage['strike_snapshots'][index_name] = {}
                    if strike not in self.ram_storage['strike_snapshots'][index_name]:
                        self.ram_storage['strike_snapshots'][index_name][strike] = {}
                    self.ram_storage['strike_snapshots'][index_name][strike][timestamp] = data
            else:
                # RAM mode
                if index_name not in self.ram_storage['strike_snapshots']:
                    self.ram_storage['strike_snapshots'][index_name] = {}
                if strike not in self.ram_storage['strike_snapshots'][index_name]:
                    self.ram_storage['strike_snapshots'][index_name][strike] = {}
                self.ram_storage['strike_snapshots'][index_name][strike][timestamp] = data
        
        # Clean old data in RAM mode
        if not self.use_redis and index_name in self.ram_storage['strike_snapshots']:
            for strike in self.ram_storage['strike_snapshots'][index_name]:
                self.ram_storage['strike_snapshots'][index_name][strike] = \
                    self._clean_old_data(self.ram_storage['strike_snapshots'][index_name][strike])
    
    def get_strike_oi_change(self, index_name: str, strike: int, 
                            current_data: dict, minutes_ago: int = 15) -> Tuple[float, float]:
        """Calculate OI change for specific strike"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:strike:{strike}:{timestamp.strftime('%H%M')}"
        
        past_data_str = None
        
        if self.use_redis:
            try:
                past_data_str = self.redis_client.get(key)
            except:
                pass
        
        if not past_data_str and index_name in self.ram_storage['strike_snapshots']:
            if strike in self.ram_storage['strike_snapshots'][index_name]:
                snapshots = self.ram_storage['strike_snapshots'][index_name][strike]
                past_times = [t for t in snapshots.keys() if t <= timestamp]
                if past_times:
                    closest = max(past_times)
                    past_data_str = json.dumps(snapshots[closest])
        
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
    
    # ===== TOTAL OI =====
    def save_total_oi(self, index_name: str, ce_total: int, pe_total: int):
        """Save total OI snapshot"""
        now = datetime.now(IST)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{timestamp.strftime('%H%M')}"
        data = json.dumps({"ce": ce_total, "pe": pe_total})
        
        if self.use_redis:
            try:
                self.redis_client.setex(key, 3600, data)
            except:
                if index_name not in self.ram_storage['total_oi']:
                    self.ram_storage['total_oi'][index_name] = {}
                self.ram_storage['total_oi'][index_name][timestamp] = {"ce": ce_total, "pe": pe_total}
        else:
            if index_name not in self.ram_storage['total_oi']:
                self.ram_storage['total_oi'][index_name] = {}
            self.ram_storage['total_oi'][index_name][timestamp] = {"ce": ce_total, "pe": pe_total}
        
        # Clean old data
        if not self.use_redis and index_name in self.ram_storage['total_oi']:
            self.ram_storage['total_oi'][index_name] = \
                self._clean_old_data(self.ram_storage['total_oi'][index_name])
    
    def get_total_oi_change(self, index_name: str, current_ce: int, 
                           current_pe: int, minutes_ago: int = 15) -> Tuple[float, float]:
        """Get total OI change"""
        now = datetime.now(IST) - timedelta(minutes=minutes_ago)
        timestamp = now.replace(second=0, microsecond=0)
        key = f"{index_name}:total_oi:{timestamp.strftime('%H%M')}"
        
        past_data = None
        
        if self.use_redis:
            try:
                past_data_str = self.redis_client.get(key)
                if past_data_str:
                    past_data = json.loads(past_data_str)
            except:
                pass
        
        if not past_data and index_name in self.ram_storage['total_oi']:
            snapshots = self.ram_storage['total_oi'][index_name]
            past_times = [t for t in snapshots.keys() if t <= timestamp]
            if past_times:
                closest = max(past_times)
                past_data = snapshots[closest]
        
        if not past_data:
            return 0.0, 0.0
        
        try:
            ce_chg = ((current_ce - past_data['ce']) / past_data['ce'] * 100
                      if past_data['ce'] > 0 else 0)
            pe_chg = ((current_pe - past_data['pe']) / past_data['pe'] * 100
                      if past_data['pe'] > 0 else 0)
            return ce_chg, pe_chg
        except:
            return 0.0, 0.0
    
    # ===== VOLUME TRACKING =====
    def track_volume(self, index_name: str, volume: float) -> Tuple[bool, float]:
        """Track volume and detect spikes"""
        now = datetime.now(IST)
        
        if index_name not in self.ram_storage['volume_history']:
            self.ram_storage['volume_history'][index_name] = []
        
        self.ram_storage['volume_history'][index_name].append({
            'time': now,
            'volume': volume
        })
        
        # Clean old data (keep last 20 minutes)
        cutoff = now - timedelta(minutes=20)
        self.ram_storage['volume_history'][index_name] = [
            v for v in self.ram_storage['volume_history'][index_name]
            if v['time'] > cutoff
        ]
        
        history = self.ram_storage['volume_history'][index_name]
        
        if len(history) < 5:
            return False, 0.0
        
        past_volumes = [v['volume'] for v in history[:-1]]
        avg_vol = sum(past_volumes) / len(past_volumes)
        
        if avg_vol == 0:
            return False, 0.0
        
        multiplier = volume / avg_vol
        has_spike = multiplier >= VOL_SPIKE_2X
        
        return has_spike, multiplier

# ==================== DATA FEED ====================
class DataFeed:
    """Fetch market data from Upstox"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.config = INDICES[index_name]
        self.spot_symbol = self.config['spot']
        self.strike_gap = self.config['strike_gap']
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def encode_symbol(self, symbol: str) -> str:
        """Proper URL encoding"""
        return symbol.replace('|', '%7C').replace(' ', '%20')
    
    def get_expiry_date(self) -> str:
        """Calculate expiry date"""
        now = datetime.now(IST)
        today = now.date()
        
        if self.config['has_weekly']:
            # NIFTY: Next Tuesday
            days_to_tuesday = (1 - today.weekday() + 7) % 7
            if days_to_tuesday == 0:
                expiry = today if now.time() <= time(15, 30) else today + timedelta(days=7)
            else:
                expiry = today + timedelta(days=days_to_tuesday)
        else:
            # Monthly: Last Tuesday
            year, month = now.year, now.month
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            days_to_tuesday = (last_date.weekday() - 1) % 7
            last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            if now.date() > last_tuesday.date() or (
                now.date() == last_tuesday.date() and now.time() > time(15, 30)
            ):
                month += 1
                if month > 12:
                    year += 1
                    month = 1
                
                last_day = monthrange(year, month)[1]
                last_date = datetime(year, month, last_day)
                days_to_tuesday = (last_date.weekday() - 1) % 7
                last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            expiry = last_tuesday.date()
        
        return expiry.strftime('%Y-%m-%d')
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession, retries: int = 3):
        """Fetch with retry"""
        for attempt in range(retries):
            try:
                async with session.get(url, headers=self.headers, 
                                     timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"⏳ [{self.index_name}] Rate limit, wait {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"❌ [{self.index_name}] HTTP {resp.status}")
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"💥 [{self.index_name}] Attempt {attempt+1}: {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        return None
    
    async def get_market_data(self) -> Tuple[float, Dict[int, dict], str, float]:
        """Fetch spot price and option chain"""
        async with aiohttp.ClientSession() as session:
            # Get spot price
            spot_encoded = self.encode_symbol(self.spot_symbol)
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={spot_encoded}"
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            
            spot_price = 0
            if ltp_data and 'data' in ltp_data:
                for key in [self.spot_symbol, 
                           self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:')]:
                    if key in ltp_data['data']:
                        spot_price = ltp_data['data'][key].get('last_price', 0)
                        if spot_price > 0:
                            break
            
            if spot_price == 0:
                logger.error(f"❌ [{self.index_name}] Failed to get spot")
                return 0, {}, "", 0
            
            # Get option chain
            expiry = self.get_expiry_date()
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={spot_encoded}&expiry_date={expiry}"
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            
            strike_data = {}
            total_volume = 0
            
            if chain_data and chain_data.get('status') == 'success':
                atm_strike = round(spot_price / self.strike_gap) * self.strike_gap
                min_strike = atm_strike - (2 * self.strike_gap)
                max_strike = atm_strike + (2 * self.strike_gap)
                
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
                        
                        total_volume += (call_data.get('volume', 0) + put_data.get('volume', 0))
            
            return spot_price, strike_data, expiry, total_volume

# ==================== ANALYZER ====================
class Analyzer:
    """Signal generation with full logic"""
    
    def __init__(self, index_name: str, memory: MemorySystem):
        self.index_name = index_name
        self.config = INDICES[index_name]
        self.memory = memory
    
    def calculate_pcr(self, strike_data: Dict[int, dict]) -> float:
        """Calculate PCR"""
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        return total_pe / total_ce if total_ce > 0 else 1.0
    
    def analyze(self, spot_price: float, strike_data: Dict[int, dict], 
                total_volume: float) -> Optional[Signal]:
        """Generate signal with complete logic"""
        if not strike_data or spot_price == 0:
            return None
        
        # Save data
        self.memory.save_strike_snapshot(self.index_name, strike_data)
        
        total_ce = sum(d['ce_oi'] for d in strike_data.values())
        total_pe = sum(d['pe_oi'] for d in strike_data.values())
        self.memory.save_total_oi(self.index_name, total_ce, total_pe)
        
        # Calculate metrics
        pcr = self.calculate_pcr(strike_data)
        atm_strike = round(spot_price / self.config['strike_gap']) * self.config['strike_gap']
        
        # ATM analysis
        atm_ce_change = 0.0
        atm_pe_change = 0.0
        
        if atm_strike in strike_data:
            atm_data = strike_data[atm_strike]
            atm_ce_change, atm_pe_change = self.memory.get_strike_oi_change(
                self.index_name, atm_strike, atm_data, minutes_ago=15
            )
        
        # Total OI changes
        ce_total_15m, pe_total_15m = self.memory.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=15
        )
        ce_total_5m, pe_total_5m = self.memory.get_total_oi_change(
            self.index_name, total_ce, total_pe, minutes_ago=5
        )
        
        # Volume analysis
        has_vol_spike, vol_mult = self.memory.track_volume(self.index_name, total_volume)
        
        logger.info(f"📊 [{self.index_name}] Spot: {spot_price:.1f} | PCR: {pcr:.2f} | "
                   f"ATM CE: {atm_ce_change:+.1f}% PE: {atm_pe_change:+.1f}% | "
                   f"Vol: {vol_mult:.1f}x")
        
        # CE Buy Signal
        if (ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD):
            
            checks = {
                "CE Unwinding": ce_total_15m < -OI_THRESHOLD_MEDIUM or atm_ce_change < -ATM_OI_THRESHOLD,
                "Strong 5m": abs(ce_total_5m) >= 3.0,
                "Volume Spike": has_vol_spike
            }
            
            passed = sum(checks.values())
            
            if passed >= 1:  # At least CE unwinding
                confidence = 70 + (passed * 5)
                target = 60 if abs(ce_total_15m) >= OI_THRESHOLD_STRONG else 50
                
                logger.info(f"🟢 [{self.index_name}] CE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    index_name=self.index_name,
                    type="CE_BUY",
                    reason=f"Call Unwinding (ATM: {atm_ce_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    strike=atm_strike,
                    target_points=target,
                    stop_loss_points=30,
                    pcr=pcr,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    ce_total_15m=ce_total_15m,
                    pe_total_15m=pe_total_15m,
                    ce_total_5m=ce_total_5m,
                    pe_total_5m=pe_total_5m,
                    volume_surge=vol_mult,
                    timestamp=datetime.now(IST)
                )
        
        # PE Buy Signal
        if (pe_total_15m < -OI_THRESHOLD_MEDIUM or atm_pe_change < -ATM_OI_THRESHOLD):
            
            checks = {
                "PE Unwinding": pe_total_15m < -OI_THRESHOLD_MEDIUM or atm_pe_change < -ATM_OI_THRESHOLD,
                "Strong 5m": abs(pe_total_5m) >= 3.0,
                "Volume Spike": has_vol_spike
            }
            
            passed = sum(checks.values())
            
            if passed >= 1:
                confidence = 70 + (passed * 5)
                target = 60 if abs(pe_total_15m) >= OI_THRESHOLD_STRONG else 50
                
                logger.info(f"🔴 [{self.index_name}] PE SIGNAL! Conf: {confidence}%")
                
                return Signal(
                    index_name=self.index_name,
                    type="PE_BUY",
                    reason=f"Put Unwinding (ATM: {atm_pe_change:.1f}%)",
                    confidence=min(confidence, 95),
                    spot_price=spot_price,
                    strike=atm_strike,
                    target_points=target,
                    stop_loss_points=30,
                    pcr=pcr,
                    atm_ce_change=atm_ce_change,
                    atm_pe_change=atm_pe_change,
                    ce_total_15m=ce_total_15m,
                    pe_total_15m=pe_total_15m,
                    ce_total_5m=ce_total_5m,
                    pe_total_5m=pe_total_5m,
                    volume_surge=vol_mult,
                    timestamp=datetime.now(IST)
                )
        
        return None

# ==================== TELEGRAM ====================
class TelegramAlerts:
    """Telegram notification system"""
    
    def __init__(self):
        self.bot = None
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                logger.info("✅ Telegram Ready")
            except:
                logger.warning("⚠️ Telegram setup failed")
    
    async def send_alert(self, signal: Signal):
        """Send signal alert"""
        if not self.bot:
            return
        
        emoji = "🟢" if signal.type == "CE_BUY" else "🔴"
        
        entry = signal.spot_price
        if signal.type == "CE_BUY":
            target = entry + signal.target_points
            stop = entry - signal.stop_loss_points
        else:
            target = entry - signal.target_points
            stop = entry + signal.stop_loss_points
        
        mode = "🧪 ALERT ONLY" if ALERT_ONLY_MODE else "⚡ LIVE"
        
        msg = f"""
{emoji} {INDICES[signal.index_name]['name']} SIGNAL

{mode}

━━━━━━━━━━━━━━━━━━━━━━━━
TRADE DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━

Signal: {signal.type}
Entry: {entry:.1f}
Target: {target:.1f} ({signal.target_points:+} pts)
Stop Loss: {stop:.1f} ({signal.stop_loss_points} pts)
Strike: {signal.strike}

━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━

{signal.reason}
Confidence: {signal.confidence}%

PCR: {signal.pcr:.2f}
Volume: {signal.volume_surge:.1f}x

ATM Changes:
  CE: {signal.atm_ce_change:+.1f}%
  PE: {signal.atm_pe_change:+.1f}%

Total OI (15m):
  CE: {signal.ce_total_15m:+.1f}%
  PE: {signal.pe_total_15m:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {signal.timestamp.strftime('%I:%M %p')}
"""
        
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            logger.info(f"✅ [{signal.index_name}] Alert sent!")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
    
    async def send_startup(self, active_indices: List[str]):
        """Send startup notification"""
        if not self.bot:
            return
        
        now = datetime.now(IST)
        
        indices_info = []
        for idx in active_indices:
            config = INDICES[idx]
            expiry_type = "Weekly" if config['has_weekly'] else "Monthly"
            indices_info.append(
                f"📊 {config['name']}\n"
                f"   Gap: {config['strike_gap']}pts | {expiry_type}"
            )
        
        msg = f"""
🚀 MULTI-INDEX BOT V13.3

━━━━━━━━━━━━━━━━━━━━━━━━
STATUS
━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Started: {now.strftime('%I:%M %p')}
🔄 Mode: {'🧪 ALERT ONLY' if ALERT_ONLY_MODE else '⚡ LIVE'}
⏱️ Scan: Every {SCAN_INTERVAL}s
🎯 Indices: {len(active_indices)}

━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVE INDICES
━━━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(indices_info)}

━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━

✅ Real-time OI tracking
✅ ATM Battle Analysis
✅ Volume Spike Detection
✅ PCR Analysis
✅ Per-index cooldown
✅ Multi-factor scoring

━━━━━━━━━━━━━━━━━━━━━━━━

🔥 Scanning all indices now!
"""
        
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            logger.info("✅ Startup notification sent")
        except Exception as e:
            logger.error(f"❌ Startup message failed: {e}")

# ==================== MAIN BOT ====================
class MultiIndexBot:
    """Master bot controlling all indices"""
    
    def __init__(self):
        self.memory = MemorySystem()
        self.feeds = {name: DataFeed(name) for name in ACTIVE_INDICES}
        self.analyzers = {name: Analyzer(name, self.memory) for name in ACTIVE_INDICES}
        self.telegram = TelegramAlerts()
        self.last_alerts = {name: None for name in ACTIVE_INDICES}
        self.alert_cooldown = 300  # 5 minutes per index
    
    def is_tradeable_time(self) -> bool:
        """Check if it's a good time to trade"""
        now = datetime.now(IST).time()
        
        if not (time(9, 15) <= now <= time(15, 30)):
            return False
        
        if AVOID_OPENING[0] <= now <= AVOID_OPENING[1]:
            return False
        
        if AVOID_CLOSING[0] <= now <= AVOID_CLOSING[1]:
            return False
        
        return True
    
    async def run_cycle(self):
        """Run one complete scan cycle"""
        if not self.is_tradeable_time():
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 SCAN CYCLE - {datetime.now(IST).strftime('%I:%M:%S %p')}")
        logger.info(f"{'='*70}")
        
        for index_name in ACTIVE_INDICES:
            try:
                # Fetch data
                feed = self.feeds[index_name]
                spot, strikes, expiry, volume = await feed.get_market_data()
                
                if spot == 0 or not strikes:
                    logger.warning(f"⏭️ [{index_name}] No data")
                    continue
                
                # Analyze
                analyzer = self.analyzers[index_name]
                signal = analyzer.analyze(spot, strikes, volume)
                
                if signal:
                    # Check cooldown
                    last_alert = self.last_alerts[index_name]
                    if last_alert:
                        elapsed = (datetime.now(IST) - last_alert).seconds
                        if elapsed < self.alert_cooldown:
                            logger.info(f"⏳ [{index_name}] Cooldown: {self.alert_cooldown - elapsed}s")
                            continue
                    
                    # Send alert
                    logger.info(f"\n🚨 [{index_name}] SIGNAL GENERATED!")
                    logger.info(f"   Type: {signal.type}")
                    logger.info(f"   Entry: {signal.spot_price:.1f}")
                    logger.info(f"   Target: {signal.target_points:+} pts")
                    logger.info(f"   Confidence: {signal.confidence}%\n")
                    
                    await self.telegram.send_alert(signal)
                    self.last_alerts[index_name] = datetime.now(IST)
                else:
                    logger.info(f"✋ [{index_name}] No signal")
                
                # Delay between indices
                await asyncio.sleep(INDEX_SCAN_DELAY)
                
            except Exception as e:
                logger.error(f"💥 [{index_name}] Error: {e}")
                continue
        
        logger.info(f"{'='*70}\n")
    
    async def start(self):
        """Start the bot"""
        logger.info("=" * 70)
        logger.info("🚀 MULTI-INDEX BOT V13.3 STARTING")
        logger.info("=" * 70)
        logger.info("")
        logger.info("🔥 ACTIVE INDICES:")
        for idx in ACTIVE_INDICES:
            config = INDICES[idx]
            logger.info(f"   ✅ {config['name']} (Gap: {config['strike_gap']}pts)")
        logger.info("")
        logger.info(f"⏱️  Scan Interval: {SCAN_INTERVAL}s")
        logger.info(f"📊 Memory: {'Redis' if self.memory.use_redis else 'RAM'}")
        logger.info(f"📱 Telegram: {'Enabled' if self.telegram.bot else 'Disabled'}")
        logger.info("")
        logger.info("=" * 70)
        
        # Send startup notification
        await self.telegram.send_startup(ACTIVE_INDICES)
        
        # Main loop
        while True:
            try:
                now = datetime.now(IST).time()
                
                if time(9, 15) <= now <= time(15, 30):
                    await self.run_cycle()
                    await asyncio.sleep(SCAN_INTERVAL)
                else:
                    logger.info("🌙 Market closed - Waiting...")
                    await asyncio.sleep(300)
            
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
            
            except Exception as e:
                logger.error(f"💥 Critical error: {e}")
                await asyncio.sleep(30)

# ==================== MAIN ====================
async def main():
    """Main entry point"""
    bot = MultiIndexBot()
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutdown complete")
