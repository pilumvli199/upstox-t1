"""
🚀 NIFTY OPTIONS BOT - PRODUCTION READY v5.0
==============================================
Version: 5.0 (COMPLETE REWRITE - STRIKE-WISE ANALYSIS!)
Author: Built for Indian Options Trading
Last Updated: Feb 2026

✅ MAJOR IMPROVEMENTS IN v5.0:
- 🔥 STRIKE-WISE SEPARATE ANALYSIS (ATM 3x focus!)
- ✅ Support/Resistance from OI levels
- ✅ DeepSeek timeout: 10→30 seconds
- ✅ Better candlestick integration
- ✅ Realistic SL/Target setup
- ✅ Enhanced AI prompt structure
- ✅ Per-strike reasoning & confidence

⚡ NEW FEATURES:
- ATM Strike: 3x weight in analysis
- ATM ±50: 2x weight
- ATM ±100/150: 1x weight
- Support = Highest PUT OI strike
- Resistance = Highest CALL OI strike
- Strike-wise entry recommendations
- Dynamic SL/Target based on volatility

🎯 STRATEGY:
- Primary: OI Changes (15-min) with strike-wise breakdown
- Secondary: Candlestick Patterns (5-min) + S/R levels
- AI: DeepSeek V3 with 30-sec timeout
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import pytz

# ======================== CONFIGURATION ========================
# Environment Variables
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")

# Upstox API
UPSTOX_API_URL = "https://api.upstox.com/v2"

# Trading Parameters
SYMBOL = "NIFTY"
ATM_RANGE = 3  # ±3 strikes (7 total)
STRIKE_INTERVAL = 50  # NIFTY strike gap
ANALYSIS_INTERVAL = 5 * 60  # 5 minutes
CACHE_SIZE = 6  # 30 min = 6 snapshots @ 5min

# Signal Thresholds
MIN_OI_CHANGE_15MIN = 10.0  # 10% = strong signal
STRONG_OI_CHANGE = 15.0     # 15% = very strong
MIN_CONFIDENCE = 7          # Only alert if confidence >= 7

# Strike Weight Multipliers (NEW!)
ATM_WEIGHT = 3.0      # ATM strike gets 3x importance
NEAR_ATM_WEIGHT = 2.0  # ATM ±50 gets 2x importance
FAR_WEIGHT = 1.0       # ATM ±100/150 gets 1x importance

# API Settings
API_DELAY = 0.2  # 200ms between calls
MAX_RETRIES = 3
DEEPSEEK_TIMEOUT = 30  # ✅ INCREASED FROM 10 to 30 SECONDS!

# Market Hours (IST)
IST = pytz.timezone('Asia/Kolkata')
MARKET_START_HOUR = 9
MARKET_START_MIN = 15
MARKET_END_HOUR = 15
MARKET_END_MIN = 30

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================== DATA STRUCTURES ========================
@dataclass
class OISnapshot:
    """Single strike OI data at a point in time"""
    strike: int
    ce_oi: int
    pe_oi: int
    ce_ltp: float
    pe_ltp: float
    timestamp: datetime


@dataclass
class MarketSnapshot:
    """Complete market data at a point in time"""
    timestamp: datetime
    spot_price: float
    atm_strike: int
    strikes_oi: Dict[int, OISnapshot]  # strike -> OISnapshot


@dataclass
class StrikeAnalysis:
    """NEW! Detailed analysis for a single strike"""
    strike: int
    is_atm: bool
    distance_from_atm: int
    weight: float
    
    # Current OI
    ce_oi: int
    pe_oi: int
    ce_ltp: float
    pe_ltp: float
    
    # OI Changes
    ce_change_5min: float
    pe_change_5min: float
    ce_change_15min: float
    pe_change_15min: float
    ce_change_30min: float
    pe_change_30min: float
    
    # OI Ratios
    put_call_ratio: float  # PE OI / CE OI
    
    # Writer Activity
    ce_writer_action: str  # "BUILDING" / "UNWINDING" / "NEUTRAL"
    pe_writer_action: str
    
    # Support/Resistance Role
    is_support_level: bool
    is_resistance_level: bool
    
    # Signal Strength
    bullish_signal_strength: float  # 0-10
    bearish_signal_strength: float  # 0-10
    
    # Recommendation
    strike_recommendation: str  # "STRONG_CALL" / "STRONG_PUT" / "WAIT"
    confidence: float  # 0-10


@dataclass
class SupportResistance:
    """NEW! Support/Resistance levels from OI"""
    support_strike: int
    support_put_oi: int
    resistance_strike: int
    resistance_call_oi: int
    spot_near_support: bool  # within 50 points
    spot_near_resistance: bool  # within 50 points


# ======================== IN-MEMORY CACHE ========================
class SimpleCache:
    """Stores last 30 min of data (6 snapshots)"""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        self.snapshots = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add(self, snapshot: MarketSnapshot):
        """Add new snapshot"""
        async with self._lock:
            self.snapshots.append(snapshot)
            logger.info(f"📦 Cached snapshot | Total: {len(self.snapshots)}")
    
    async def get_minutes_ago(self, minutes: int) -> Optional[MarketSnapshot]:
        """Get snapshot from N minutes ago"""
        async with self._lock:
            if len(self.snapshots) < 2:
                return None
            
            target_time = datetime.now(IST) - timedelta(minutes=minutes)
            
            # Find closest match
            best = None
            min_diff = float('inf')
            
            for snap in self.snapshots:
                diff = abs((snap.timestamp - target_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best = snap
            
            # Accept if within reasonable tolerance (3 minutes)
            if best and min_diff <= 180:
                return best
            
            return None
    
    def size(self) -> int:
        return len(self.snapshots)


# ======================== UPSTOX CLIENT ========================
class UpstoxClient:
    """Upstox v2 API client"""
    
    def __init__(self, token: str):
        self.token = token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
    
    async def init(self):
        """Initialize session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Request with retry"""
        for attempt in range(MAX_RETRIES):
            try:
                async with getattr(self.session, method)(url, **kwargs) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = (attempt + 1) * 2
                        logger.warning(f"⚠️ Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        logger.warning(f"⚠️ Request failed: {resp.status} - {text[:200]}")
                        return None
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(1)
        return None
    
    async def get_option_chain(self, expiry: str) -> Optional[Dict]:
        """Get option chain for NIFTY"""
        url = f"{UPSTOX_API_URL}/option/chain"
        params = {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry_date": expiry
        }
        return await self._request('get', url, params=params)
    
    async def get_1min_candles(self) -> pd.DataFrame:
        """Get NIFTY 50 spot 1-min candles"""
        instrument_key = "NSE_INDEX|Nifty 50"
        url = f"{UPSTOX_API_URL}/historical-candle/intraday/{instrument_key}/1minute"
        
        logger.info(f"📈 Fetching NIFTY 50 spot candles...")
        data = await self._request('get', url)
        
        if not data or data.get("status") != "success":
            logger.warning("⚠️ Could not fetch candle data from API")
            return pd.DataFrame()
        
        candles = data.get("data", {}).get("candles", [])
        
        if not candles or len(candles) == 0:
            logger.warning("⚠️ Empty candle data from Upstox")
            logger.info("💡 Continuing with OI-only analysis")
            return pd.DataFrame()
        
        df_data = []
        for candle in candles:
            try:
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': int(candle[5]) if len(candle) > 5 else 0
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ Skipping malformed candle: {e}")
                continue
        
        if not df_data:
            logger.warning("⚠️ No valid candle data after parsing")
            return pd.DataFrame()
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"✅ Fetched {len(df)} 1-min NIFTY 50 spot candles")
        return df
    
    async def get_available_expiries(self) -> List[str]:
        """Get all available expiry dates from Upstox API"""
        url = f"{UPSTOX_API_URL}/option/contract"
        params = {"instrument_key": "NSE_INDEX|Nifty 50"}
        
        data = await self._request('get', url, params=params)
        
        if not data or data.get("status") != "success":
            logger.warning("⚠️ Could not fetch available expiries")
            return []
        
        contracts = data.get("data", [])
        
        if not contracts:
            logger.warning("⚠️ No option contracts available")
            return []
        
        expiries = sorted(set(item.get("expiry") for item in contracts if item.get("expiry")))
        logger.info(f"📅 Found {len(expiries)} available expiries")
        return expiries
    
    async def get_nearest_expiry(self) -> Optional[str]:
        """Get ACTUAL nearest expiry from Upstox"""
        expiries = await self.get_available_expiries()
        
        if not expiries:
            logger.error("❌ No expiries available from Upstox")
            return None
        
        now = datetime.now(IST).date()
        future_expiries = [
            exp for exp in expiries 
            if datetime.strptime(exp, '%Y-%m-%d').date() >= now
        ]
        
        if not future_expiries:
            logger.warning("⚠️ No future expiries found, using last available")
            return expiries[-1]
        
        nearest = future_expiries[0]
        logger.info(f"✅ Using nearest expiry: {nearest}")
        return nearest


# ======================== PATTERN DETECTOR ========================
class PatternDetector:
    """Enhanced candlestick pattern detection"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> List[Dict]:
        """Detect last 5 strong patterns"""
        patterns = []
        
        if df.empty or len(df) < 2:
            return patterns
        
        for i in range(len(df)):
            if i < 1:
                continue
            
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            body_curr = abs(curr['close'] - curr['open'])
            body_prev = abs(prev['close'] - prev['open'])
            range_curr = curr['high'] - curr['low']
            
            if range_curr == 0:
                continue
            
            # Bullish Engulfing
            if (curr['close'] > curr['open'] and 
                prev['close'] < prev['open'] and
                curr['open'] <= prev['close'] and
                curr['close'] >= prev['open'] and
                body_curr > body_prev * 1.2):
                patterns.append({
                    'time': curr.name,
                    'pattern': 'BULLISH_ENGULFING',
                    'type': 'BULLISH',
                    'strength': 8,
                    'price': curr['close']
                })
            
            # Bearish Engulfing
            elif (curr['close'] < curr['open'] and 
                  prev['close'] > prev['open'] and
                  curr['open'] >= prev['close'] and
                  curr['close'] <= prev['open'] and
                  body_curr > body_prev * 1.2):
                patterns.append({
                    'time': curr.name,
                    'pattern': 'BEARISH_ENGULFING',
                    'type': 'BEARISH',
                    'strength': 8,
                    'price': curr['close']
                })
            
            else:
                lower_wick = min(curr['open'], curr['close']) - curr['low']
                upper_wick = curr['high'] - max(curr['open'], curr['close'])
                
                # Hammer
                if (lower_wick > body_curr * 2 and 
                    upper_wick < body_curr * 0.3 and
                    body_curr < range_curr * 0.35):
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'HAMMER',
                        'type': 'BULLISH',
                        'strength': 6,
                        'price': curr['close']
                    })
                
                # Shooting Star
                elif (upper_wick > body_curr * 2 and 
                      lower_wick < body_curr * 0.3 and
                      body_curr < range_curr * 0.35):
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'SHOOTING_STAR',
                        'type': 'BEARISH',
                        'strength': 6,
                        'price': curr['close']
                    })
                
                # Doji
                elif body_curr < range_curr * 0.1:
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'DOJI',
                        'type': 'NEUTRAL',
                        'strength': 4,
                        'price': curr['close']
                    })
        
        return patterns[-5:] if patterns else []
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate S/R from recent price action"""
        if df.empty or len(df) < 10:
            return 0.0, 0.0
        
        last_20 = df.tail(20)
        support = last_20['low'].min()
        resistance = last_20['high'].max()
        
        return support, resistance


# ======================== ENHANCED OI ANALYZER ========================
class EnhancedOIAnalyzer:
    """NEW! Strike-wise OI analysis with ATM focus"""
    
    def __init__(self, cache: SimpleCache):
        self.cache = cache
    
    def _calculate_strike_weight(self, strike: int, atm: int) -> float:
        """Calculate weight based on distance from ATM"""
        distance = abs(strike - atm)
        
        if distance == 0:
            return ATM_WEIGHT  # 3x
        elif distance == STRIKE_INTERVAL:
            return NEAR_ATM_WEIGHT  # 2x
        else:
            return FAR_WEIGHT  # 1x
    
    def _determine_writer_action(self, oi_change: float) -> str:
        """Determine if writers are building or unwinding"""
        if oi_change >= 10:
            return "BUILDING"
        elif oi_change <= -10:
            return "UNWINDING"
        else:
            return "NEUTRAL"
    
    def _calculate_signal_strength(self, 
                                   ce_change: float, 
                                   pe_change: float,
                                   weight: float) -> Tuple[float, float]:
        """Calculate bullish/bearish signal strength"""
        
        # PUT OI increase = BULLISH (writers adding support)
        # CALL OI increase = BEARISH (writers adding resistance)
        
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        # PUT OI building = BULLISH
        if pe_change >= STRONG_OI_CHANGE:
            bullish_strength = 9.0 * weight
        elif pe_change >= MIN_OI_CHANGE_15MIN:
            bullish_strength = 7.0 * weight
        elif pe_change >= 5:
            bullish_strength = 4.0 * weight
        
        # CALL OI building = BEARISH
        if ce_change >= STRONG_OI_CHANGE:
            bearish_strength = 9.0 * weight
        elif ce_change >= MIN_OI_CHANGE_15MIN:
            bearish_strength = 7.0 * weight
        elif ce_change >= 5:
            bearish_strength = 4.0 * weight
        
        # PUT OI unwinding = BEARISH
        if pe_change <= -STRONG_OI_CHANGE:
            bearish_strength = max(bearish_strength, 8.0 * weight)
        elif pe_change <= -MIN_OI_CHANGE_15MIN:
            bearish_strength = max(bearish_strength, 6.0 * weight)
        
        # CALL OI unwinding = BULLISH
        if ce_change <= -STRONG_OI_CHANGE:
            bullish_strength = max(bullish_strength, 8.0 * weight)
        elif ce_change <= -MIN_OI_CHANGE_15MIN:
            bullish_strength = max(bullish_strength, 6.0 * weight)
        
        return bullish_strength, bearish_strength
    
    async def analyze_strike(self, 
                           strike: int,
                           current: MarketSnapshot,
                           snap_5min: Optional[MarketSnapshot],
                           snap_15min: Optional[MarketSnapshot],
                           snap_30min: Optional[MarketSnapshot]) -> StrikeAnalysis:
        """NEW! Detailed analysis for single strike"""
        
        curr_oi = current.strikes_oi.get(strike)
        if not curr_oi:
            return None
        
        # Calculate changes
        def calc_change(current, previous):
            if previous and previous > 0:
                return ((current - previous) / previous * 100)
            return 0
        
        prev_5 = snap_5min.strikes_oi.get(strike) if snap_5min else None
        prev_15 = snap_15min.strikes_oi.get(strike) if snap_15min else None
        prev_30 = snap_30min.strikes_oi.get(strike) if snap_30min else None
        
        ce_5min = calc_change(curr_oi.ce_oi, prev_5.ce_oi if prev_5 else 0)
        pe_5min = calc_change(curr_oi.pe_oi, prev_5.pe_oi if prev_5 else 0)
        ce_15min = calc_change(curr_oi.ce_oi, prev_15.ce_oi if prev_15 else 0)
        pe_15min = calc_change(curr_oi.pe_oi, prev_15.pe_oi if prev_15 else 0)
        ce_30min = calc_change(curr_oi.ce_oi, prev_30.ce_oi if prev_30 else 0)
        pe_30min = calc_change(curr_oi.pe_oi, prev_30.pe_oi if prev_30 else 0)
        
        # Calculate weight
        is_atm = (strike == current.atm_strike)
        distance = abs(strike - current.atm_strike)
        weight = self._calculate_strike_weight(strike, current.atm_strike)
        
        # Put/Call Ratio
        pcr = (curr_oi.pe_oi / curr_oi.ce_oi) if curr_oi.ce_oi > 0 else 0
        
        # Writer actions
        ce_action = self._determine_writer_action(ce_15min)
        pe_action = self._determine_writer_action(pe_15min)
        
        # Signal strengths
        bull_strength, bear_strength = self._calculate_signal_strength(ce_15min, pe_15min, weight)
        
        # Strike recommendation
        if bull_strength >= 7 and bull_strength > bear_strength:
            recommendation = "STRONG_CALL"
            confidence = min(10, bull_strength)
        elif bear_strength >= 7 and bear_strength > bull_strength:
            recommendation = "STRONG_PUT"
            confidence = min(10, bear_strength)
        else:
            recommendation = "WAIT"
            confidence = max(bull_strength, bear_strength)
        
        return StrikeAnalysis(
            strike=strike,
            is_atm=is_atm,
            distance_from_atm=distance,
            weight=weight,
            ce_oi=curr_oi.ce_oi,
            pe_oi=curr_oi.pe_oi,
            ce_ltp=curr_oi.ce_ltp,
            pe_ltp=curr_oi.pe_ltp,
            ce_change_5min=ce_5min,
            pe_change_5min=pe_5min,
            ce_change_15min=ce_15min,
            pe_change_15min=pe_15min,
            ce_change_30min=ce_30min,
            pe_change_30min=pe_30min,
            put_call_ratio=pcr,
            ce_writer_action=ce_action,
            pe_writer_action=pe_action,
            is_support_level=False,  # Will be set later
            is_resistance_level=False,  # Will be set later
            bullish_signal_strength=bull_strength,
            bearish_signal_strength=bear_strength,
            strike_recommendation=recommendation,
            confidence=confidence
        )
    
    async def analyze(self, current: MarketSnapshot) -> Dict:
        """Complete market analysis with strike-wise breakdown"""
        snap_5min = await self.cache.get_minutes_ago(5)
        snap_15min = await self.cache.get_minutes_ago(15)
        snap_30min = await self.cache.get_minutes_ago(30)
        
        if not snap_5min:
            return {
                "available": False, 
                "reason": "Building cache (need at least 5 min)..."
            }
        
        # Analyze each strike
        strike_analyses = []
        for strike in sorted(current.strikes_oi.keys()):
            analysis = await self.analyze_strike(strike, current, snap_5min, snap_15min, snap_30min)
            if analysis:
                strike_analyses.append(analysis)
        
        # Find Support/Resistance
        support_resistance = self._find_support_resistance(current, strike_analyses)
        
        # Mark S/R strikes
        for sa in strike_analyses:
            sa.is_support_level = (sa.strike == support_resistance.support_strike)
            sa.is_resistance_level = (sa.strike == support_resistance.resistance_strike)
        
        # Overall market signal
        total_bull = sum(sa.bullish_signal_strength for sa in strike_analyses)
        total_bear = sum(sa.bearish_signal_strength for sa in strike_analyses)
        
        if total_bull > total_bear and total_bull >= 10:
            overall_signal = "BULLISH"
        elif total_bear > total_bull and total_bear >= 10:
            overall_signal = "BEARISH"
        else:
            overall_signal = "NEUTRAL"
        
        return {
            "available": True,
            "strike_analyses": strike_analyses,
            "support_resistance": support_resistance,
            "overall_signal": overall_signal,
            "total_bullish_strength": total_bull,
            "total_bearish_strength": total_bear,
            "has_15min": snap_15min is not None,
            "has_30min": snap_30min is not None,
            "has_strong_signal": any(sa.confidence >= 7 for sa in strike_analyses)
        }
    
    def _find_support_resistance(self, 
                                 current: MarketSnapshot,
                                 analyses: List[StrikeAnalysis]) -> SupportResistance:
        """Find S/R levels from OI"""
        
        # Support = Highest PUT OI
        max_put_oi = 0
        support_strike = current.atm_strike
        
        for sa in analyses:
            if sa.pe_oi > max_put_oi:
                max_put_oi = sa.pe_oi
                support_strike = sa.strike
        
        # Resistance = Highest CALL OI
        max_call_oi = 0
        resistance_strike = current.atm_strike
        
        for sa in analyses:
            if sa.ce_oi > max_call_oi:
                max_call_oi = sa.ce_oi
                resistance_strike = sa.strike
        
        # Check if spot near S/R
        spot = current.spot_price
        near_support = abs(spot - support_strike) <= 50
        near_resistance = abs(spot - resistance_strike) <= 50
        
        return SupportResistance(
            support_strike=support_strike,
            support_put_oi=max_put_oi,
            resistance_strike=resistance_strike,
            resistance_call_oi=max_call_oi,
            spot_near_support=near_support,
            spot_near_resistance=near_resistance
        )


# ======================== DEEPSEEK CLIENT ========================
class DeepSeekClient:
    """DeepSeek API integration with 30-sec timeout"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
    
    async def analyze(self, prompt: str) -> Optional[Dict]:
        """Send prompt to DeepSeek with 30-sec timeout"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1200  # Increased for strike-wise analysis
        }
        
        try:
            # ✅ INCREASED TIMEOUT: 10 → 30 seconds
            timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Extract JSON
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]
                        if content.endswith('```'):
                            content = content[:-3]
                        content = content.strip()
                        
                        return json.loads(content)
                    else:
                        logger.error(f"❌ DeepSeek API error: {resp.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"❌ DeepSeek timeout (>{DEEPSEEK_TIMEOUT} seconds)")
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return None


# ======================== ENHANCED PROMPT BUILDER ========================
class EnhancedPromptBuilder:
    """Build detailed strike-wise prompts"""
    
    @staticmethod
    def build(
        spot: float,
        atm: int,
        oi_analysis: Dict,
        candles_5min: pd.DataFrame,
        patterns: List[Dict],
        price_support: float,
        price_resistance: float
    ) -> str:
        """Build comprehensive prompt with strike-wise analysis"""
        
        now_time = datetime.now(IST).strftime('%H:%M IST')
        
        strike_analyses = oi_analysis.get("strike_analyses", [])
        sr = oi_analysis.get("support_resistance")
        
        # Header
        prompt = f"""You are an expert NIFTY options trader with deep OI analysis skills.

MARKET STATE:
═════════════
Time: {now_time}
NIFTY Spot: ₹{spot:,.2f}
ATM Strike: {atm}

SUPPORT/RESISTANCE (OI-Based):
═══════════════════════════════
🟢 Support: {sr.support_strike} (PUT OI: {sr.support_put_oi:,})
🔴 Resistance: {sr.resistance_strike} (CALL OI: {sr.resistance_call_oi:,})
"""
        
        if sr.spot_near_support:
            prompt += f"⚡ ALERT: Spot NEAR SUPPORT ({sr.support_strike})!\n"
        if sr.spot_near_resistance:
            prompt += f"⚡ ALERT: Spot NEAR RESISTANCE ({sr.resistance_strike})!\n"
        
        prompt += "\n"
        
        # Strike-wise breakdown
        prompt += "STRIKE-WISE OI ANALYSIS (15-MIN PRIMARY):\n"
        prompt += "═" * 60 + "\n\n"
        
        for sa in strike_analyses:
            weight_marker = ""
            if sa.weight == ATM_WEIGHT:
                weight_marker = " ⭐⭐⭐ (ATM - 3x WEIGHT)"
            elif sa.weight == NEAR_ATM_WEIGHT:
                weight_marker = " ⭐⭐ (NEAR ATM - 2x WEIGHT)"
            else:
                weight_marker = " ⭐ (1x WEIGHT)"
            
            sr_marker = ""
            if sa.is_support_level:
                sr_marker = " 🟢 SUPPORT LEVEL"
            elif sa.is_resistance_level:
                sr_marker = " 🔴 RESISTANCE LEVEL"
            
            prompt += f"Strike: {sa.strike}{weight_marker}{sr_marker}\n"
            prompt += f"├─ CE OI: {sa.ce_oi:,} | 15min: {sa.ce_change_15min:+.1f}% ({sa.ce_writer_action})\n"
            prompt += f"├─ PE OI: {sa.pe_oi:,} | 15min: {sa.pe_change_15min:+.1f}% ({sa.pe_writer_action})\n"
            prompt += f"├─ PCR: {sa.put_call_ratio:.2f}\n"
            prompt += f"├─ Bull Strength: {sa.bullish_signal_strength:.1f}/10\n"
            prompt += f"├─ Bear Strength: {sa.bearish_signal_strength:.1f}/10\n"
            prompt += f"└─ Signal: {sa.strike_recommendation} (Conf: {sa.confidence:.1f}/10)\n\n"
        
        # Price action
        prompt += "\nPRICE ACTION (Last 1 Hour - 5min candles):\n"
        prompt += "═" * 60 + "\n\n"
        
        if not candles_5min.empty and len(candles_5min) > 0:
            last_12 = candles_5min.tail(min(12, len(candles_5min)))
            for idx, row in last_12.iterrows():
                time_str = idx.strftime('%H:%M')
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                dir_emoji = "🟢" if c > o else "🔴" if c < o else "⚪"
                delta = c - o
                prompt += f"{time_str} | {o:.0f}→{c:.0f} (Δ{delta:+.0f}) | H:{h:.0f} L:{l:.0f} {dir_emoji}\n"
            
            prompt += f"\nPrice S/R (from candles):\n"
            prompt += f"├─ Support: ₹{price_support:.2f}\n"
            prompt += f"└─ Resistance: ₹{price_resistance:.2f}\n"
        else:
            prompt += "No candle data available (focus on OI only)\n"
        
        # Patterns
        prompt += "\n\nKEY CANDLESTICK PATTERNS:\n"
        prompt += "═" * 60 + "\n\n"
        
        if patterns:
            for p in patterns:
                time_str = p['time'].strftime('%H:%M')
                prompt += f"{time_str}: {p['pattern']} | {p['type']} | Strength: {p['strength']}/10 | @ ₹{p['price']:.0f}\n"
        else:
            prompt += "No significant patterns detected\n"
        
        # Instructions
        prompt += f"""

ANALYSIS INSTRUCTIONS:
═════════════════════

🚨 CRITICAL OI LOGIC (PAY ATTENTION):
─────────────────────────────────────
OI = Option Writers/Sellers (NOT Buyers!)

✅ CORRECT INTERPRETATION:
• CALL OI ↑ = Call Writers Building = RESISTANCE = BEARISH → BUY_PUT
• PUT OI ↑ = Put Writers Building = SUPPORT = BULLISH → BUY_CALL
• CALL OI ↓ = Writers Covering = Resistance Breaking = BULLISH → BUY_CALL
• PUT OI ↓ = Writers Covering = Support Breaking = BEARISH → BUY_PUT

📊 FOCUS PRIORITY:
─────────────────
1. ATM Strike (3x importance) - Look here FIRST
2. ATM ±50 Strikes (2x importance)
3. Support/Resistance strikes from OI
4. Candlestick confirmation

🎯 SIGNAL DECISION:
──────────────────
- If ATM shows STRONG CALL signal (7+) → BUY_CALL
- If ATM shows STRONG PUT signal (7+) → BUY_PUT
- If ATM neutral, check ATM±50 strikes
- Confirm with candlestick patterns
- Check if spot near S/R for better entry

⚡ ENTRY TIMING:
───────────────
- Spot near Support + Bullish OI = ENTER NOW (BUY_CALL)
- Spot near Resistance + Bearish OI = ENTER NOW (BUY_PUT)
- Strong OI signal but spot mid-range = WAIT for S/R test

RESPOND IN JSON:
{{
    "signal": "BUY_CALL" | "BUY_PUT" | "WAIT",
    "primary_strike": {atm},  // Which strike to trade
    "confidence": 0-10,
    "stop_loss_strike": strike_number,  // Realistic SL strike
    "target_strike": strike_number,     // Realistic target strike
    
    "atm_analysis": {{
        "ce_oi_action": "BUILDING/UNWINDING/NEUTRAL",
        "pe_oi_action": "BUILDING/UNWINDING/NEUTRAL",
        "atm_signal": "CALL/PUT/WAIT",
        "atm_confidence": 0-10
    }},
    
    "strike_breakdown": [
        {{
            "strike": {atm},
            "recommendation": "STRONG_CALL/STRONG_PUT/WAIT",
            "reason": "Why this strike signals this direction"
        }}
        // Include 2-3 most important strikes
    ],
    
    "oi_support_resistance": {{
        "oi_support": {sr.support_strike},
        "oi_resistance": {sr.resistance_strike},
        "spot_position": "NEAR_SUPPORT/NEAR_RESISTANCE/MID_RANGE",
        "sr_impact": "How S/R affects trade decision"
    }},
    
    "candlestick_confirmation": {{
        "patterns_detected": ["list of patterns"],
        "patterns_confirm_oi": true/false,
        "pattern_strength": 0-10
    }},
    
    "entry_timing": {{
        "enter_now": true/false,
        "reason": "Why now or why wait",
        "wait_for": "What to wait for (if not entering now)"
    }},
    
    "risk_reward": {{
        "entry_premium_estimate": 0,  // Rough estimate in ₹
        "sl_points": 0,  // Points from entry
        "target_points": 0,  // Points from entry
        "rr_ratio": 0  // Risk:Reward ratio
    }}
}}

ONLY output valid JSON, no extra text.
"""
        
        return prompt


# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    """Enhanced Telegram alerts with strike-wise details"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.session = None
    
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    async def send_signal(self, signal: Dict, spot: float, oi_data: Dict):
        """Send enhanced trade signal with strike breakdown"""
        
        confidence = signal.get('confidence', 0)
        signal_type = signal.get('signal', 'WAIT')
        primary_strike = signal.get('primary_strike', 0)
        sl_strike = signal.get('stop_loss_strike', 0)
        target_strike = signal.get('target_strike', 0)
        
        atm = oi_data.get("strike_analyses", [{}])[0]
        sr = oi_data.get("support_resistance")
        
        # ATM analysis
        atm_analysis = signal.get('atm_analysis', {})
        
        # Strike breakdown
        strike_breakdown = signal.get('strike_breakdown', [])
        
        # S/R
        sr_analysis = signal.get('oi_support_resistance', {})
        
        # Candlestick
        candle_analysis = signal.get('candlestick_confirmation', {})
        
        # Entry timing
        entry_timing = signal.get('entry_timing', {})
        
        # Risk/Reward
        rr = signal.get('risk_reward', {})
        
        message = f"""🚨 NIFTY TRADE SIGNAL v5.0

⏰ {datetime.now(IST).strftime('%d-%b %H:%M:%S IST')}

💰 Spot: ₹{spot:,.2f}
📊 Signal: <b>{signal_type}</b>
⭐ Confidence: {confidence}/10

💼 TRADE SETUP:
━━━━━━━━━━━━━━━━━━━
Entry Strike: {primary_strike} {"CE" if "CALL" in signal_type else "PE" if "PUT" in signal_type else ""}
Stop Loss: {sl_strike}
Target: {target_strike}
Risk:Reward = {rr.get('rr_ratio', 'N/A')}

📊 ATM ANALYSIS ({atm.strike if hasattr(atm, 'strike') else 'N/A'}):
━━━━━━━━━━━━━━━━━━━
CE Writers: {atm_analysis.get('ce_oi_action', 'N/A')}
PE Writers: {atm_analysis.get('pe_oi_action', 'N/A')}
ATM Signal: {atm_analysis.get('atm_signal', 'N/A')}
ATM Confidence: {atm_analysis.get('atm_confidence', 0)}/10

🎯 KEY STRIKES:
━━━━━━━━━━━━━━━━━━━
"""
        
        for sb in strike_breakdown[:3]:
            message += f"\n{sb.get('strike')}: {sb.get('recommendation')}\n└─ {sb.get('reason')}\n"
        
        message += f"""
🟢🔴 SUPPORT/RESISTANCE:
━━━━━━━━━━━━━━━━━━━
Support: {sr.support_strike if sr else 'N/A'} (PUT OI: {sr.support_put_oi if sr else 0:,})
Resistance: {sr.resistance_strike if sr else 'N/A'} (CALL OI: {sr.resistance_call_oi if sr else 0:,})
Position: {sr_analysis.get('spot_position', 'N/A')}

🕯️ CANDLESTICK:
━━━━━━━━━━━━━━━━━━━
Patterns: {', '.join(candle_analysis.get('patterns_detected', ['None']))}
Confirms OI: {"✅" if candle_analysis.get('patterns_confirm_oi') else "❌"}
Strength: {candle_analysis.get('pattern_strength', 0)}/10

⏰ ENTRY TIMING:
━━━━━━━━━━━━━━━━━━━
Enter Now: {"✅ YES" if entry_timing.get('enter_now') else "⏳ WAIT"}
Reason: {entry_timing.get('reason', 'N/A')}
"""
        
        if not entry_timing.get('enter_now'):
            message += f"Wait For: {entry_timing.get('wait_for', 'N/A')}\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━\n"
        message += f"🤖 DeepSeek V3 ({DEEPSEEK_TIMEOUT}s timeout)\n"
        message += "📊 Strike-wise Analysis v5.0"
        
        try:
            await self._ensure_session()
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.info("✅ Enhanced alert sent to Telegram")
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ Telegram error: {resp.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")


# ======================== MAIN BOT ========================
class NiftyOptionsBot:
    """Enhanced main trading bot v5.0"""
    
    def __init__(self):
        self.upstox = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.cache = SimpleCache()
        self.oi_analyzer = EnhancedOIAnalyzer(self.cache)
        self.deepseek = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.pattern_detector = PatternDetector()
        self.prompt_builder = EnhancedPromptBuilder()
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now(IST)
        
        if now.weekday() >= 5:
            return False
        
        market_start = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MIN)
        market_end = now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MIN)
        
        return market_start <= now <= market_end
    
    async def fetch_market_data(self) -> Optional[MarketSnapshot]:
        """Fetch current market data"""
        try:
            expiry = await self.upstox.get_nearest_expiry()
            if not expiry:
                logger.warning("⚠️ Could not determine expiry from Upstox API")
                return None
            
            logger.info(f"📅 Using expiry: {expiry}")
            
            await asyncio.sleep(API_DELAY)
            chain_data = await self.upstox.get_option_chain(expiry)
            
            if not chain_data or chain_data.get("status") != "success":
                logger.warning("⚠️ Could not fetch option chain")
                return None
            
            chain = chain_data.get("data", [])
            
            if not chain or len(chain) == 0:
                logger.warning(f"⚠️ Empty option chain for expiry: {expiry}")
                return None
            
            # Extract spot
            first_item = chain[0]
            spot = first_item.get("underlying_spot_price", 0.0)
            
            if spot == 0:
                for item in chain:
                    spot = item.get("underlying_spot_price", 0.0)
                    if spot > 0:
                        break
            
            if spot == 0:
                logger.warning("⚠️ Could not extract spot price")
                return None
            
            logger.info(f"💰 NIFTY Spot: ₹{spot:,.2f}")
            
            # Calculate ATM
            atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            
            # Extract strikes
            min_strike = atm - (ATM_RANGE * STRIKE_INTERVAL)
            max_strike = atm + (ATM_RANGE * STRIKE_INTERVAL)
            
            strikes_oi = {}
            
            for item in chain:
                strike = item.get("strike_price")
                
                if not (min_strike <= strike <= max_strike):
                    continue
                
                ce_data = item.get("call_options", {}).get("market_data", {})
                pe_data = item.get("put_options", {}).get("market_data", {})
                
                strikes_oi[strike] = OISnapshot(
                    strike=strike,
                    ce_oi=ce_data.get("oi", 0),
                    pe_oi=pe_data.get("oi", 0),
                    ce_ltp=ce_data.get("ltp", 0.0),
                    pe_ltp=pe_data.get("ltp", 0.0),
                    timestamp=datetime.now(IST)
                )
            
            if not strikes_oi:
                logger.warning(f"⚠️ No strikes found in range")
                return None
            
            logger.info(f"📊 Fetched {len(strikes_oi)} strikes (ATM: {atm})")
            
            return MarketSnapshot(
                timestamp=datetime.now(IST),
                spot_price=spot,
                atm_strike=atm,
                strikes_oi=strikes_oi
            )
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            logger.exception("Full traceback:")
            return None
    
    async def analyze_cycle(self):
        """Main enhanced analysis cycle"""
        logger.info("\n" + "="*60)
        logger.info(f"🔍 ANALYSIS CYCLE v5.0 - {datetime.now(IST).strftime('%H:%M:%S')}")
        logger.info("="*60)
        
        # Fetch data
        current_snapshot = await self.fetch_market_data()
        
        if not current_snapshot:
            logger.warning("⚠️ Skipping cycle - no data")
            return
        
        # Add to cache
        await self.cache.add(current_snapshot)
        
        # Enhanced OI analysis
        oi_analysis = await self.oi_analyzer.analyze(current_snapshot)
        
        if not oi_analysis.get("available"):
            logger.info(f"⏳ {oi_analysis.get('reason', 'Building cache...')}")
            return
        
        # Log strike-wise signals
        strike_analyses = oi_analysis.get("strike_analyses", [])
        logger.info("\n📊 STRIKE-WISE SIGNALS:")
        for sa in strike_analyses:
            if sa.confidence >= 5:
                logger.info(f"  {sa.strike}: {sa.strike_recommendation} (Conf: {sa.confidence:.1f}/10)")
        
        # Check for strong signals
        if not oi_analysis.get("has_strong_signal"):
            logger.info("📊 No strong signals (all strikes < 7 confidence)")
            return
        
        logger.info("🚨 Strong signal detected! Proceeding to AI analysis...")
        
        # Fetch candles
        candles_1min = await self.upstox.get_1min_candles()
        
        # Resample to 5-min
        if not candles_1min.empty and len(candles_1min) >= 5:
            try:
                candles_5min = candles_1min.resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"📊 Resampled to {len(candles_5min)} 5-min candles")
            except Exception as e:
                logger.warning(f"⚠️ Candle resampling error: {e}")
                candles_5min = pd.DataFrame()
        else:
            candles_5min = pd.DataFrame()
        
        # Detect patterns
        patterns = self.pattern_detector.detect(candles_5min) if not candles_5min.empty else []
        
        if patterns:
            logger.info(f"🎯 Detected {len(patterns)} patterns")
            for p in patterns:
                logger.info(f"  {p['pattern']}: {p['type']} (Strength: {p['strength']}/10)")
        
        # Calculate price S/R
        price_support, price_resistance = self.pattern_detector.calculate_support_resistance(candles_5min)
        
        # Build enhanced prompt
        prompt = self.prompt_builder.build(
            spot=current_snapshot.spot_price,
            atm=current_snapshot.atm_strike,
            oi_analysis=oi_analysis,
            candles_5min=candles_5min,
            patterns=patterns,
            price_support=price_support,
            price_resistance=price_resistance
        )
        
        logger.info(f"🤖 Sending to DeepSeek (timeout: {DEEPSEEK_TIMEOUT}s)...")
        
        # Get AI signal
        ai_signal = await self.deepseek.analyze(prompt)
        
        if not ai_signal:
            logger.warning("⚠️ DeepSeek timeout/error - using enhanced fallback")
            
            # Enhanced fallback with strike-wise logic
            atm_strike_analysis = next((sa for sa in strike_analyses if sa.is_atm), None)
            
            if atm_strike_analysis:
                if atm_strike_analysis.bullish_signal_strength > atm_strike_analysis.bearish_signal_strength:
                    fallback_signal = "BUY_CALL"
                    fallback_conf = min(10, atm_strike_analysis.bullish_signal_strength)
                elif atm_strike_analysis.bearish_signal_strength > atm_strike_analysis.bullish_signal_strength:
                    fallback_signal = "BUY_PUT"
                    fallback_conf = min(10, atm_strike_analysis.bearish_signal_strength)
                else:
                    fallback_signal = "WAIT"
                    fallback_conf = 5
            else:
                fallback_signal = "WAIT"
                fallback_conf = 4
            
            ai_signal = {
                'signal': fallback_signal,
                'confidence': fallback_conf,
                'primary_strike': current_snapshot.atm_strike,
                'stop_loss_strike': current_snapshot.atm_strike - 100 if fallback_signal == "BUY_CALL" else current_snapshot.atm_strike + 100,
                'target_strike': current_snapshot.atm_strike + 150 if fallback_signal == "BUY_CALL" else current_snapshot.atm_strike - 150,
                'atm_analysis': {
                    'ce_oi_action': 'UNKNOWN',
                    'pe_oi_action': 'UNKNOWN',
                    'atm_signal': fallback_signal.replace('BUY_', ''),
                    'atm_confidence': fallback_conf
                },
                'strike_breakdown': [{'strike': current_snapshot.atm_strike, 'recommendation': fallback_signal, 'reason': 'Fallback - AI unavailable'}],
                'oi_support_resistance': {'spot_position': 'UNKNOWN', 'sr_impact': 'AI unavailable'},
                'candlestick_confirmation': {'patterns_detected': [], 'patterns_confirm_oi': False, 'pattern_strength': 0},
                'entry_timing': {'enter_now': False, 'reason': 'AI timeout - manual verification needed', 'wait_for': 'Manual check'},
                'risk_reward': {'rr_ratio': 1.5}
            }
        
        confidence = ai_signal.get('confidence', 0)
        signal_type = ai_signal.get('signal', 'WAIT')
        
        logger.info(f"🎯 Signal: {signal_type} | Confidence: {confidence}/10")
        
        # Send alert if confidence >= threshold
        if confidence >= MIN_CONFIDENCE:
            logger.info("✅ Sending enhanced Telegram alert...")
            await self.alerter.send_signal(ai_signal, current_snapshot.spot_price, oi_analysis)
        else:
            logger.info(f"⏳ Low confidence ({confidence}/10), no alert sent")
        
        logger.info("="*60 + "\n")
    
    async def run(self):
        """Main bot loop"""
        logger.info("\n" + "="*60)
        logger.info("🚀 NIFTY OPTIONS BOT v5.0 - STRIKE-WISE ANALYSIS!")
        logger.info("="*60)
        logger.info(f"📅 {datetime.now(IST).strftime('%d-%b-%Y %A')}")
        logger.info(f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"⏱️  Interval: {ANALYSIS_INTERVAL // 60} minutes")
        logger.info(f"📊 Symbol: {SYMBOL}")
        logger.info(f"🎯 ATM Range: ±{ATM_RANGE} strikes")
        logger.info(f"⭐ Strike Weights: ATM={ATM_WEIGHT}x, ±50={NEAR_ATM_WEIGHT}x, Others={FAR_WEIGHT}x")
        logger.info(f"💾 Cache: {CACHE_SIZE} snapshots (30 min)")
        logger.info(f"🤖 AI: DeepSeek V3 ({DEEPSEEK_TIMEOUT}s timeout)")
        logger.info(f"📈 Features: Strike-wise OI + S/R + Candles")
        logger.info("="*60 + "\n")
        
        await self.upstox.init()
        
        try:
            while True:
                try:
                    if self.is_market_open():
                        await self.analyze_cycle()
                    else:
                        logger.info("💤 Market closed, waiting...")
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next cycle: {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Full traceback:")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        
        finally:
            await self.upstox.close()
            await self.alerter.close()
            logger.info("👋 Session closed")


# ======================== KOYEB HTTP WRAPPER ========================
async def health_check(request):
    """Health check endpoint"""
    return aiohttp.web.Response(text="✅ NIFTY Bot v5.0 Running! (Strike-wise Analysis)")


async def start_bot_background(app):
    """Start bot in background"""
    app['bot_task'] = asyncio.create_task(run_trading_bot())


async def run_trading_bot():
    """Run the bot"""
    bot = NiftyOptionsBot()
    await bot.run()


# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    from aiohttp import web
    
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.on_startup.append(start_bot_background)
    
    port = int(os.getenv('PORT', 8000))
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║   🚀 NIFTY OPTIONS BOT v5.0                         ║
║   COMPLETE REWRITE - Strike-wise Analysis!          ║
╚══════════════════════════════════════════════════════╝

✅ MAJOR IMPROVEMENTS:
  • Strike-wise separate analysis
  • ATM strike 3x focus weight
  • Support/Resistance from OI
  • DeepSeek timeout: 10→30 seconds
  • Better candlestick integration
  • Realistic SL/Target setup
  • Enhanced AI prompts

⚡ NEW FEATURES:
  • Per-strike signal strength
  • OI-based S/R levels
  • Price action S/R
  • Strike-wise recommendations
  • Entry timing logic
  • Risk:Reward calculation

Starting HTTP server on port {port}...
Bot will run in background.

Access: http://localhost:{port}/
""")
    
    web.run_app(app, host='0.0.0.0', port=port)
