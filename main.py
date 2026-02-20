"""
🚀 NIFTY 50 OPTIONS BOT v6.1 PRO
==================================
Platform  : NSE via Upstox API v2
Asset     : NIFTY 50 Weekly Options
Lot Size  : 65
Updated   : Feb 2026

✅ v6.1 CHANGES FROM v6.0:
- ATM_RANGE: 6 → 4  (±200 pts, focused strikes)
- PHASE1_OI_BUILD_PCT: 4.0 → 6.0  (less false positives on NIFTY)
- PHASE2_VOL_SPIKE_PCT: 20.0 → 15.0  (realistic threshold)
- Overall MTF threshold: 12 → 8  (adjusted for 9 strikes)
- Candle fetch FIX: 3 API calls → 2 (1min + 30min only, Upstox V2 supports these)
  → 1min fetched once, Pandas resample to 5min (price action / patterns)
  → 30min fetched directly (day trend / S/R levels)
  → 15min was INVALID on Upstox V2 — removed
- Expiry refresh: har 30min → fakt daily once (unnecessary API calls hotte)
- Koyeb deployment compatible (PORT env, health endpoint)

✅ v6.0 FEATURES (unchanged):
- ETH bot architecture (DualCache + PhaseDetector + MTFAnalyzer)
- 5-min polling (real-time OI/Volume/PCR)
- DUAL CACHE: 5-min x72 (6hr) + 30-min x12 (6hr)
- MTF Analysis: 5min + 15min + 30min OI/Volume/PCR (via DualCache snapshots)
- PHASE DETECTION: Phase1 → Phase2 → Phase3
- Pandas/Numpy pre-calculation before AI call
- Smart AI trigger: DeepSeek only when needed
- Expiry auto-detection (3:30PM + holiday safe)
- Market hours check (IST 9:15 - 3:30)
- DeepSeek V3 (deepseek-chat)
- NIFTY specific: Strike interval 50, Lot size 65, IV analysis
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
import time as time_module

# ============================================================
#  CONFIGURATION
# ============================================================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_TOKEN")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN",  "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID",     "YOUR_CHAT_ID")
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY",     "YOUR_DEEPSEEK_KEY")

UPSTOX_BASE    = "https://api.upstox.com/v2"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

LOT_SIZE        = 65
STRIKE_INTERVAL = 50
ATM_RANGE       = 4          # v6.1: was 6 → now ±200 pts (9 strikes)

SNAPSHOT_INTERVAL = 5 * 60
ANALYSIS_INTERVAL = 30 * 60

CACHE_5MIN_SIZE  = 72
CACHE_30MIN_SIZE = 12
CANDLE_COUNT     = 24        # candles after resample (not raw 1min)

MIN_OI_CHANGE    = 8.0
STRONG_OI_CHANGE = 15.0
MIN_VOLUME_CHG   = 15.0
PCR_BULL         = 1.2
PCR_BEAR         = 0.8
MIN_CONFIDENCE   = 7

PHASE1_OI_BUILD_PCT   = 6.0   # v6.1: was 4.0 → less false positives on NIFTY
PHASE1_VOL_MAX_PCT    = 12.0
PHASE2_VOL_SPIKE_PCT  = 15.0  # v6.1: was 20.0 → realistic for NIFTY
PHASE2_OI_MIN_PCT     = 3.0
PHASE3_PRICE_MOVE_PCT = 0.25

OI_ALERT_PCT  = 12.0
VOL_SPIKE_PCT = 25.0
PCR_ALERT_PCT = 10.0
ATM_PROX_PTS  = 50

ATM_WEIGHT      = 3.0
NEAR_ATM_WEIGHT = 2.0
FAR_WEIGHT      = 1.0

# v6.1: adjusted for 9 strikes (ATM_RANGE=4)
MTF_OVERALL_THRESHOLD = 8    # was 12

MAX_RETRIES      = 3
API_DELAY        = 0.3
DEEPSEEK_TIMEOUT = 45

IST = pytz.timezone("Asia/Kolkata")
MARKET_START = (9, 15)
MARKET_END   = (15, 30)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class OISnapshot:
    strike:    int
    ce_oi:     float
    pe_oi:     float
    ce_volume: float
    pe_volume: float
    ce_ltp:    float
    pe_ltp:    float
    ce_iv:     float
    pe_iv:     float
    pcr:       float
    timestamp: datetime


@dataclass
class MarketSnapshot:
    timestamp:    datetime
    spot_price:   float
    atm_strike:   int
    expiry:       str
    strikes_oi:   Dict[int, OISnapshot]
    overall_pcr:  float
    total_ce_oi:  float
    total_pe_oi:  float
    total_ce_vol: float
    total_pe_vol: float


@dataclass
class PhaseSignal:
    phase:            int
    dominant_side:    str
    direction:        str
    oi_change_pct:    float
    vol_change_pct:   float
    price_change_pct: float
    atm_strike:       int
    spot_price:       float
    confidence:       float
    message:          str


@dataclass
class TrendInfo:
    day_trend:      str
    intraday_trend: str
    trend_5min:     str
    trend_30min:    str     # v6.1: was trend_15min, now uses 30min candles
    vwap:           float
    spot_vs_vwap:   str
    all_agree:      bool
    summary:        str


@dataclass
class PriceActionInsight:
    price_change_5m:   float
    price_change_15m:  float
    price_change_30m:  float
    price_momentum:    str
    vol_rolling_avg:   float
    vol_spike_ratio:   float
    oi_vol_corr:       float
    price_oi_corr:     float
    support_levels:    List[float]
    resistance_levels: List[float]
    trend_strength:    float
    triple_confirmed:  bool
    trend:             TrendInfo


@dataclass
class StrikeAnalysis:
    strike:        int
    is_atm:        bool
    distance_atm:  int
    weight:        float
    ce_oi:         float
    pe_oi:         float
    ce_volume:     float
    pe_volume:     float
    ce_ltp:        float
    pe_ltp:        float
    ce_iv:         float
    pe_iv:         float
    ce_oi_5:       float
    pe_oi_5:       float
    ce_vol_5:      float
    pe_vol_5:      float
    ce_oi_15:      float
    pe_oi_15:      float
    ce_vol_15:     float
    pe_vol_15:     float
    pcr_ch_15:     float
    ce_oi_30:      float
    pe_oi_30:      float
    ce_vol_30:     float
    pe_vol_30:     float
    pcr:           float
    ce_action:     str
    pe_action:     str
    tf5_signal:    str
    tf15_signal:   str
    tf30_signal:   str
    mtf_confirmed: bool
    vol_confirms:  bool
    vol_strength:  str
    is_support:    bool
    is_resistance: bool
    bull_strength: float
    bear_strength: float
    recommendation: str
    confidence:    float


@dataclass
class SupportResistance:
    support_strike:     int
    support_put_oi:     float
    resistance_strike:  int
    resistance_call_oi: float
    near_support:       bool
    near_resistance:    bool


# ============================================================
#  DUAL CACHE
# ============================================================

class DualCache:
    def __init__(self):
        self._c5   = deque(maxlen=CACHE_5MIN_SIZE)
        self._c30  = deque(maxlen=CACHE_30MIN_SIZE)
        self._lock = asyncio.Lock()

    async def add_5min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c5.append(snap)
        logger.info(f"📦 5min cache: {len(self._c5)}/{CACHE_5MIN_SIZE} | PCR:{snap.overall_pcr:.2f}")

    async def add_30min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c30.append(snap)

    async def get_5min_ago(self, n: int) -> Optional[MarketSnapshot]:
        async with self._lock:
            idx = len(self._c5) - 1 - n
            return self._c5[idx] if idx >= 0 else None

    async def get_30min_ago(self, n: int) -> Optional[MarketSnapshot]:
        async with self._lock:
            idx = len(self._c30) - 1 - n
            return self._c30[idx] if idx >= 0 else None

    async def get_recent(self, n: int) -> List[MarketSnapshot]:
        async with self._lock:
            lst = list(self._c5)
            return lst[-n:] if len(lst) >= n else lst

    def sizes(self) -> Tuple[int, int]:
        return len(self._c5), len(self._c30)

    def has_data(self) -> bool:
        return len(self._c5) >= 3


# ============================================================
#  UPSTOX CLIENT
# ============================================================

class UpstoxClient:
    """
    v6.1 Candle fix:
    Upstox V2 intraday supports ONLY 1minute and 30minute.
    5min / 15min endpoints return error on V2.
    Solution:
      - get_candles() → fetches 1min + 30min (2 API calls)
      - 1min → Pandas resample → 5min (used for price action, patterns, trend)
      - 30min → used for day trend, S/R levels
    OI/Volume MTF (5min/15min/30min) uses DualCache snapshots — NO candle API needed.
    """

    def __init__(self, token: str):
        self.token   = token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json"
        }

    async def init(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self):
        if self.session:
            await self.session.close()

    async def _get(self, url: str, params: Dict = None) -> Optional[Dict]:
        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429:
                        await asyncio.sleep((attempt + 1) * 3)
                        continue
                    txt = await r.text()
                    logger.warning(f"⚠️ {url} {r.status}: {txt[:100]}")
                    return None
            except aiohttp.ClientConnectorError:
                logger.error(f"❌ Network ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Request: {e}")
                await asyncio.sleep(1)
        return None

    async def get_nearest_expiry(self) -> Optional[str]:
        """Auto expiry — 3:30PM cutoff + holiday safe"""
        data = await self._get(f"{UPSTOX_BASE}/option/contract",
                               params={"instrument_key": INSTRUMENT_KEY})
        if not data or data.get("status") != "success":
            return None

        now_ist = datetime.now(IST)
        cutoff  = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        today   = now_ist.date()

        expiry_set = sorted(set(
            c.get("expiry") for c in data.get("data", []) if c.get("expiry")
        ))

        for exp in expiry_set:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            if exp_date < today:
                continue
            if exp_date == today and now_ist >= cutoff:
                continue
            logger.info(f"📅 Expiry: {exp}")
            return exp

        logger.error("❌ No valid expiry")
        return None

    async def _fetch_raw_candles(self, resolution: str) -> pd.DataFrame:
        """
        Internal: fetch intraday candles for given resolution.
        resolution = '1minute' or '30minute' (only these valid on Upstox V2)
        Returns raw DataFrame with all intraday candles (no tail applied here).
        """
        url  = f"{UPSTOX_BASE}/historical-candle/intraday/{INSTRUMENT_KEY}/{resolution}"
        data = await self._get(url)

        if not data or data.get("status") != "success":
            logger.warning(f"⚠️ Candle fetch failed: {resolution}")
            return pd.DataFrame()

        raw = data.get("data", {}).get("candles", [])
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            try:
                rows.append({
                    "timestamp": pd.to_datetime(c[0]),
                    "open":      float(c[1]),
                    "high":      float(c[2]),
                    "low":       float(c[3]),
                    "close":     float(c[4]),
                    "volume":    int(c[5]) if len(c) > 5 else 0
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    async def get_candles(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        v6.1 FIX: 2 API calls instead of 3.
        Returns: (df_1m, df_5m, df_30m)
          - df_1m  : last CANDLE_COUNT 1min candles (for VWAP calculation)
          - df_5m  : 1min resampled to 5min, last CANDLE_COUNT candles
                     → used for: price action, patterns, intraday trend
          - df_30m : 30min candles directly from Upstox, last CANDLE_COUNT candles
                     → used for: day trend, S/R levels
        NOTE: OI/Volume MTF (5min/15min/30min) uses DualCache — NOT these candles.
        """
        # Fetch 1min (all intraday data — do NOT tail before resample)
        df_1m_raw = await self._fetch_raw_candles("1minute")
        await asyncio.sleep(API_DELAY)
        # Fetch 30min directly (Upstox V2 supports this)
        df_30m_raw = await self._fetch_raw_candles("30minute")

        # 1min tail for VWAP
        df_1m = df_1m_raw.tail(CANDLE_COUNT) if not df_1m_raw.empty else pd.DataFrame()

        # Resample 1min → 5min using ALL intraday data, then tail
        if not df_1m_raw.empty:
            df_5m = df_1m_raw.resample("5min").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            df_5m = df_5m.tail(CANDLE_COUNT)
            logger.info(f"📊 1min→5min resample: {len(df_5m)} candles")
        else:
            df_5m = pd.DataFrame()

        # 30min tail
        df_30m = df_30m_raw.tail(CANDLE_COUNT) if not df_30m_raw.empty else pd.DataFrame()
        logger.info(f"📊 30min candles: {len(df_30m)}")

        return df_1m, df_5m, df_30m

    async def fetch_snapshot(self, expiry: str) -> Optional[MarketSnapshot]:
        url  = f"{UPSTOX_BASE}/option/chain"
        data = await self._get(url, params={
            "instrument_key": INSTRUMENT_KEY,
            "expiry_date":    expiry
        })

        if not data or data.get("status") != "success":
            return None

        chain = data.get("data", [])
        if not chain:
            return None

        spot = 0.0
        for item in chain:
            spot = float(item.get("underlying_spot_price", 0))
            if spot > 0:
                break
        if spot <= 0:
            return None

        logger.info(f"💰 NIFTY: {spot:,.2f}")
        atm   = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        min_s = atm - ATM_RANGE * STRIKE_INTERVAL   # ATM_RANGE=4 → ±200 pts
        max_s = atm + ATM_RANGE * STRIKE_INTERVAL

        strikes_oi: Dict[int, OISnapshot] = {}
        t_ce_oi = t_pe_oi = t_ce_vol = t_pe_vol = 0.0

        for item in chain:
            strike = int(item.get("strike_price", 0))
            if not (min_s <= strike <= max_s):
                continue

            ce = item.get("call_options", {}).get("market_data", {})
            pe = item.get("put_options",  {}).get("market_data", {})
            cg = item.get("call_options", {}).get("option_greeks", {})
            pg = item.get("put_options",  {}).get("option_greeks", {})

            ce_oi  = float(ce.get("oi",     0) or 0)
            pe_oi  = float(pe.get("oi",     0) or 0)
            ce_vol = float(ce.get("volume", 0) or 0)
            pe_vol = float(pe.get("volume", 0) or 0)
            ce_ltp = float(ce.get("ltp",    0) or 0)
            pe_ltp = float(pe.get("ltp",    0) or 0)
            ce_iv  = float(cg.get("iv",     0) or 0) * 100
            pe_iv  = float(pg.get("iv",     0) or 0) * 100
            pcr    = (pe_oi / ce_oi) if ce_oi > 0 else 0.0

            t_ce_oi  += ce_oi;  t_pe_oi  += pe_oi
            t_ce_vol += ce_vol; t_pe_vol += pe_vol

            strikes_oi[strike] = OISnapshot(
                strike=strike, ce_oi=ce_oi, pe_oi=pe_oi,
                ce_volume=ce_vol, pe_volume=pe_vol,
                ce_ltp=ce_ltp, pe_ltp=pe_ltp,
                ce_iv=ce_iv, pe_iv=pe_iv, pcr=pcr,
                timestamp=datetime.now(IST)
            )

        if not strikes_oi:
            return None

        overall_pcr = (t_pe_oi / t_ce_oi) if t_ce_oi > 0 else 0.0
        logger.info(f"✅ {len(strikes_oi)} strikes | ATM:{atm} | PCR:{overall_pcr:.2f}")

        return MarketSnapshot(
            timestamp=datetime.now(IST),
            spot_price=spot, atm_strike=atm, expiry=expiry,
            strikes_oi=strikes_oi, overall_pcr=overall_pcr,
            total_ce_oi=t_ce_oi, total_pe_oi=t_pe_oi,
            total_ce_vol=t_ce_vol, total_pe_vol=t_pe_vol
        )


# ============================================================
#  TREND CALCULATOR
# ============================================================

class TrendCalculator:

    @staticmethod
    def vwap(df: pd.DataFrame) -> float:
        if df.empty or df["volume"].sum() == 0:
            return 0.0
        tp = (df["high"] + df["low"] + df["close"]) / 3
        return float((tp * df["volume"]).sum() / df["volume"].sum())

    @staticmethod
    def trend(df: pd.DataFrame) -> str:
        if df.empty or len(df) < 3:
            return "SIDEWAYS"
        c = df["close"].values
        if c[-1] > c[-3] * 1.001:  return "UPTREND"
        if c[-1] < c[-3] * 0.999:  return "DOWNTREND"
        return "SIDEWAYS"

    @staticmethod
    def calculate(df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                  df_30m: pd.DataFrame, spot: float) -> TrendInfo:
        """
        v6.1: df_15m replaced with df_30m (Upstox V2 only supports 1min + 30min)
        - VWAP from 1min data
        - Intraday trend from 5min
        - Day trend from 30min
        """
        vwap_val  = TrendCalculator.vwap(df_1m)
        day_t     = TrendCalculator.trend(df_30m)   # 30min for day trend
        intra_t   = TrendCalculator.trend(df_5m)    # 5min for intraday
        t5        = TrendCalculator.trend(df_5m.tail(4) if len(df_5m) >= 4 else df_5m)
        spot_vwap = "ABOVE" if vwap_val > 0 and spot > vwap_val else "BELOW"
        tfs       = [day_t, intra_t, t5]
        all_agree = len(set(tfs)) == 1 and tfs[0] != "SIDEWAYS"

        if all(t == "UPTREND"   for t in tfs):
            summary = f"📈 ALL UPTREND | Spot {spot_vwap} VWAP"
        elif all(t == "DOWNTREND" for t in tfs):
            summary = f"📉 ALL DOWNTREND | Spot {spot_vwap} VWAP"
        else:
            summary = f"Mixed trends | Spot {spot_vwap} VWAP ₹{vwap_val:.0f}"

        return TrendInfo(day_t, intra_t, t5, day_t, vwap_val, spot_vwap, all_agree, summary)


# ============================================================
#  PANDAS/NUMPY PRE-CALCULATOR
# ============================================================

class PriceActionCalculator:

    @staticmethod
    def calculate(snaps: List[MarketSnapshot],
                  df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                  df_30m: pd.DataFrame) -> PriceActionInsight:
        """
        v6.1: df_15m → df_30m for S/R detection + trend
        OI/Volume changes use DualCache snapshots (not candles).
        Price changes use snapshot spot prices.
        """
        if len(snaps) < 3:
            return PriceActionCalculator._empty(df_1m, df_5m, df_30m)

        prices = np.array([s.spot_price for s in snaps])
        curr   = prices[-1]

        def pct(ago: int) -> float:
            return ((curr - prices[-(ago+1)]) / prices[-(ago+1)] * 100
                    if len(prices) > ago else 0.0)

        p5m, p15m, p30m = pct(1), pct(3), pct(6)
        momentum = "BULLISH" if p5m > 0.2 else "BEARISH" if p5m < -0.2 else "NEUTRAL"

        vols        = np.array([s.total_ce_vol + s.total_pe_vol for s in snaps])
        vol_rolling = float(np.mean(vols[:-1])) if len(vols) > 1 else float(vols[-1])
        vol_spike   = (float(vols[-1]) / vol_rolling) if vol_rolling > 0 else 1.0

        ce_ois   = np.array([s.total_ce_oi for s in snaps])
        pe_ois   = np.array([s.total_pe_oi for s in snaps])
        oi_total = ce_ois + pe_ois

        oi_vol_corr = float(np.corrcoef(oi_total, vols)[0, 1]) if (
            len(oi_total) > 2 and np.std(oi_total) > 0 and np.std(vols) > 0) else 0.0
        price_oi_corr = float(np.corrcoef(prices, oi_total)[0, 1]) if (
            len(prices) > 2 and np.std(prices) > 0 and np.std(oi_total) > 0) else 0.0

        # S/R from 30min candles (v6.1: was 15min)
        supports, resistances = [], []
        if not df_30m.empty and len(df_30m) >= 5:
            df  = df_30m.tail(20)
            lws = df["low"].values
            hws = df["high"].values
            for i in range(1, len(lws) - 1):
                if lws[i] < lws[i-1] and lws[i] < lws[i+1]:
                    supports.append(float(lws[i]))
                if hws[i] > hws[i-1] and hws[i] > hws[i+1]:
                    resistances.append(float(hws[i]))
            supports    = sorted(supports,    key=lambda x: abs(x - curr))[:3]
            resistances = sorted(resistances, key=lambda x: abs(x - curr))[:3]

        trend = TrendCalculator.calculate(df_1m, df_5m, df_30m, curr)

        ts = 0.0
        if abs(p5m) >= 0.4:   ts += 3.0
        elif abs(p5m) >= 0.2: ts += 1.5
        if vol_spike >= 1.5:   ts += 3.0
        elif vol_spike >= 1.2: ts += 1.5
        oi_ch = ((oi_total[-1] - oi_total[0]) / oi_total[0] * 100) if oi_total[0] > 0 else 0
        if abs(oi_ch) >= 10:  ts += 4.0
        elif abs(oi_ch) >= 5: ts += 2.0

        price_bull = p5m > 0.2
        price_bear = p5m < -0.2
        oi_bull    = len(pe_ois) > 1 and pe_ois[-1] > pe_ois[0]
        oi_bear    = len(ce_ois) > 1 and ce_ois[-1] > ce_ois[0]
        vol_ok     = vol_spike >= 1.2
        triple     = (price_bull and oi_bull and vol_ok) or (price_bear and oi_bear and vol_ok)

        return PriceActionInsight(
            price_change_5m=round(p5m, 3),   price_change_15m=round(p15m, 3),
            price_change_30m=round(p30m, 3), price_momentum=momentum,
            vol_rolling_avg=round(vol_rolling, 0), vol_spike_ratio=round(vol_spike, 2),
            oi_vol_corr=round(oi_vol_corr, 2), price_oi_corr=round(price_oi_corr, 2),
            support_levels=supports, resistance_levels=resistances,
            trend_strength=round(min(10.0, ts), 1), triple_confirmed=triple, trend=trend
        )

    @staticmethod
    def _empty(df_1m, df_5m, df_30m) -> PriceActionInsight:
        t = TrendInfo("SIDEWAYS","SIDEWAYS","SIDEWAYS","SIDEWAYS",0.0,"ABOVE",False,"Building...")
        if not df_5m.empty:
            try:
                t = TrendCalculator.calculate(df_1m, df_5m, df_30m,
                                              float(df_5m.iloc[-1]["close"]))
            except Exception:
                pass
        return PriceActionInsight(0, 0, 0, "NEUTRAL", 0, 1.0, 0, 0, [], [], 0, False, t)


# ============================================================
#  MTF OI ANALYZER
# ============================================================

class MTFAnalyzer:
    """
    OI/Volume MTF Analysis uses DualCache snapshots exclusively.
    5min ago  = cache[-1]
    15min ago = cache[-3]
    30min ago = cache[-6]
    This is correct and does NOT depend on candle API.
    v6.1: MTF_OVERALL_THRESHOLD = 8 (was 12, adjusted for 9 strikes with ATM_RANGE=4)
    """

    def __init__(self, cache: DualCache):
        self.cache = cache

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    @staticmethod
    def _action(ch: float) -> str:
        if ch >= 8:  return "BUILDING"
        if ch <= -8: return "UNWINDING"
        return "NEUTRAL"

    @staticmethod
    def _tf_signal(ce: float, pe: float, cv: float, pv: float) -> str:
        if pe >= MIN_OI_CHANGE and pv >= MIN_VOLUME_CHG: return "BULLISH"
        if ce >= MIN_OI_CHANGE and cv >= MIN_VOLUME_CHG: return "BEARISH"
        if pe <= -MIN_OI_CHANGE: return "BEARISH"
        if ce <= -MIN_OI_CHANGE: return "BULLISH"
        return "NEUTRAL"

    @staticmethod
    def _vol_confirm(oi_ch: float, vol_ch: float) -> Tuple[bool, str]:
        if oi_ch > 8  and vol_ch > MIN_VOLUME_CHG: return True,  "STRONG"
        if oi_ch > 4  and vol_ch > 10:             return True,  "MODERATE"
        if abs(oi_ch) < 4 and abs(vol_ch) < 4:    return True,  "WEAK"
        return False, "WEAK"

    def _strength(self, ce30: float, pe30: float, cv30: float, pv30: float,
                  weight: float, mtf: bool) -> Tuple[float, float]:
        bull = bear = 0.0
        boost = 1.5 if mtf else 0.8
        if   pe30 >= STRONG_OI_CHANGE: bull = 9.0
        elif pe30 >= MIN_OI_CHANGE:    bull = 7.0
        elif pe30 >= 5:                bull = 4.0
        if   ce30 >= STRONG_OI_CHANGE: bear = 9.0
        elif ce30 >= MIN_OI_CHANGE:    bear = 7.0
        elif ce30 >= 5:                bear = 4.0
        if pe30 <= -STRONG_OI_CHANGE:  bear = max(bear, 8.0)
        elif pe30 <= -MIN_OI_CHANGE:   bear = max(bear, 6.0)
        if ce30 <= -STRONG_OI_CHANGE:  bull = max(bull, 8.0)
        elif ce30 <= -MIN_OI_CHANGE:   bull = max(bull, 6.0)
        return min(10.0, bull * weight * boost), min(10.0, bear * weight * boost)

    async def analyze(self, current: MarketSnapshot) -> Dict:
        # DualCache snapshot comparisons
        s5  = await self.cache.get_5min_ago(1)   # 5min ago
        s15 = await self.cache.get_5min_ago(3)   # 15min ago
        s30 = await self.cache.get_5min_ago(6)   # 30min ago

        if not s5:
            return {"available": False, "reason": "Building cache (need >= 5 min)..."}

        analyses: List[StrikeAnalysis] = []

        for strike in sorted(current.strikes_oi.keys()):
            c   = current.strikes_oi[strike]
            p5  = s5.strikes_oi.get(strike)
            p15 = s15.strikes_oi.get(strike) if s15 else None
            p30 = s30.strikes_oi.get(strike) if s30 else None

            # 5-min OI/Volume change
            ce5  = self._pct(c.ce_oi,     p5.ce_oi     if p5 else 0)
            pe5  = self._pct(c.pe_oi,     p5.pe_oi     if p5 else 0)
            cv5  = self._pct(c.ce_volume, p5.ce_volume  if p5 else 0)
            pv5  = self._pct(c.pe_volume, p5.pe_volume  if p5 else 0)

            # 15-min OI/Volume change
            ce15 = self._pct(c.ce_oi,     p15.ce_oi     if p15 else 0)
            pe15 = self._pct(c.pe_oi,     p15.pe_oi     if p15 else 0)
            cv15 = self._pct(c.ce_volume, p15.ce_volume  if p15 else 0)
            pv15 = self._pct(c.pe_volume, p15.pe_volume  if p15 else 0)
            pc15 = self._pct(c.pcr,       p15.pcr        if p15 else 0)

            # 30-min OI/Volume change
            ce30 = self._pct(c.ce_oi,     p30.ce_oi     if p30 else 0)
            pe30 = self._pct(c.pe_oi,     p30.pe_oi     if p30 else 0)
            cv30 = self._pct(c.ce_volume, p30.ce_volume  if p30 else 0)
            pv30 = self._pct(c.pe_volume, p30.pe_volume  if p30 else 0)

            is_atm = (strike == current.atm_strike)
            dist   = abs(strike - current.atm_strike)
            weight = ATM_WEIGHT if is_atm else (
                NEAR_ATM_WEIGHT if dist <= STRIKE_INTERVAL else FAR_WEIGHT
            )

            tf5  = self._tf_signal(ce5,  pe5,  cv5,  pv5)
            tf15 = self._tf_signal(ce15, pe15, cv15, pv15)
            tf30 = self._tf_signal(ce30, pe30, cv30, pv30)
            tfs  = [tf5, tf15, tf30]
            mtf  = len(set(tfs)) == 1 and tfs[0] != "NEUTRAL"

            vc, vs = self._vol_confirm((ce30 + pe30) / 2, (cv30 + pv30) / 2)
            bull, bear = self._strength(ce30, pe30, cv30, pv30, weight, mtf)
            if mtf:
                bull = min(10.0, bull * 1.3)
                bear = min(10.0, bear * 1.3)

            if   bull >= 7 and bull > bear: rec, conf = "STRONG_CALL", bull
            elif bear >= 7 and bear > bull: rec, conf = "STRONG_PUT",  bear
            else:                           rec, conf = "WAIT", max(bull, bear)

            analyses.append(StrikeAnalysis(
                strike=strike, is_atm=is_atm, distance_atm=dist, weight=weight,
                ce_oi=c.ce_oi, pe_oi=c.pe_oi,
                ce_volume=c.ce_volume, pe_volume=c.pe_volume,
                ce_ltp=c.ce_ltp, pe_ltp=c.pe_ltp,
                ce_iv=c.ce_iv, pe_iv=c.pe_iv,
                ce_oi_5=ce5, pe_oi_5=pe5, ce_vol_5=cv5, pe_vol_5=pv5,
                ce_oi_15=ce15, pe_oi_15=pe15, ce_vol_15=cv15, pe_vol_15=pv15,
                pcr_ch_15=pc15,
                ce_oi_30=ce30, pe_oi_30=pe30, ce_vol_30=cv30, pe_vol_30=pv30,
                pcr=c.pcr, ce_action=self._action(ce15), pe_action=self._action(pe15),
                tf5_signal=tf5, tf15_signal=tf15, tf30_signal=tf30,
                mtf_confirmed=mtf, vol_confirms=vc, vol_strength=vs,
                is_support=False, is_resistance=False,
                bull_strength=bull, bear_strength=bear,
                recommendation=rec, confidence=conf
            ))

        sr = self._find_sr(current, analyses)
        for sa in analyses:
            sa.is_support    = (sa.strike == sr.support_strike)
            sa.is_resistance = (sa.strike == sr.resistance_strike)

        prev_pcr  = s30.overall_pcr if s30 else current.overall_pcr
        pcr_trend = "BULLISH" if current.overall_pcr > prev_pcr else "BEARISH"
        pcr_ch    = self._pct(current.overall_pcr, prev_pcr)
        tb        = sum(sa.bull_strength for sa in analyses)
        tr_b      = sum(sa.bear_strength for sa in analyses)

        # v6.1: threshold = 8 (was 12), adjusted for ATM_RANGE=4 (9 strikes)
        overall = ("BULLISH" if tb > tr_b and tb >= MTF_OVERALL_THRESHOLD
                   else "BEARISH" if tr_b > tb and tr_b >= MTF_OVERALL_THRESHOLD
                   else "NEUTRAL")

        return {
            "available":       True,
            "strike_analyses": analyses,
            "sr":              sr,
            "overall":         overall,
            "total_bull":      tb,
            "total_bear":      tr_b,
            "overall_pcr":     current.overall_pcr,
            "pcr_trend":       pcr_trend,
            "pcr_ch_pct":      pcr_ch,
            "has_strong":      any(
                sa.mtf_confirmed and sa.confidence >= MIN_CONFIDENCE
                for sa in analyses
            )
        }

    def _find_sr(self, current: MarketSnapshot,
                 analyses: List[StrikeAnalysis]) -> SupportResistance:
        mp = max(analyses, key=lambda x: x.pe_oi, default=None)
        mc = max(analyses, key=lambda x: x.ce_oi, default=None)
        sup = mp.strike if mp else current.atm_strike
        res = mc.strike if mc else current.atm_strike
        return SupportResistance(
            support_strike=sup, support_put_oi=mp.pe_oi if mp else 0,
            resistance_strike=res, resistance_call_oi=mc.ce_oi if mc else 0,
            near_support=abs(current.spot_price - sup) <= ATM_PROX_PTS,
            near_resistance=abs(current.spot_price - res) <= ATM_PROX_PTS
        )


# ============================================================
#  PHASE DETECTOR
# ============================================================

class PhaseDetector:

    COOLDOWN_P1 = 15 * 60
    COOLDOWN_P2 = 10 * 60
    COOLDOWN_P3 =  5 * 60

    def __init__(self):
        self._last: Dict[str, float] = {}
        self._p1_at   = 0.0
        self._p2_at   = 0.0
        self._p1_side = ""

    def _can(self, k: str, cd: int) -> bool:
        return (time_module.time() - self._last.get(k, 0)) >= cd

    def _mark(self, k: str):
        self._last[k] = time_module.time()

    @staticmethod
    def _pct(c: float, p: float) -> float:
        return ((c - p) / p * 100) if p > 0 else 0.0

    async def detect(self, curr: MarketSnapshot,
                     prev: Optional[MarketSnapshot],
                     pa: PriceActionInsight) -> List[PhaseSignal]:
        signals = []
        if not prev:
            return signals

        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap:
            return signals

        ce_oi_ch  = self._pct(ac.ce_oi,    ap.ce_oi)
        pe_oi_ch  = self._pct(ac.pe_oi,    ap.pe_oi)
        ce_vol_ch = self._pct(ac.ce_volume, ap.ce_volume)
        pe_vol_ch = self._pct(ac.pe_volume, ap.pe_volume)

        call_bld = ce_oi_ch >= PHASE1_OI_BUILD_PCT   # v6.1: 6.0% threshold
        put_bld  = pe_oi_ch >= PHASE1_OI_BUILD_PCT
        if not (call_bld or put_bld):
            return signals

        dom   = "PUT" if (put_bld and pe_oi_ch >= ce_oi_ch) else "CALL"
        oi_ch = pe_oi_ch if dom == "PUT" else ce_oi_ch
        v_ch  = pe_vol_ch if dom == "PUT" else ce_vol_ch
        dirn  = "BULLISH" if dom == "PUT" else "BEARISH"
        now   = time_module.time()

        # Phase 1 — Smart Money OI buildup (quiet volume)
        if (oi_ch >= PHASE1_OI_BUILD_PCT and abs(v_ch) < PHASE1_VOL_MAX_PCT
                and self._can("P1", self.COOLDOWN_P1)):
            self._p1_at = now; self._p1_side = dom; self._mark("P1")
            signals.append(PhaseSignal(
                phase=1, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, oi_ch / 1.5),
                message=(
                    f"⚡ PHASE 1 - SMART MONEY POSITIONING\n\n"
                    f"NIFTY: {curr.spot_price:,.2f}\nATM: {curr.atm_strike}\n\n"
                    f"{'PUT' if dom=='PUT' else 'CALL'} OI: {oi_ch:+.1f}% (quiet buildup)\n"
                    f"Volume: {v_ch:+.1f}% (still low — not a trap)\n\n"
                    f"Signal: {dirn}\nTrend: {pa.trend.summary}\n\n"
                    f"⏳ Wait for Phase 2 before entering!\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}"
                )
            ))

        # Phase 2 — Volume spike after Phase 1
        p1_recent = (now - self._p1_at) < (25 * 60)
        if (pa.vol_spike_ratio >= (1 + PHASE2_VOL_SPIKE_PCT / 100)  # v6.1: 15%
                and oi_ch >= PHASE2_OI_MIN_PCT
                and p1_recent and self._p1_side == dom
                and self._can("P2", self.COOLDOWN_P2)):
            self._p2_at = now; self._mark("P2")
            signals.append(PhaseSignal(
                phase=2, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, pa.vol_spike_ratio * 3),
                message=(
                    f"🔥 PHASE 2 - VOLUME SPIKE! MOVE IMMINENT\n\n"
                    f"NIFTY: {curr.spot_price:,.2f}\nATM: {curr.atm_strike}\n\n"
                    f"Volume: {pa.vol_spike_ratio:.1f}x average\n"
                    f"OI: {oi_ch:+.1f}% (building since Phase 1)\n\n"
                    f"Signal: {dirn}\nTrend: {pa.trend.summary}\n\n"
                    f"👆 Finger on trigger! Wait for price confirmation!\n"
                    f"{'BUY CALL' if dirn=='BULLISH' else 'BUY PUT'} near {curr.atm_strike}\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}"
                )
            ))

        # Phase 3 — Price confirmation (triple: OI + Volume + Price)
        p2_recent  = (now - self._p2_at) < (15 * 60)
        price_ok   = ((dirn == "BULLISH" and pa.price_change_5m >= PHASE3_PRICE_MOVE_PCT) or
                      (dirn == "BEARISH" and pa.price_change_5m <= -PHASE3_PRICE_MOVE_PCT))
        if p2_recent and price_ok and pa.triple_confirmed and self._can("P3", self.COOLDOWN_P3):
            self._mark("P3")
            signals.append(PhaseSignal(
                phase=3, dominant_side=dom, direction=dirn,
                oi_change_pct=oi_ch, vol_change_pct=v_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, 7 + pa.trend_strength / 3),
                message=(
                    f"🚀 PHASE 3 - CONFIRMED! EXECUTE NOW!\n\n"
                    f"NIFTY: {curr.spot_price:,.2f} ({pa.price_change_5m:+.2f}%/5min)\n"
                    f"ATM: {curr.atm_strike}\n\n"
                    f"Signal: {'BUY_CALL' if dirn=='BULLISH' else 'BUY_PUT'}\n"
                    f"✅ Triple Confirmed: OI + Volume + Price ALL agree!\n\n"
                    f"Trend: {pa.trend.summary}\n"
                    f"VWAP: {pa.trend.vwap:.0f} | Spot {pa.trend.spot_vs_vwap} VWAP\n\n"
                    f"OI: {oi_ch:+.1f}% | Vol: {pa.vol_spike_ratio:.1f}x | "
                    f"Price: {pa.price_change_5m:+.2f}%\n"
                    f"Trend Strength: {pa.trend_strength:.1f}/10\n\n"
                    f"{datetime.now(IST).strftime('%H:%M IST')}\n"
                    f"⏳ Sending to AI for confirmation..."
                )
            ))

        return signals


# ============================================================
#  STANDALONE ALERT CHECKER
# ============================================================

class AlertChecker:
    COOLDOWN = 30 * 60

    def __init__(self, cache: DualCache, alerter):
        self.cache   = cache
        self.alerter = alerter
        self._last: Dict[str, float] = {}

    def _can(self, k: str) -> bool:
        return (time_module.time() - self._last.get(k, 0)) >= self.COOLDOWN

    def _mark(self, k: str):
        self._last[k] = time_module.time()

    async def check_all(self, curr: MarketSnapshot):
        prev = await self.cache.get_5min_ago(6)
        if not prev:
            return
        await self._oi(curr, prev)
        await self._vol(curr, prev)
        await self._pcr(curr, prev)
        await self._prox(curr)

    async def _oi(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("OI"): return
        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap or ap.ce_oi == 0 or ap.pe_oi == 0: return
        ce = (ac.ce_oi - ap.ce_oi) / ap.ce_oi * 100
        pe = (ac.pe_oi - ap.pe_oi) / ap.pe_oi * 100
        if abs(ce) < OI_ALERT_PCT and abs(pe) < OI_ALERT_PCT: return
        txt = (
            f"📊 OI CHANGE ALERT (30-min)\n\n"
            f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n\n"
            f"CALL OI: {ce:+.1f}% {'BUILDING' if ce > 0 else 'UNWINDING'}\n"
            f"PUT  OI: {pe:+.1f}% {'BUILDING' if pe > 0 else 'UNWINDING'}\n\n"
            f"PCR: {curr.overall_pcr:.2f}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("OI")

    async def _vol(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("VOL"): return
        ac = curr.strikes_oi.get(curr.atm_strike)
        ap = prev.strikes_oi.get(curr.atm_strike)
        if not ac or not ap or ap.ce_volume == 0 or ap.pe_volume == 0: return
        cv = (ac.ce_volume - ap.ce_volume) / ap.ce_volume * 100
        pv = (ac.pe_volume - ap.pe_volume) / ap.pe_volume * 100
        if max(cv, pv) < VOL_SPIKE_PCT: return
        dom = "CALL" if cv >= pv else "PUT"
        txt = (
            f"⚡ VOLUME SPIKE ALERT\n\n"
            f"NIFTY: {curr.spot_price:,.2f} | ATM: {curr.atm_strike}\n\n"
            f"CALL Vol: {cv:+.1f}% | PUT Vol: {pv:+.1f}%\n"
            f"Dominant: {dom} → {'BEARISH' if dom=='CALL' else 'BULLISH'}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("VOL")

    async def _pcr(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("PCR") or prev.overall_pcr <= 0: return
        ch = (curr.overall_pcr - prev.overall_pcr) / prev.overall_pcr * 100
        if abs(ch) < PCR_ALERT_PCT: return
        txt = (
            f"📈 PCR CHANGE ALERT\n\n"
            f"NIFTY: {curr.spot_price:,.2f}\n\n"
            f"PCR: {prev.overall_pcr:.2f} → {curr.overall_pcr:.2f} ({ch:+.1f}%)\n"
            f"{'Bulls gaining' if ch > 0 else 'Bears gaining'}\n"
            f"{datetime.now(IST).strftime('%H:%M IST')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("PCR")

    async def _prox(self, curr: MarketSnapshot):
        if not self._can("PROX"): return
        max_pe = max(curr.strikes_oi.items(), key=lambda x: x[1].pe_oi, default=None)
        max_ce = max(curr.strikes_oi.items(), key=lambda x: x[1].ce_oi, default=None)
        for level, item, kind in [("SUPPORT", max_pe, "PUT"), ("RESISTANCE", max_ce, "CALL")]:
            if not item: continue
            strike, oi_s = item
            dist = abs(curr.spot_price - strike)
            if dist > ATM_PROX_PTS: continue
            oi_v = oi_s.pe_oi if kind == "PUT" else oi_s.ce_oi
            txt  = (
                f"🎯 PRICE NEAR {level}\n\n"
                f"NIFTY: {curr.spot_price:,.2f}\n"
                f"{level}: {strike} (OI: {oi_v:,.0f})\n"
                f"Distance: {dist:.0f} pts\n"
                f"{datetime.now(IST).strftime('%H:%M IST')}"
            )
            await self.alerter.send_raw(txt)
            self._mark("PROX")
            break


# ============================================================
#  CANDLESTICK PATTERN DETECTOR
# ============================================================

class PatternDetector:

    @staticmethod
    def detect(df: pd.DataFrame) -> List[Dict]:
        """Detects patterns from 5min resampled candles"""
        pats = []
        if df.empty or len(df) < 2: return pats
        for i in range(1, len(df)):
            c, p = df.iloc[i], df.iloc[i-1]
            bc  = abs(c.close - c.open)
            bp  = abs(p.close - p.open)
            rng = c.high - c.low
            if rng == 0: continue

            if (c.close > c.open and p.close < p.open
                    and c.open <= p.close and c.close >= p.open and bc > bp * 1.2):
                pats.append({"time": c.name, "pattern": "BULLISH_ENGULFING",
                             "type": "BULLISH", "strength": 8, "price": c.close})
            elif (c.close < c.open and p.close > p.open
                    and c.open >= p.close and c.close <= p.open and bc > bp * 1.2):
                pats.append({"time": c.name, "pattern": "BEARISH_ENGULFING",
                             "type": "BEARISH", "strength": 8, "price": c.close})
            else:
                lw = min(c.open, c.close) - c.low
                hw = c.high - max(c.open, c.close)
                if lw > bc * 2 and hw < bc * 0.3 and bc < rng * 0.35:
                    pats.append({"time": c.name, "pattern": "HAMMER",
                                 "type": "BULLISH", "strength": 7, "price": c.close})
                elif hw > bc * 2 and lw < bc * 0.3 and bc < rng * 0.35:
                    pats.append({"time": c.name, "pattern": "SHOOTING_STAR",
                                 "type": "BEARISH", "strength": 7, "price": c.close})
                elif bc < rng * 0.1:
                    pats.append({"time": c.name, "pattern": "DOJI",
                                 "type": "NEUTRAL", "strength": 4, "price": c.close})
        return pats[-5:]

    @staticmethod
    def sr(df: pd.DataFrame) -> Tuple[float, float]:
        """S/R from 30min candles (v6.1: was 15min)"""
        if df.empty or len(df) < 5: return 0.0, 0.0
        d = df.tail(20)
        return float(d.low.min()), float(d.high.max())


# ============================================================
#  DEEPSEEK PROMPT BUILDER
# ============================================================

class PromptBuilder:

    @staticmethod
    def _candles(df: pd.DataFrame, label: str) -> str:
        if df.empty: return f"{label}: no data\n"
        out = f"\n{label} (TIME|O|H|L|C|VOL|DIR):\n"
        for ts, row in df.tail(CANDLE_COUNT).iterrows():
            t = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[:5]
            d = "▲" if row.close > row.open else "▼" if row.close < row.open else "-"
            out += (f"{t}|{row.open:.0f}|{row.high:.0f}|{row.low:.0f}|"
                    f"{row.close:.0f}|{int(row.volume)}|{d}\n")
        return out

    @staticmethod
    def build(spot: float, atm: int, expiry: str, oi: Dict,
              df_5m: pd.DataFrame, df_30m: pd.DataFrame,
              patterns: List[Dict], p_sup: float, p_res: float,
              pa: PriceActionInsight,
              phase: Optional[PhaseSignal] = None) -> str:
        """
        v6.1: df_15m replaced with df_30m throughout.
        OI analysis uses DualCache 5/15/30min snapshot comparisons.
        Price action uses Pandas/Numpy pre-calculations.
        """
        now = datetime.now(IST).strftime("%H:%M IST")
        sr  = oi["sr"]
        pcr = oi["overall_pcr"]
        t   = pa.trend

        p  = "You are an expert NIFTY 50 options trader. Analyze all data carefully and give a precise trade signal.\n\n"
        p += f"=== MARKET | {now} | Expiry:{expiry} | Lot:{LOT_SIZE} | Strike interval:₹{STRIKE_INTERVAL} ===\n"
        p += f"NIFTY Spot: ₹{spot:,.2f} | ATM: {atm} | PCR:{pcr:.2f}({oi['pcr_trend']}) Δ30m:{oi['pcr_ch_pct']:+.1f}%\n"
        p += f"OI Support: {sr.support_strike} | OI Resistance: {sr.resistance_strike}\n"
        if sr.near_support:    p += "⚠️ PRICE NEAR OI SUPPORT!\n"
        if sr.near_resistance: p += "⚠️ PRICE NEAR OI RESISTANCE!\n"

        p += f"\n=== MULTI-TIMEFRAME TREND ===\n"
        p += f"Day(30min):{t.day_trend} | Intraday(5min):{t.intraday_trend} | Short(recent 5min):{t.trend_5min}\n"
        p += f"VWAP: ₹{t.vwap:.0f} | Spot {t.spot_vs_vwap} VWAP | All TFs agree: {'YES ✅' if t.all_agree else 'NO ❌'}\n"
        p += f"Summary: {t.summary}\n"

        p += f"\n=== PRICE ACTION (Pandas/Numpy pre-calculated) ===\n"
        p += f"Price change: 5m={pa.price_change_5m:+.2f}% | 15m={pa.price_change_15m:+.2f}% | 30m={pa.price_change_30m:+.2f}%\n"
        p += f"Momentum: {pa.price_momentum} | Volume spike: {pa.vol_spike_ratio:.2f}x rolling avg\n"
        p += f"OI-Volume correlation: {pa.oi_vol_corr:.2f} | Price-OI correlation: {pa.price_oi_corr:.2f}\n"
        p += f"Trend strength: {pa.trend_strength:.1f}/10 | Triple confirmed (OI+Vol+Price): {'YES ✅' if pa.triple_confirmed else 'NO'}\n"
        if pa.support_levels:
            p += f"Price support levels: {', '.join(f'₹{s:.0f}' for s in pa.support_levels)}\n"
        if pa.resistance_levels:
            p += f"Price resistance levels: {', '.join(f'₹{r:.0f}' for r in pa.resistance_levels)}\n"

        if phase:
            p += f"\n=== ⚡ PHASE {phase.phase} SIGNAL DETECTED ===\n"
            p += f"Side:{phase.dominant_side} | Direction:{phase.direction}\n"
            p += f"OI change:{phase.oi_change_pct:+.1f}% | Vol change:{phase.vol_change_pct:+.1f}% | Price:{phase.price_change_pct:+.2f}%\n"

        p += f"\n=== OI MULTI-TIMEFRAME ANALYSIS (DualCache snapshots) ===\n"
        p += f"Note: 5min/15min/30min OI changes are from live option chain snapshots, NOT candles.\n"
        p += "STRIKE | W | CE_OI(5%/15%/30%) | CE_VOL15% | CE_ACT | PE_OI(5%/15%/30%) | PE_VOL15% | PE_ACT | PCR | TF5/TF15/TF30 | MTF | CE_IV | PE_IV | Bull | Bear | Conf\n"
        for sa in oi["strike_analyses"]:
            tag = "ATM" if sa.is_atm else ("SUP" if sa.is_support else ("RES" if sa.is_resistance else "   "))
            p += (
                f"{sa.strike}({tag}) W{sa.weight:.0f} | "
                f"CE:{sa.ce_oi_5:+.0f}%/{sa.ce_oi_15:+.0f}%/{sa.ce_oi_30:+.0f}% V:{sa.ce_vol_15:+.0f}%({sa.ce_action}) | "
                f"PE:{sa.pe_oi_5:+.0f}%/{sa.pe_oi_15:+.0f}%/{sa.pe_oi_30:+.0f}% V:{sa.pe_vol_15:+.0f}%({sa.pe_action}) | "
                f"PCR:{sa.pcr:.2f} | {sa.tf5_signal[:3]}/{sa.tf15_signal[:3]}/{sa.tf30_signal[:3]} | "
                f"MTF:{'YES' if sa.mtf_confirmed else 'NO'} | "
                f"IV:{sa.ce_iv:.1f}%/{sa.pe_iv:.1f}% | "
                f"B:{sa.bull_strength:.0f} Br:{sa.bear_strength:.0f} C:{sa.confidence:.0f}\n"
            )

        # 5min resampled candles for short-term price action
        p += PromptBuilder._candles(df_5m,  "5MIN CANDLES (1min resampled to 5min)")
        # 30min candles for day context
        p += PromptBuilder._candles(df_30m, "30MIN CANDLES (Upstox direct)")

        if patterns:
            p += "\n=== CANDLESTICK PATTERNS (on 5min candles) ===\n"
            for pat in patterns:
                ts = pat["time"].strftime("%H:%M") if hasattr(pat["time"], "strftime") else str(pat["time"])[:5]
                p += f"{ts} | {pat['pattern']} | {pat['type']} | Strength:{pat['strength']}/10 | @₹{pat['price']:.0f}\n"

        if p_sup or p_res:
            p += f"\nCandle S/R (30min): Support=₹{p_sup:.0f} | Resistance=₹{p_res:.0f}\n"

        p += f"""
=== NIFTY 50 OPTIONS TRADING RULES ===
OI INTERPRETATION:
  CALL OI↑ + Volume↑ = Resistance forming = BEARISH → BUY PUT
  PUT OI↑  + Volume↑ = Support forming   = BULLISH → BUY CALL
  OI↑ but Volume flat = TRAP — do NOT trade!
  MTF confirmed (TF5+TF15+TF30 all agree) = HIGHEST confidence signal

TREND RULES:
  Always trade WITH the day trend (30min)
  All TFs agree + Spot above/below VWAP = HIGH confidence entry
  Conflicting TFs = WAIT, do not force entry

PRICE ACTION:
  Triple confirmed (OI+Volume+Price all agree) = strongest signal
  Spot above VWAP = bullish bias | Spot below VWAP = bearish bias
  Price-OI correlation > 0.7 = strong confirmation

IV RULES:
  IV > 20% = expensive premium, prefer ATM only
  IV < 12% = cheap, can consider ATM or 1 strike ITM
  High IV difference (CE vs PE) = directional bias

PCR:
  PCR > {PCR_BULL} = bullish market bias
  PCR < {PCR_BEAR} = bearish market bias

NIFTY SPECIFICS:
  Lot size={LOT_SIZE} | Strike interval=₹{STRIKE_INTERVAL}
  SL = 1 strike away | Target = 2 strikes away
  Best trading window: 9:30 AM - 3:00 PM IST
  Entry: ONLY when MTF + Volume + Price + Trend ALL confirm

RESPOND ONLY WITH VALID JSON (no markdown, no explanation):
{{
  "signal": "BUY_CALL" | "BUY_PUT" | "WAIT",
  "primary_strike": {atm},
  "confidence": 0-10,
  "stop_loss_strike": 0,
  "target_strike": 0,
  "trend_analysis": {{
    "day": "",
    "intraday": "",
    "all_agree": true,
    "vwap_confirms": true,
    "note": ""
  }},
  "mtf": {{
    "tf5": "",
    "tf15": "",
    "tf30": "",
    "confirmed": true
  }},
  "price_action": {{
    "momentum": "",
    "triple_confirmed": true,
    "confirms_signal": true
  }},
  "candle_pattern": {{
    "pattern": "",
    "type": "",
    "confirms_signal": true,
    "near_sr": true
  }},
  "atm": {{
    "ce_action": "",
    "pe_action": "",
    "vol_confirms": true,
    "strength": ""
  }},
  "iv_note": {{
    "ce_iv": 0,
    "pe_iv": 0,
    "note": ""
  }},
  "pcr": {{
    "value": {pcr:.2f},
    "trend": "{oi['pcr_trend']}",
    "supports": true
  }},
  "volume": {{
    "ok": true,
    "spike_ratio": {pa.vol_spike_ratio:.2f},
    "trap_warning": ""
  }},
  "entry": {{
    "now": true,
    "reason": "",
    "wait_for": ""
  }},
  "rr": {{
    "sl_pts": 0,
    "tgt_pts": 0,
    "ratio": 0
  }},
  "levels": {{
    "oi_support": {sr.support_strike},
    "oi_resistance": {sr.resistance_strike},
    "candle_sup": {p_sup:.0f},
    "candle_res": {p_res:.0f}
  }}
}}"""
        return p


# ============================================================
#  DEEPSEEK CLIENT
# ============================================================

class DeepSeekClient:
    URL   = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self, key: str):
        self.key = key

    async def analyze(self, prompt: str) -> Optional[Dict]:
        hdrs = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model":       self.MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens":  1500
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            ) as sess:
                async with sess.post(self.URL, headers=hdrs, json=payload) as r:
                    if r.status != 200:
                        logger.error(f"DeepSeek {r.status}")
                        return None
                    data    = await r.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    # Strip markdown code blocks if present
                    for f in ("```json", "```"):
                        content = content.replace(f, "")
                    return json.loads(content.strip())
        except asyncio.TimeoutError:
            logger.error("❌ DeepSeek timeout")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek: {e}")
            return None


# ============================================================
#  TELEGRAM ALERTER
# ============================================================

class TelegramAlerter:

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.session = None

    async def _sess(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def send_raw(self, text: str):
        await self._sess()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with self.session.post(url, json={
                "chat_id": self.chat_id,
                "text":    text,
                "parse_mode": "HTML"
            }) as r:
                if r.status != 200:
                    txt = await r.text()
                    logger.error(f"Telegram {r.status}: {txt[:100]}")
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

    async def send_signal(self, sig: Dict, snap: MarketSnapshot,
                          oi: Dict, pa: PriceActionInsight):
        mtf  = sig.get("mtf",            {})
        atma = sig.get("atm",            {})
        pcra = sig.get("pcr",            {})
        vol  = sig.get("volume",         {})
        ent  = sig.get("entry",          {})
        rr   = sig.get("rr",             {})
        prce = sig.get("price_action",   {})
        cndl = sig.get("candle_pattern", {})
        trnd = sig.get("trend_analysis", {})
        iv   = sig.get("iv_note",        {})
        st   = sig.get("signal", "WAIT")
        opt  = "CE" if "CALL" in st else "PE" if "PUT" in st else ""
        t    = pa.trend

        conf = sig.get("confidence", 0)
        conf_bar = "🟢" * min(conf, 10) + "⚪" * (10 - min(conf, 10))

        msg = (
            f"🚀 NIFTY OPTIONS SIGNAL v6.1 PRO\n"
            f"📅 {datetime.now(IST).strftime('%d-%b-%Y %H:%M IST')}\n\n"
            f"💰 NIFTY: ₹{snap.spot_price:,.2f}\n"
            f"📊 Signal: <b>{st}</b>\n"
            f"⭐ Confidence: {conf}/10  {conf_bar}\n"
            f"📅 Expiry: {snap.expiry} | Lot: {LOT_SIZE}\n\n"
            f"━━━ TRADE SETUP ━━━\n"
            f"Entry: <b>{sig.get('primary_strike', 0)} {opt}</b>\n"
            f"SL: {sig.get('stop_loss_strike', 0)} {opt}\n"
            f"Target: {sig.get('target_strike', 0)} {opt}\n"
            f"RR Ratio: {rr.get('ratio', 'N/A')} "
            f"(SL:{rr.get('sl_pts', 0)}pts → Tgt:{rr.get('tgt_pts', 0)}pts)\n\n"
            f"━━━ TREND ━━━\n"
            f"Day(30m):{t.day_trend} | Intraday:{t.intraday_trend} | 5min:{t.trend_5min}\n"
            f"VWAP: ₹{t.vwap:.0f} | Spot {t.spot_vs_vwap} VWAP\n"
            f"All TFs agree: {'✅ YES' if trnd.get('all_agree') else '❌ NO'} | "
            f"VWAP confirms: {'✅' if trnd.get('vwap_confirms') else '❌'}\n\n"
            f"━━━ MTF OI ANALYSIS ━━━\n"
            f"TF5:{mtf.get('tf5','N/A')} | TF15:{mtf.get('tf15','N/A')} | TF30:{mtf.get('tf30','N/A')}\n"
            f"MTF Confirmed: {'✅ HIGH CONFIDENCE' if mtf.get('confirmed') else '❌ Single TF only'}\n\n"
            f"━━━ OI DATA ━━━\n"
            f"ATM CE: {atma.get('ce_action','N/A')} | ATM PE: {atma.get('pe_action','N/A')}\n"
            f"Volume confirms: {'✅ YES' if atma.get('vol_confirms') else '❌ NO — POSSIBLE TRAP'}\n\n"
            f"━━━ PRICE ACTION ━━━\n"
            f"5m:{pa.price_change_5m:+.2f}% | 15m:{pa.price_change_15m:+.2f}% | "
            f"30m:{pa.price_change_30m:+.2f}%\n"
            f"Triple confirmed: {'✅' if pa.triple_confirmed else '❌'} | "
            f"Vol: {pa.vol_spike_ratio:.1f}x avg\n\n"
            f"━━━ PATTERNS & LEVELS ━━━\n"
            f"Pattern: {cndl.get('pattern','None')} ({cndl.get('type','')})\n"
            f"Near S/R: {'✅ YES' if cndl.get('near_sr') else 'NO'} | "
            f"Confirms: {'✅' if cndl.get('confirms_signal') else '❌'}\n\n"
            f"━━━ PCR & IV ━━━\n"
            f"PCR: {pcra.get('value','N/A')} ({pcra.get('trend','N/A')}) | "
            f"Supports signal: {'✅' if pcra.get('supports') else '❌'}\n"
            f"IV: CE={iv.get('ce_iv',0):.1f}% PE={iv.get('pe_iv',0):.1f}%\n"
            f"{iv.get('note','')}\n\n"
            f"━━━ ENTRY DECISION ━━━\n"
            f"Enter now: {'✅ YES' if ent.get('now') else '⏳ WAIT'}\n"
            f"{ent.get('reason','')}\n"
            f"{('Wait for: ' + ent.get('wait_for','')) if not ent.get('now') and ent.get('wait_for') else ''}\n\n"
            f"DeepSeek V3 | NIFTY v6.1 Pro | Upstox V2"
        )
        if vol.get("trap_warning"):
            msg += f"\n\n⚠️ WARNING: {vol['trap_warning']}"
        await self.send_raw(msg.strip())


# ============================================================
#  MAIN BOT
# ============================================================

class NiftyOptionsBot:

    def __init__(self):
        self.upstox  = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.cache   = DualCache()
        self.mtf     = MTFAnalyzer(self.cache)
        self.ai      = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.checker = AlertChecker(self.cache, self.alerter)
        self.phase   = PhaseDetector()
        self._cycle  = 0
        self._expiry: Optional[str] = None
        # v6.1: expiry refreshed daily once, not every 30min
        self._expiry_fetched_date: Optional[str] = None

    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5: return False
        s = now.replace(hour=MARKET_START[0], minute=MARKET_START[1], second=0)
        e = now.replace(hour=MARKET_END[0],   minute=MARKET_END[1],   second=0)
        return s <= now <= e

    async def _refresh_expiry_if_needed(self):
        """
        v6.1 fix: Expiry refresh only once per trading day.
        Previous code refreshed every 30min — unnecessary API calls.
        """
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        if self._expiry_fetched_date == today_str and self._expiry:
            return  # Already fetched today
        logger.info("🔄 Fetching expiry (daily refresh)...")
        self._expiry = await self.upstox.get_nearest_expiry()
        self._expiry_fetched_date = today_str

    async def run(self):
        logger.info("=" * 60)
        logger.info("NIFTY OPTIONS BOT v6.1 PRO — Koyeb")
        logger.info(f"ATM_RANGE=±{ATM_RANGE} (±{ATM_RANGE*STRIKE_INTERVAL}pts) | "
                    f"Strikes={(ATM_RANGE*2)+1}")
        logger.info(f"Phase1 OI>={PHASE1_OI_BUILD_PCT}% | Phase2 Vol>={PHASE2_VOL_SPIKE_PCT}%")
        logger.info(f"MTF threshold={MTF_OVERALL_THRESHOLD} | Lot={LOT_SIZE}")
        logger.info(f"Candles: 1min+resample5min + 30min direct (Upstox V2)")
        logger.info("=" * 60)

        await self.upstox.init()
        await self._refresh_expiry_if_needed()

        try:
            while True:
                if not self.is_market_open():
                    logger.info("Market closed — waiting 60s")
                    await asyncio.sleep(60)
                    continue
                try:
                    await self._cycle_run()
                except aiohttp.ClientConnectorError:
                    logger.error("❌ Network error — retry 60s")
                    await asyncio.sleep(60)
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(60)

                s5, s30 = self.cache.sizes()
                logger.info(f"⏱ Next in {SNAPSHOT_INTERVAL//60}min | Cache: 5m={s5} 30m={s30}")
                await asyncio.sleep(SNAPSHOT_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            await self.upstox.close()
            await self.alerter.close()

    async def _cycle_run(self):
        self._cycle += 1
        is_analysis = (self._cycle % 6 == 0)   # every 30min

        # v6.1: expiry daily refresh (not every 30min)
        await self._refresh_expiry_if_needed()

        if not self._expiry:
            logger.warning("No expiry — skipping cycle")
            return

        logger.info(
            f"{'📊 ANALYSIS' if is_analysis else '📦 SNAPSHOT'} "
            f"#{self._cycle} | {datetime.now(IST).strftime('%H:%M IST')}"
        )

        # 1. Fetch option chain snapshot
        snap = await self.upstox.fetch_snapshot(self._expiry)
        if not snap:
            logger.warning("⚠️ Snapshot failed")
            return

        await self.cache.add_5min(snap)
        if is_analysis:
            await self.cache.add_30min(snap)

        # 2. Standalone OI/PCR/Volume alerts
        await self.checker.check_all(snap)

        # 3. v6.1 FIXED: 2 API calls (was 3, 5min/15min were invalid on Upstox V2)
        #    df_1m  = last CANDLE_COUNT 1min candles (for VWAP)
        #    df_5m  = 1min resampled to 5min (price action, patterns, trend)
        #    df_30m = 30min direct from Upstox (day trend, S/R)
        df_1m, df_5m, df_30m = await self.upstox.get_candles()

        # 4. Pre-calculate price action with Pandas/Numpy
        recent = await self.cache.get_recent(12)
        pa = PriceActionCalculator.calculate(recent, df_1m, df_5m, df_30m)
        logger.info(
            f"Price: 5m={pa.price_change_5m:+.2f}% | "
            f"Vol:{pa.vol_spike_ratio:.2f}x | Triple:{pa.triple_confirmed}"
        )
        logger.info(f"Trend: {pa.trend.summary}")

        # 5. Phase detection (OI+Volume+Price progression)
        prev   = await self.cache.get_5min_ago(1)
        phases = await self.phase.detect(snap, prev, pa)

        for ps in phases:
            logger.info(f"⚡ Phase {ps.phase}: {ps.direction}")
            await self.alerter.send_raw(ps.message)
            if ps.phase == 3:
                await self._ai_call(snap, pa, ps, df_5m, df_30m)

        # 6. Full MTF analysis every 30min
        if is_analysis:
            await self._full_analysis(snap, pa, df_5m, df_30m)

    async def _ai_call(self, snap: MarketSnapshot, pa: PriceActionInsight,
                       ps: PhaseSignal, df_5m: pd.DataFrame, df_30m: pd.DataFrame):
        logger.info("🤖 Phase 3 → calling DeepSeek V3...")
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            oi = {
                "available": True, "strike_analyses": [],
                "sr": SupportResistance(snap.atm_strike, 0, snap.atm_strike, 0, False, False),
                "overall": ps.direction, "total_bull": 0, "total_bear": 0,
                "overall_pcr": snap.overall_pcr, "pcr_trend": "N/A",
                "pcr_ch_pct": 0, "has_strong": True
            }

        patterns     = PatternDetector.detect(df_5m)
        p_sup, p_res = PatternDetector.sr(df_30m)
        prompt       = PromptBuilder.build(
            snap.spot_price, snap.atm_strike, snap.expiry,
            oi, df_5m, df_30m, patterns, p_sup, p_res, pa, ps
        )
        ai_sig = await self.ai.analyze(prompt)
        if ai_sig and ai_sig.get("confidence", 0) >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa)

    async def _full_analysis(self, snap: MarketSnapshot, pa: PriceActionInsight,
                             df_5m: pd.DataFrame, df_30m: pd.DataFrame):
        logger.info("📊 Full MTF analysis (30min cycle)...")
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            logger.info(oi["reason"])
            return
        if not oi["has_strong"]:
            logger.info("No strong MTF signal — skipping AI call")
            return

        patterns     = PatternDetector.detect(df_5m)
        p_sup, p_res = PatternDetector.sr(df_30m)
        prompt       = PromptBuilder.build(
            snap.spot_price, snap.atm_strike, snap.expiry,
            oi, df_5m, df_30m, patterns, p_sup, p_res, pa, None
        )
        logger.info("🤖 DeepSeek V3...")
        ai_sig = await self.ai.analyze(prompt)

        if not ai_sig:
            # Fallback: use MTF pre-calculations
            sa = next((s for s in oi["strike_analyses"] if s.is_atm), None)
            fb = ("BUY_CALL" if sa and sa.bull_strength > sa.bear_strength
                  else "BUY_PUT" if sa else "WAIT")
            fc = min(10, max(sa.bull_strength, sa.bear_strength)) if sa else 3
            ai_sig = {
                "signal": fb, "confidence": fc,
                "primary_strike": snap.atm_strike,
                "mtf":          {"tf5":"N/A","tf15":"N/A","tf30":"N/A","confirmed":False},
                "entry":        {"now":False,"reason":"AI timeout — using MTF fallback","wait_for":""},
                "price_action": {"momentum":pa.price_momentum,
                                 "triple_confirmed":pa.triple_confirmed,
                                 "confirms_signal":False},
                "candle_pattern":{"pattern":"N/A","type":"","confirms_signal":False,"near_sr":False},
                "trend_analysis":{"day":pa.trend.day_trend,"intraday":pa.trend.intraday_trend,
                                  "all_agree":pa.trend.all_agree,"vwap_confirms":False,"note":""},
                "iv_note":      {"ce_iv":0,"pe_iv":0,"note":""},
                "volume":       {"ok":False,"spike_ratio":pa.vol_spike_ratio,"trap_warning":""},
                "rr":{},"atm":{},"pcr":{},"levels":{}
            }

        conf = ai_sig.get("confidence", 0)
        logger.info(f"Signal: {ai_sig.get('signal','WAIT')} | Conf:{conf}/10")
        if conf >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa)


# ============================================================
#  HTTP SERVER — Koyeb keep-alive
# ============================================================

bot_instance: Optional[NiftyOptionsBot] = None

async def health(request):
    """
    Health endpoint for Koyeb.
    Cache-Control: no-cache ensures Koyeb/UptimeRobot always gets fresh response.
    """
    if bot_instance:
        s5, s30 = bot_instance.cache.sizes()
        mkt     = "OPEN" if bot_instance.is_market_open() else "CLOSED"
        expiry  = bot_instance._expiry or "fetching..."
        status  = (
            f"NIFTY Bot v6.1 PRO | ALIVE ✅\n"
            f"Time: {datetime.now(IST).strftime('%d-%b %H:%M IST')}\n"
            f"Market: {mkt} | Expiry: {expiry}\n"
            f"Cache: 5min={s5}/{CACHE_5MIN_SIZE} | 30min={s30}/{CACHE_30MIN_SIZE}\n"
            f"ATM_RANGE=±{ATM_RANGE} | Phase1≥{PHASE1_OI_BUILD_PCT}% | Phase2≥{PHASE2_VOL_SPIKE_PCT}%"
        )
    else:
        status = "NIFTY Bot v6.1 | Starting..."

    return aiohttp.web.Response(
        text=status,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0"
        }
    )

async def start_bot(app):
    global bot_instance
    bot_instance  = NiftyOptionsBot()
    app["bot_task"] = asyncio.create_task(bot_instance.run())

async def stop_bot(app):
    if "bot_task" in app:
        app["bot_task"].cancel()
        try:
            await app["bot_task"]
        except asyncio.CancelledError:
            pass
    if bot_instance:
        await bot_instance.upstox.close()
        await bot_instance.alerter.close()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/",       health)
    app.router.add_get("/health", health)
    app.on_startup.append(start_bot)
    app.on_cleanup.append(stop_bot)

    port = int(os.getenv("PORT", 8000))

    print(f"""
╔══════════════════════════════════════════╗
║   NIFTY 50 OPTIONS BOT v6.1 PRO          ║
║   Platform: Koyeb | Upstox V2            ║
╠══════════════════════════════════════════╣
║  Asset    : NIFTY 50 Weekly Options      ║
║  Lot Size : {LOT_SIZE} | Strike: ₹{STRIKE_INTERVAL}           ║
║  ATM Range: ±{ATM_RANGE} strikes (±₹{ATM_RANGE*STRIKE_INTERVAL})       ║
║  Polling  : Every {SNAPSHOT_INTERVAL//60} min                  ║
╠══════════════════════════════════════════╣
║  v6.1 FIXES:                             ║
║  ✅ PHASE1 OI ≥ {PHASE1_OI_BUILD_PCT}% (was 4%)          ║
║  ✅ PHASE2 Vol ≥ {PHASE2_VOL_SPIKE_PCT}% (was 20%)       ║
║  ✅ MTF threshold = {MTF_OVERALL_THRESHOLD} (was 12)          ║
║  ✅ Candle: 1min+5min+30min (2 API calls)║
║  ✅ Expiry: daily refresh (not 30min)    ║
╠══════════════════════════════════════════╣
║  CANDLE DATA:                            ║
║  1min  → Pandas resample → 5min          ║
║  30min → Upstox direct (V2 supported)    ║
║  5min+15min+30min OI → DualCache         ║
╠══════════════════════════════════════════╣
║  ENV VARS NEEDED:                        ║
║  UPSTOX_ACCESS_TOKEN                     ║
║  TELEGRAM_BOT_TOKEN                      ║
║  TELEGRAM_CHAT_ID                        ║
║  DEEPSEEK_API_KEY                        ║
║  PORT (default: 8000)                    ║
╚══════════════════════════════════════════╝
Starting on port {port}...
""")

    web.run_app(app, host="0.0.0.0", port=port)
