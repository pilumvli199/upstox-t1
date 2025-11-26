#!/usr/bin/env python3
"""
NIFTY50 WEEKLY FUTURES BOT - INTRADAY API VERSION
==================================================
✅ Only NIFTY50 with Weekly Expiry (Tuesday)
✅ Auto-detects nearest Tuesday expiry
✅ Uses Upstox INTRADAY API for live data
✅ Sends to Telegram every 60 seconds
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import pytz
import json
import logging
import gzip
from io import BytesIO

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Install: pip install python-telegram-bot")
    exit(1)

# ==================== CONFIG ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("NiftyWeeklyBot")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Upstox Instruments JSON URL
INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# ==================== EXPIRY CALCULATOR ====================
def get_next_tuesday_expiry(from_date=None):
    """
    Calculate next Tuesday expiry for NIFTY50 weekly
    
    NIFTY50 Weekly Expiry Rules:
    - Expires every Tuesday
    - If Tuesday is holiday, then previous trading day
    """
    if from_date is None:
        from_date = datetime.now(IST)
    
    # Find next Tuesday
    days_until_tuesday = (1 - from_date.weekday()) % 7  # Tuesday = 1
    
    if days_until_tuesday == 0:
        # Today is Tuesday
        # Check if after market hours (3:30 PM)
        market_close = from_date.replace(hour=15, minute=30, second=0, microsecond=0)
        if from_date > market_close:
            # Look for next Tuesday
            next_tuesday = from_date + timedelta(days=7)
        else:
            # Current Tuesday expiry
            next_tuesday = from_date
    else:
        # Next Tuesday
        next_tuesday = from_date + timedelta(days=days_until_tuesday)
    
    # Set to end of day (expiry time)
    expiry_datetime = next_tuesday.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return expiry_datetime

# ==================== INSTRUMENTS FETCHER ====================
class NiftyWeeklyFetcher:
    """Download and find NIFTY50 weekly futures"""
    
    def __init__(self):
        self.instruments = []
        self.nifty_weekly = None
    
    async def download_instruments(self):
        """Download Upstox instruments JSON file (gzipped)"""
        logger.info("📥 Downloading Upstox instruments...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(INSTRUMENTS_JSON_URL) as resp:
                    if resp.status == 200:
                        compressed = await resp.read()
                        decompressed = gzip.decompress(compressed)
                        self.instruments = json.loads(decompressed)
                        logger.info(f"✅ Loaded {len(self.instruments)} instruments")
                        return True
                    else:
                        logger.error(f"❌ HTTP {resp.status}")
                        return False
            except Exception as e:
                logger.error(f"💥 Download failed: {e}")
                return False
    
    def find_nifty_weekly_future(self):
        """
        Find NIFTY50 weekly futures with nearest Tuesday expiry
        
        Logic:
        1. Calculate next Tuesday expiry
        2. Search for NIFTY futures expiring on that date
        3. Filter FUT type only (not options)
        """
        logger.info("🔍 Finding NIFTY50 weekly futures...")
        
        # Calculate target expiry
        target_expiry = get_next_tuesday_expiry()
        target_date = target_expiry.date()
        
        logger.info(f"🎯 Target Weekly Expiry: {target_date.strftime('%d-%b-%Y')} (Tuesday)")
        
        # Search in instruments
        nifty_futures = []
        
        for instrument in self.instruments:
            # Must be NSE Futures & Options segment
            if instrument.get('segment') != 'NSE_FO':
                continue
            
            # Must be Futures (not options)
            if instrument.get('instrument_type') != 'FUT':
                continue
            
            # Must be NIFTY (not BANKNIFTY, FINNIFTY, etc)
            name = instrument.get('name', '')
            if name != 'NIFTY':
                continue
            
            # Check expiry date
            expiry_ms = instrument.get('expiry', 0)
            if not expiry_ms:
                continue
            
            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
            expiry_date = expiry_dt.date()
            
            # Match with target Tuesday
            if expiry_date == target_date:
                nifty_futures.append({
                    'instrument_key': instrument.get('instrument_key'),
                    'exchange_token': instrument.get('exchange_token'),
                    'trading_symbol': instrument.get('trading_symbol'),
                    'expiry': expiry_dt.strftime('%d-%b-%Y'),
                    'expiry_timestamp': expiry_ms,
                    'expiry_day': expiry_dt.strftime('%A')  # Should be Tuesday
                })
        
        if len(nifty_futures) == 0:
            logger.error(f"❌ No NIFTY weekly futures found for {target_date}")
            return False
        
        # Should typically find 1 weekly future
        # If multiple, pick the first one
        self.nifty_weekly = nifty_futures[0]
        
        logger.info(f"✅ Found NIFTY Weekly Future:")
        logger.info(f"   Symbol: {self.nifty_weekly['trading_symbol']}")
        logger.info(f"   Expiry: {self.nifty_weekly['expiry']} ({self.nifty_weekly['expiry_day']})")
        logger.info(f"   Key: {self.nifty_weekly['instrument_key']}")
        
        return True
    
    async def initialize(self):
        """Download and parse instruments"""
        success = await self.download_instruments()
        if not success:
            return False
        
        return self.find_nifty_weekly_future()

# ==================== DATA FETCHER (INTRADAY API) ====================
class NiftyDataFetcher:
    """Fetch NIFTY50 weekly futures data using INTRADAY API"""
    
    def __init__(self, nifty_info):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.nifty_info = nifty_info
    
    async def fetch_candles(self) -> dict:
        """
        Fetch last 10 candles using INTRADAY API
        
        Endpoint: /v2/historical-candle/intraday/{instrument_key}/1minute
        Returns: Today's 1-minute candles
        Format: [timestamp, open, high, low, close, volume, oi]
        """
        instrument_key = self.nifty_info['instrument_key']
        
        # Check market hours
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_hours = market_open <= now <= market_close
        is_weekday = now.weekday() < 5  # Mon-Fri
        
        async with aiohttp.ClientSession() as session:
            # URL encode instrument key
            enc_key = urllib.parse.quote(instrument_key)
            
            # INTRADAY API - Returns ONLY today's candles
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{enc_key}/1minute"
            
            logger.info(f"🔍 NIFTY50 Weekly: {instrument_key}")
            
            # Market status
            if is_market_hours and is_weekday:
                logger.info(f"   📊 Market: OPEN (Live data)")
            else:
                logger.info(f"   ⏸️ Market: CLOSED")
            
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success' and 'data' in data:
                            raw_candles = data['data'].get('candles', [])
                            
                            if not raw_candles:
                                logger.warning(f"⚠️ NIFTY50: No candles (market not started?)")
                                return None
                            
                            # Last 10 candles
                            last_10 = raw_candles[:10]
                            
                            # Parse
                            parsed = []
                            total_vol = 0
                            
                            for c in last_10:
                                candle = {
                                    "timestamp": c[0],
                                    "open": float(c[1]),
                                    "high": float(c[2]),
                                    "low": float(c[3]),
                                    "close": float(c[4]),
                                    "volume": int(c[5]),
                                    "oi": int(c[6]) if len(c) > 6 else 0
                                }
                                parsed.append(candle)
                                total_vol += candle['volume']
                            
                            # Check data freshness
                            latest_time = datetime.fromisoformat(parsed[0]['timestamp'])
                            data_age_minutes = (now - latest_time).total_seconds() / 60
                            
                            if data_age_minutes < 5:
                                logger.info(f"🟢 NIFTY50: {len(parsed)} candles | Vol: {total_vol:,} | LIVE ({data_age_minutes:.1f}m old)")
                            else:
                                logger.info(f"🟡 NIFTY50: {len(parsed)} candles | Vol: {total_vol:,} | {data_age_minutes:.0f}m old")
                            
                            return {
                                "index": "NIFTY50",
                                "instrument_key": instrument_key,
                                "trading_symbol": self.nifty_info['trading_symbol'],
                                "expiry": self.nifty_info['expiry'],
                                "expiry_day": self.nifty_info['expiry_day'],
                                "candles": parsed,
                                "total_volume": total_vol,
                                "data_age_minutes": round(data_age_minutes, 1),
                                "is_live": data_age_minutes < 5,
                                "market_status": "OPEN" if (is_market_hours and is_weekday) else "CLOSED",
                                "timestamp": datetime.now(IST).isoformat()
                            }
                        else:
                            logger.error(f"❌ NIFTY50: Invalid response")
                            return None
                    
                    elif resp.status == 401:
                        logger.error(f"🔑 Invalid token!")
                        return None
                    
                    elif resp.status == 429:
                        logger.warning(f"⏳ Rate limit")
                        return None
                    
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ NIFTY50: HTTP {resp.status}")
                        logger.error(f"   {error_text[:200]}")
                        return None
            
            except Exception as e:
                logger.error(f"💥 NIFTY50: {e}")
                return None

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send to Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_data(self, data: dict):
        """Send summary + JSON file"""
        
        if data is None:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="❌ Failed to fetch NIFTY50 data"
            )
            return
        
        is_live = data.get('is_live', False)
        data_age = data.get('data_age_minutes', 0)
        
        market_emoji = "🟢" if is_live else "🟡"
        status_text = "LIVE DATA" if is_live else "DELAYED DATA"
        
        candles = data.get('candles', [])
        latest = candles[0] if candles else None
        
        summary = f"""
{market_emoji} NIFTY50 WEEKLY FUTURES - {status_text}

⏰ {datetime.now(IST).strftime('%d-%b-%Y %I:%M:%S %p')}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 NIFTY50 DATA
━━━━━━━━━━━━━━━━━━━━━━━━

📈 Symbol: {data['trading_symbol']}
📅 Expiry: {data['expiry']} ({data['expiry_day']})
🕐 Status: {market_emoji} {"Live" if is_live else f"{data_age:.0f}m ago"}
📊 Market: {data.get('market_status', 'UNKNOWN')}
📉 Candles: {len(candles)}
📦 Volume: {data['total_volume']:,}
"""
        
        if latest:
            latest_time = datetime.fromisoformat(latest['timestamp'])
            summary += f"""
💰 Latest Price: ₹{latest['close']:.2f}
📈 Open: ₹{latest['open']:.2f}
📊 High: ₹{latest['high']:.2f}
📉 Low: ₹{latest['low']:.2f}
🕐 Time: {latest_time.strftime('%I:%M %p')}
📊 OI: {latest['oi']:,}
"""
        
        summary += "\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if not is_live:
            summary += "ℹ️ Market hours: Mon-Fri, 9:15 AM - 3:30 PM IST\n"
            summary += "ℹ️ Weekly expiry: Every Tuesday\n\n"
        
        summary += "📎 JSON attached"
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send JSON
            json_str = json.dumps(data, indent=2)
            json_file = BytesIO(json_str.encode('utf-8'))
            json_file.name = f"nifty50_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Full NIFTY50 Data"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")

# ==================== MAIN ====================
async def main():
    """Main loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 NIFTY50 WEEKLY FUTURES BOT - INTRADAY API VERSION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Only NIFTY50 with Weekly Expiry (Tuesday)")
    logger.info("🔥 Using INTRADAY API for live today's data")
    logger.info("")
    
    # Initialize instruments
    logger.info("📥 Loading NIFTY50 weekly futures from Upstox...")
    nifty_fetcher = NiftyWeeklyFetcher()
    
    success = await nifty_fetcher.initialize()
    if not success:
        logger.error("❌ Failed to find NIFTY50 weekly futures!")
        return
    
    logger.info("")
    logger.info("✅ NIFTY50 weekly future loaded successfully")
    logger.info("")
    logger.info("⏱️ Interval: 60 seconds")
    logger.info("📦 Data: Last 10 candles (1-min)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Create fetcher and sender
    fetcher = NiftyDataFetcher(nifty_fetcher.nifty_weekly)
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*60}\n")
            
            # Fetch data
            data = await fetcher.fetch_candles()
            
            # Send to Telegram
            await sender.send_data(data)
            
            # Wait
            logger.info("\n⏳ Waiting 60 seconds...\n")
            await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
            break
        
        except Exception as e:
            logger.error(f"💥 Error: {e}")
            logger.info("   Retrying in 60s...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Bye")
