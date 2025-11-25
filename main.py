#!/usr/bin/env python3
"""
FUTURES DATA BOT - WORKING VERSION
===================================
Fetches last 10 candles for 4 indices from Upstox
Sends to Telegram every 60 seconds

FIXED: Using hardcoded current month symbols
"""

import os
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import pytz
import json
import logging

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
logger = logging.getLogger("FuturesBot")

# API Credentials
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ==================== FUTURES SYMBOLS ====================
# 🔥 HARDCODED SYMBOLS (November 2025)
# तुला हे manually update करावे लागतील दर महिन्याला!

FUTURES_SYMBOLS = {
    'NIFTY': {
        'name': 'NIFTY 50',
        'symbol': 'NSE_FO|NIFTY28NOV24FUT',  # Last Thursday
        'expiry': '28-Nov-2024'
    },
    'BANKNIFTY': {
        'name': 'BANK NIFTY',
        'symbol': 'NSE_FO|BANKNIFTY27NOV24FUT',  # Last Wednesday
        'expiry': '27-Nov-2024'
    },
    'FINNIFTY': {
        'name': 'FIN NIFTY',
        'symbol': 'NSE_FO|FINNIFTY26NOV24FUT',  # Last Tuesday
        'expiry': '26-Nov-2024'
    },
    'MIDCPNIFTY': {
        'name': 'MIDCAP NIFTY',
        'symbol': 'NSE_FO|MIDCPNIFTY25NOV24FUT',  # Last Monday
        'expiry': '25-Nov-2024'
    }
}

# ==================== DATA FETCHER ====================
class FuturesDataFetcher:
    """Fetch historical candles from Upstox"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    async def fetch_candles(self, index_name: str) -> dict:
        """
        Fetch last 10 candles (1-minute interval)
        
        Upstox API:
        GET /v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}
        
        Response format: [timestamp, open, high, low, close, volume, oi]
        """
        config = FUTURES_SYMBOLS[index_name]
        symbol = config['symbol']
        
        async with aiohttp.ClientSession() as session:
            # Date range: last 3 days
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=3)).strftime('%Y-%m-%d')
            
            # URL encode symbol
            enc_symbol = urllib.parse.quote(symbol)
            
            url = f"https://api.upstox.com/v2/historical-candle/{enc_symbol}/1minute/{to_date}/{from_date}"
            
            logger.info(f"🔍 {config['name']}: {symbol}")
            
            try:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        if data.get('status') == 'success' and 'data' in data:
                            raw_candles = data['data'].get('candles', [])
                            
                            if not raw_candles:
                                logger.warning(f"⚠️ {index_name}: No candles")
                                return None
                            
                            # Last 10 candles
                            last_10 = raw_candles[:10]
                            
                            # Parse candles
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
                            
                            logger.info(f"✅ {index_name}: {len(parsed)} candles | Vol: {total_vol:,}")
                            
                            return {
                                "index": config['name'],
                                "symbol": symbol,
                                "expiry": config['expiry'],
                                "candles": parsed,
                                "total_volume": total_vol,
                                "timestamp": datetime.now(IST).isoformat()
                            }
                        else:
                            logger.error(f"❌ {index_name}: Invalid response")
                            return None
                    
                    elif resp.status == 429:
                        logger.warning(f"⏳ Rate limit")
                        return None
                    
                    elif resp.status == 401:
                        logger.error(f"🔑 Invalid token!")
                        return None
                    
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ {index_name}: HTTP {resp.status}")
                        logger.error(f"   {error_text[:200]}")
                        return None
            
            except Exception as e:
                logger.error(f"💥 {index_name}: {e}")
                return None
    
    async def fetch_all_indices(self) -> dict:
        """Fetch all 4 indices"""
        results = {
            "fetch_time": datetime.now(IST).strftime('%d-%b-%Y %I:%M:%S %p'),
            "indices": {}
        }
        
        for index_name in FUTURES_SYMBOLS.keys():
            data = await self.fetch_candles(index_name)
            
            if data:
                results['indices'][index_name] = data
            else:
                results['indices'][index_name] = {
                    "error": "Failed to fetch"
                }
            
            # Delay to avoid rate limit
            await asyncio.sleep(0.5)
        
        return results

# ==================== TELEGRAM SENDER ====================
class TelegramSender:
    """Send to Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_data(self, data: dict):
        """Send summary + JSON file"""
        
        # Summary message
        summary = f"""
🔥 FUTURES DATA

⏰ {data['fetch_time']}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for idx_name, idx_data in data['indices'].items():
            if 'error' not in idx_data:
                config = FUTURES_SYMBOLS[idx_name]
                candles = idx_data.get('candles', [])
                
                if candles:
                    latest = candles[0]
                    summary += f"""
📈 {config['name']}
   Symbol: {idx_data['symbol']}
   Expiry: {config['expiry']}
   Candles: {len(candles)}
   Volume: {idx_data['total_volume']:,}
   Latest: ₹{latest['close']:.2f}
   Time: {latest['timestamp'][-14:-9]}

"""
            else:
                summary += f"""
❌ {FUTURES_SYMBOLS[idx_name]['name']}
   Status: Failed

"""
        
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n📎 JSON file attached"
        
        try:
            # Send summary
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=summary
            )
            
            # Send JSON file
            from io import BytesIO
            
            json_str = json.dumps(data, indent=2)
            json_file = BytesIO(json_str.encode('utf-8'))
            json_file.name = f"futures_{datetime.now(IST).strftime('%H%M%S')}.json"
            
            await self.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=json_file,
                caption="📊 Full Data"
            )
            
            logger.info("✅ Sent to Telegram")
        
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

# ==================== MAIN ====================
async def main():
    """Main loop"""
    
    logger.info("=" * 80)
    logger.info("🚀 FUTURES DATA BOT - WORKING VERSION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Indices:")
    for name, config in FUTURES_SYMBOLS.items():
        logger.info(f"   {config['name']}: {config['symbol']}")
    logger.info("")
    logger.info("⏱️ Interval: 60 seconds")
    logger.info("📦 Data: Last 10 candles (1-min)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    fetcher = FuturesDataFetcher()
    sender = TelegramSender()
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration #{iteration}")
            logger.info(f"{'='*60}\n")
            
            # Fetch data
            data = await fetcher.fetch_all_indices()
            
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
