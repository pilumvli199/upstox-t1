#!/usr/bin/env python3
"""
NIFTY OPTIONS BOT V13.1 - FIXED VERSION
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
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from calendar import monthrange

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("MultiIndexMaster-V13.1-FIXED")

UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')

# ✅ VERIFIED INSTRUMENT KEYS (Upstox Official)
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

ACTIVE_INDICES = list(INDICES.keys())

# ==================== FIXED DATA FEED ====================
class MultiIndexDataFeed:
    """Fixed version with proper URL encoding"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_config = INDICES[index_name]
        self.spot_symbol = self.index_config['spot']
        self.strike_gap = self.index_config['strike_gap']
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.futures_symbol = self.get_futures_symbol()
    
    def get_futures_symbol(self) -> str:
        """Generate correct futures symbol"""
        now = datetime.now(IST)
        year = now.year
        month = now.month
        
        # Calculate expiry month
        last_day = monthrange(year, month)[1]
        last_date = datetime(year, month, last_day, tzinfo=IST)
        days_to_tuesday = (last_date.weekday() - 1) % 7
        last_tuesday = last_date - timedelta(days=days_to_tuesday)
        
        if now.date() > last_tuesday.date() or (
            now.date() == last_tuesday.date() and now.time() > time(15, 30)
        ):
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        
        year_short = year % 100
        month_name = datetime(year, month, 1).strftime('%b').upper()
        
        prefix = self.index_config['futures_prefix']
        return f"NSE_FO|{prefix}{year_short:02d}{month_name}FUT"
    
    async def fetch_with_retry(self, url: str, session: aiohttp.ClientSession):
        """Fetch with detailed logging"""
        for attempt in range(3):
            try:
                logger.info(f"🔗 [{self.index_name}] Fetching: {url}")
                
                async with session.get(url, headers=self.headers, 
                                     timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    status = resp.status
                    text = await resp.text()
                    
                    logger.info(f"📡 [{self.index_name}] Status: {status}")
                    
                    if status == 200:
                        return json.loads(text)
                    else:
                        logger.error(f"❌ [{self.index_name}] Response: {text[:200]}")
                        await asyncio.sleep(2 * (attempt + 1))
                        
            except Exception as e:
                logger.error(f"💥 [{self.index_name}] Error: {e}")
                await asyncio.sleep(2 * (attempt + 1))
        
        return None
    
    async def get_market_data(self):
        """Fetch market data with proper encoding"""
        async with aiohttp.ClientSession() as session:
            spot_price = 0
            futures_price = 0
            
            # 1️⃣ GET SPOT PRICE - Use proper URL encoding
            # Don't encode pipe and colon separately
            spot_encoded = self.spot_symbol.replace('|', '%7C').replace(' ', '%20')
            ltp_url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={spot_encoded}"
            
            logger.info(f"🎯 [{self.index_name}] Spot Symbol: {self.spot_symbol}")
            logger.info(f"🔗 [{self.index_name}] LTP URL: {ltp_url}")
            
            ltp_data = await self.fetch_with_retry(ltp_url, session)
            
            if ltp_data and 'data' in ltp_data:
                # Try multiple key formats
                for key in [self.spot_symbol, 
                           self.spot_symbol.replace('NSE_INDEX|', 'NSE_INDEX:'),
                           self.index_config['name']]:
                    if key in ltp_data['data']:
                        spot_price = ltp_data['data'][key].get('last_price', 0)
                        if spot_price > 0:
                            logger.info(f"✅ [{self.index_name}] Spot Price: {spot_price}")
                            break
            
            if spot_price == 0:
                logger.error(f"❌ [{self.index_name}] Failed to get spot price")
                logger.error(f"Response keys: {list(ltp_data.get('data', {}).keys()) if ltp_data else 'None'}")
                return pd.DataFrame(), {}, "", 0, 0, 0
            
            # 2️⃣ GET FUTURES PRICE from Option Chain
            # We'll use spot price as futures approximation for now
            futures_price = spot_price
            
            # 3️⃣ GET OPTION CHAIN
            expiry = self.get_expiry_date()
            
            # Proper URL encoding for option chain
            spot_encoded = self.spot_symbol.replace('|', '%7C').replace(' ', '%20')
            chain_url = f"https://api.upstox.com/v2/option/chain?instrument_key={spot_encoded}&expiry_date={expiry}"
            
            logger.info(f"🔗 [{self.index_name}] Chain URL: {chain_url}")
            
            strike_data = {}
            total_volume = 0
            
            chain_data = await self.fetch_with_retry(chain_url, session)
            
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
                
                logger.info(f"✅ [{self.index_name}] Got {len(strike_data)} strikes")
            
            # Create dummy DataFrame for now (since historical candles don't work for indices)
            df = pd.DataFrame()
            
            return df, strike_data, expiry, spot_price, futures_price, total_volume
    
    def get_expiry_date(self) -> str:
        """Calculate expiry date"""
        now = datetime.now(IST)
        today = now.date()
        config = self.index_config
        
        if config['has_weekly']:
            # NIFTY: Next Tuesday
            days_to_tuesday = (1 - today.weekday() + 7) % 7
            
            if days_to_tuesday == 0:
                if now.time() > time(15, 30):
                    expiry = today + timedelta(days=7)
                else:
                    expiry = today
            else:
                expiry = today + timedelta(days=days_to_tuesday)
        else:
            # Others: Last Tuesday of month
            year = now.year
            month = now.month
            
            last_day = monthrange(year, month)[1]
            last_date = datetime(year, month, last_day)
            
            days_to_tuesday = (last_date.weekday() - 1) % 7
            last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            if now.date() > last_tuesday.date() or (
                now.date() == last_tuesday.date() and now.time() > time(15, 30)
            ):
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                
                last_day = monthrange(year, month)[1]
                last_date = datetime(year, month, last_day)
                days_to_tuesday = (last_date.weekday() - 1) % 7
                last_tuesday = last_date - timedelta(days=days_to_tuesday)
            
            expiry = last_tuesday.date()
        
        return expiry.strftime('%Y-%m-%d')

# ==================== TEST FUNCTION ====================
async def test_api():
    """Test API with detailed logging"""
    logger.info("🧪 Testing API Connections...")
    
    for index_name in ['NIFTY']:  # Test NIFTY first
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {index_name}")
        logger.info(f"{'='*60}")
        
        feed = MultiIndexDataFeed(index_name)
        df, strikes, expiry, spot, futures, vol = await feed.get_market_data()
        
        logger.info(f"\n📊 Results:")
        logger.info(f"   Spot: {spot}")
        logger.info(f"   Futures: {futures}")
        logger.info(f"   Expiry: {expiry}")
        logger.info(f"   Strikes: {len(strikes)}")
        logger.info(f"   Volume: {vol}")

if __name__ == "__main__":
    # Run test first
    asyncio.run(test_api())
