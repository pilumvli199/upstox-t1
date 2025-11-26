#!/usr/bin/env python3
"""
UPSTOX API DIAGNOSTIC TOOL
==========================
Test your API token and troubleshoot issues

Usage:
    export UPSTOX_ACCESS_TOKEN="your_token"
    python upstox_diagnostic.py
"""

import os
import asyncio
import aiohttp
import urllib.parse
import json
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

# Configuration
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'YOUR_TOKEN_HERE')

# Test symbols
TEST_SYMBOLS = {
    'NIFTY': "NSE_INDEX|Nifty 50",
    'BANKNIFTY': "NSE_INDEX|Nifty Bank",
    'FINNIFTY': "NSE_INDEX|Nifty Fin Service",
    'MIDCPNIFTY': "NSE_INDEX|NIFTY MID SELECT"
}

print("=" * 80)
print("🔧 UPSTOX API DIAGNOSTIC TOOL")
print("=" * 80)
print()

# Check token
if UPSTOX_ACCESS_TOKEN == 'YOUR_TOKEN_HERE':
    print("❌ ERROR: Please set UPSTOX_ACCESS_TOKEN environment variable")
    print("   export UPSTOX_ACCESS_TOKEN='your_actual_token'")
    exit(1)

print(f"✅ Token found: {UPSTOX_ACCESS_TOKEN[:20]}...")
print()

async def test_api():
    """Test API endpoints"""
    
    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
        "Accept": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Profile API (token validation)
        print("🔍 TEST 1: Token Validation")
        print("-" * 80)
        
        try:
            async with session.get(
                "https://api.upstox.com/v2/user/profile",
                headers=headers,
                timeout=10
            ) as resp:
                status = resp.status
                data = await resp.json()
                
                print(f"Status: {status}")
                print(f"Response: {json.dumps(data, indent=2)}")
                
                if status == 200 and data.get('status') == 'success':
                    user_data = data.get('data', {})
                    print(f"\n✅ Token Valid!")
                    print(f"   User: {user_data.get('user_name', 'Unknown')}")
                    print(f"   Email: {user_data.get('email', 'Unknown')}")
                elif status == 401:
                    print("\n❌ Token INVALID or EXPIRED!")
                    print("   Please generate new token from:")
                    print("   https://account.upstox.com/developer/apps")
                    return False
                else:
                    print(f"\n⚠️ Unexpected response: {status}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
        
        print("\n")
        
        # Test 2: Market Quote API (spot prices)
        print("🔍 TEST 2: Spot Price Fetch")
        print("-" * 80)
        
        for index_name, symbol in TEST_SYMBOLS.items():
            print(f"\n{index_name}:")
            print(f"  Symbol: {symbol}")
            
            enc_symbol = urllib.parse.quote(symbol, safe='')
            url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={enc_symbol}"
            
            print(f"  URL: {url}")
            
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    status = resp.status
                    data = await resp.json()
                    
                    print(f"  Status: {status}")
                    
                    if status == 200 and data.get('status') == 'success':
                        quote_data = data.get('data', {})
                        
                        print(f"  Available keys: {list(quote_data.keys())}")
                        
                        if symbol in quote_data:
                            price_info = quote_data[symbol]
                            price = price_info.get('last_price', 0)
                            print(f"  ✅ Price: ₹{price:.2f}")
                        else:
                            print(f"  ❌ Symbol not in response!")
                            print(f"  Response: {json.dumps(data, indent=2)}")
                    elif status == 400:
                        print(f"  ❌ Bad Request (400)")
                        print(f"  Response: {json.dumps(data, indent=2)}")
                        
                        if 'errors' in data:
                            for error in data['errors']:
                                print(f"  Error: {error}")
                    else:
                        print(f"  ⚠️ Status: {status}")
                        print(f"  Response: {json.dumps(data, indent=2)}")
                        
            except Exception as e:
                print(f"  ❌ Exception: {e}")
            
            await asyncio.sleep(1)  # Rate limit
        
        print("\n")
        
        # Test 3: Historical Candle API
        print("🔍 TEST 3: Historical Candle (Futures)")
        print("-" * 80)
        
        # Try NIFTY futures
        now = datetime.now(IST)
        year = now.year % 100
        month = now.strftime('%b').upper()
        
        futures_symbol = f"NSE_FO|NIFTY{year:02d}{month}FUT"
        enc_futures = urllib.parse.quote(futures_symbol, safe='')
        
        to_date = now.strftime('%Y-%m-%d')
        from_date = (now - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        
        url = f"https://api.upstox.com/v2/historical-candle/{enc_futures}/1minute/{to_date}/{from_date}"
        
        print(f"  Symbol: {futures_symbol}")
        print(f"  URL: {url}")
        
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                status = resp.status
                data = await resp.json()
                
                print(f"  Status: {status}")
                
                if status == 200 and data.get('status') == 'success':
                    candles = data.get('data', {}).get('candles', [])
                    print(f"  ✅ Candles: {len(candles)}")
                    if candles:
                        print(f"  Latest: {candles[0]}")
                elif status == 400:
                    print(f"  ❌ Bad Request (400)")
                    print(f"  Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"  ⚠️ Status: {status}")
                    print(f"  Response: {json.dumps(data, indent=2)}")
                    
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print("\n")
        
        # Test 4: Option Chain API
        print("🔍 TEST 4: Option Chain")
        print("-" * 80)
        
        symbol = TEST_SYMBOLS['NIFTY']
        enc_symbol = urllib.parse.quote(symbol, safe='')
        
        # Calculate next Tuesday expiry
        today = now.date()
        days_to_tuesday = (1 - today.weekday() + 7) % 7
        if days_to_tuesday == 0:
            days_to_tuesday = 7
        expiry = today + datetime.timedelta(days=days_to_tuesday)
        expiry_str = expiry.strftime('%Y-%m-%d')
        
        url = f"https://api.upstox.com/v2/option/chain?instrument_key={enc_symbol}&expiry_date={expiry_str}"
        
        print(f"  Symbol: {symbol}")
        print(f"  Expiry: {expiry_str}")
        print(f"  URL: {url}")
        
        try:
            async with session.get(url, headers=headers, timeout=15) as resp:
                status = resp.status
                data = await resp.json()
                
                print(f"  Status: {status}")
                
                if status == 200 and data.get('status') == 'success':
                    options = data.get('data', [])
                    print(f"  ✅ Strikes: {len(options)}")
                    if options:
                        sample = options[0]
                        print(f"  Sample strike: {sample.get('strike_price')}")
                elif status == 400:
                    print(f"  ❌ Bad Request (400)")
                    print(f"  Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"  ⚠️ Status: {status}")
                    print(f"  Response: {json.dumps(data, indent=2)}")
                    
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print("\n")
        print("=" * 80)
        print("🏁 DIAGNOSTIC COMPLETE")
        print("=" * 80)
        print()
        print("📋 RECOMMENDATIONS:")
        print()
        print("1. If Status 401: Token expired → Generate new token")
        print("2. If Status 400: Check symbol format or API changes")
        print("3. If 'Symbol not found': Upstox may have changed symbol names")
        print("4. If market closed: Try during 9:15 AM - 3:30 PM IST")
        print()
        print("🔗 Useful Links:")
        print("   • Token: https://account.upstox.com/developer/apps")
        print("   • Docs: https://upstox.com/developer/api-documentation/")
        print()

if __name__ == "__main__":
    asyncio.run(test_api())
