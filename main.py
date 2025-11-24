#!/usr/bin/env python3
"""
FUTURES SYMBOL FORMAT TESTER
=============================
Test different symbol formats to find working one

We'll test:
- NIFTY24NOVFUT vs NIFTY24NOV vs NIFTY24NOV25FUT
- Different month variations
- Different year formats

Author: Symbol Tester
Date: November 24, 2025
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from calendar import monthrange
import pytz

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', '')

# ==================== SYMBOL GENERATOR ====================
class SymbolTester:
    """Test different futures symbol formats"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def generate_symbol_variations(self, index_prefix: str):
        """
        Generate all possible symbol variations for testing
        
        Patterns to test:
        1. NIFTY24NOVFUT
        2. NIFTY24NOV25FUT
        3. NIFTY24NOV
        4. NIFTYNOV24FUT
        5. NIFTY2024NOVFUT
        """
        now = datetime.now(IST)
        
        # Current month
        current_month = now.strftime('%b').upper()  # NOV
        current_year_2digit = now.year % 100  # 24 (for 2024) or 25 (for 2025)
        current_year_4digit = now.year  # 2025
        
        # Next month
        next_month_date = now + timedelta(days=32)
        next_month = next_month_date.strftime('%b').upper()  # DEC
        next_month_year_2digit = next_month_date.year % 100
        
        # Previous month
        prev_month_date = now - timedelta(days=32)
        prev_month = prev_month_date.strftime('%b').upper()  # OCT
        
        # Contract year (April-March cycle)
        contract_year = current_year_2digit
        if now.month <= 3:  # Jan, Feb, Mar
            contract_year = current_year_2digit - 1
        
        variations = []
        
        # Pattern 1: NIFTY24NOVFUT (Contract year + Month + FUT)
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{current_month}FUT")
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{next_month}FUT")
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{prev_month}FUT")
        
        # Pattern 2: NIFTY24NOV25FUT (Contract year + Month + Expiry year + FUT)
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{current_month}{current_year_2digit:02d}FUT")
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{next_month}{next_month_year_2digit:02d}FUT")
        
        # Pattern 3: NIFTY24NOV (No FUT suffix)
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{current_month}")
        variations.append(f"NSE_FO|{index_prefix}{contract_year:02d}{next_month}")
        
        # Pattern 4: NIFTYNOV24FUT (Month + Year + FUT)
        variations.append(f"NSE_FO|{index_prefix}{current_month}{current_year_2digit:02d}FUT")
        variations.append(f"NSE_FO|{index_prefix}{next_month}{next_month_year_2digit:02d}FUT")
        
        # Pattern 5: NIFTY2024NOVFUT (Full year)
        variations.append(f"NSE_FO|{index_prefix}{current_year_4digit}{current_month}FUT")
        
        # Pattern 6: Current year instead of contract year
        variations.append(f"NSE_FO|{index_prefix}{current_year_2digit:02d}{current_month}FUT")
        variations.append(f"NSE_FO|{index_prefix}{current_year_2digit:02d}{next_month}FUT")
        
        return variations
    
    async def test_symbol(self, symbol: str, session: aiohttp.ClientSession):
        """Test if a symbol works by fetching LTP"""
        
        symbol_encoded = symbol.replace('|', '%7C')
        url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={symbol_encoded}"
        
        try:
            async with session.get(url, headers=self.headers,
                                 timeout=aiohttp.ClientTimeout(total=10)) as resp:
                
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Check if data exists
                    if 'data' in data and symbol in data['data']:
                        ltp = data['data'][symbol].get('last_price', 0)
                        if ltp > 0:
                            return True, ltp, None
                    
                    return False, 0, "No price data"
                
                elif resp.status == 404:
                    return False, 0, "Symbol not found (404)"
                
                elif resp.status == 400:
                    return False, 0, "Invalid format (400)"
                
                else:
                    error = await resp.text()
                    return False, 0, f"HTTP {resp.status}"
        
        except asyncio.TimeoutError:
            return False, 0, "Timeout"
        
        except Exception as e:
            return False, 0, str(e)
    
    async def find_working_symbol(self, index_name: str, index_prefix: str):
        """Find working symbol for an index"""
        
        print(f"\n{'='*70}")
        print(f"🔍 TESTING: {index_name}")
        print(f"{'='*70}")
        
        variations = self.generate_symbol_variations(index_prefix)
        
        print(f"\n📋 Generated {len(variations)} symbol variations to test")
        print(f"⏳ Testing each variation...\n")
        
        async with aiohttp.ClientSession() as session:
            
            working_symbols = []
            
            for i, symbol in enumerate(variations, 1):
                print(f"{i:2d}. Testing: {symbol:<55}", end=" ")
                
                success, ltp, error = await self.test_symbol(symbol, session)
                
                if success:
                    print(f"✅ WORKS! Price: ₹{ltp}")
                    working_symbols.append({
                        'symbol': symbol,
                        'ltp': ltp
                    })
                else:
                    print(f"❌ {error}")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
            
            print(f"\n{'='*70}")
            
            if working_symbols:
                print(f"✅ FOUND {len(working_symbols)} WORKING SYMBOL(S):\n")
                for item in working_symbols:
                    print(f"   💡 {item['symbol']}")
                    print(f"      Last Price: ₹{item['ltp']}")
                    print()
            else:
                print(f"❌ NO WORKING SYMBOLS FOUND")
            
            print(f"{'='*70}")
            
            return working_symbols
    
    async def test_all_indices(self):
        """Test all indices"""
        
        indices = {
            'NIFTY': 'NIFTY',
            'BANKNIFTY': 'BANKNIFTY',
            'FINNIFTY': 'FINNIFTY',
            'MIDCPNIFTY': 'MIDCPNIFTY'
        }
        
        print("="*70)
        print("🧪 FUTURES SYMBOL FORMAT TESTER")
        print("="*70)
        print(f"⏰ Time: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
        print(f"🎯 Testing: {len(indices)} indices")
        print("="*70)
        
        if not UPSTOX_ACCESS_TOKEN:
            print("\n❌ ERROR: UPSTOX_ACCESS_TOKEN not set!")
            return
        
        print(f"\n✅ Token: {UPSTOX_ACCESS_TOKEN[:20]}...")
        
        all_results = {}
        
        for index_name, index_prefix in indices.items():
            results = await self.find_working_symbol(index_name, index_prefix)
            all_results[index_name] = results
            await asyncio.sleep(2)  # Delay between indices
        
        # Final Summary
        print("\n\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        for index_name, results in all_results.items():
            if results:
                print(f"\n✅ {index_name}:")
                for item in results:
                    print(f"   {item['symbol']}")
            else:
                print(f"\n❌ {index_name}: No working symbols found")
        
        print("\n" + "="*70)
        print("🎯 SUMMARY")
        print("="*70)
        
        total_found = sum(len(r) for r in all_results.values())
        
        if total_found > 0:
            print(f"\n✅ Found {total_found} working symbols!")
            print("\n💡 Copy the working symbols and update your data fetching code")
        else:
            print("\n❌ No working symbols found")
            print("\n💡 Possible issues:")
            print("   1. Token expired/invalid")
            print("   2. Market closed (but LTP should still work)")
            print("   3. Upstox API format changed")
        
        print("="*70)

# ==================== MAIN ====================
async def main():
    """Main entry point"""
    tester = SymbolTester()
    await tester.test_all_indices()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Testing stopped")
    except Exception as e:
        print(f"\n\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
