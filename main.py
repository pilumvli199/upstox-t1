#!/usr/bin/env python3
"""
UPSTOX INSTRUMENT SEARCH
========================
Find correct futures symbols for NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY

This will search Upstox's instrument database and show:
- Available futures contracts
- Correct symbol format
- Expiry dates

Author: Instrument Search Tool
Date: November 24, 2025
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
import pytz

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', '')

# Indices to search
SEARCH_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']

# ==================== INSTRUMENT SEARCHER ====================
class InstrumentSearcher:
    """Search Upstox instrument database"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        self.base_url = "https://api.upstox.com/v2"
    
    async def download_instruments(self, session: aiohttp.ClientSession):
        """
        Download instrument master file from Upstox
        This contains ALL available instruments
        """
        print("\n📥 Downloading Upstox Instrument Database...")
        print("   (This may take 10-15 seconds)")
        
        # Upstox provides instrument CSV
        url = f"{self.base_url}/instruments"
        
        try:
            async with session.get(url, headers=self.headers,
                                 timeout=aiohttp.ClientTimeout(total=30)) as resp:
                
                if resp.status == 200:
                    content = await resp.text()
                    print(f"   ✅ Downloaded ({len(content)} bytes)")
                    return content
                else:
                    error = await resp.text()
                    print(f"   ❌ Failed: HTTP {resp.status}")
                    print(f"   Error: {error[:200]}")
                    return None
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def parse_instruments(self, csv_content: str, index_prefix: str):
        """
        Parse CSV and find futures for specific index
        """
        if not csv_content:
            return []
        
        lines = csv_content.strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # First line is header
        header = lines[0].split(',')
        
        # Find column indices
        try:
            instrument_key_idx = header.index('instrument_key')
            name_idx = header.index('name')
            expiry_idx = header.index('expiry')
            instrument_type_idx = header.index('instrument_type')
            exchange_idx = header.index('exchange')
        except ValueError as e:
            print(f"   ⚠️ CSV format issue: {e}")
            return []
        
        futures = []
        
        for line in lines[1:]:
            parts = line.split(',')
            
            if len(parts) <= max(instrument_key_idx, name_idx, expiry_idx, 
                                instrument_type_idx, exchange_idx):
                continue
            
            instrument_key = parts[instrument_key_idx]
            name = parts[name_idx]
            expiry = parts[expiry_idx]
            inst_type = parts[instrument_type_idx]
            exchange = parts[exchange_idx]
            
            # Filter: NSE_FO exchange, FUTIDX type, matching index
            if (exchange == 'NSE_FO' and 
                inst_type == 'FUTIDX' and 
                index_prefix.upper() in name.upper()):
                
                futures.append({
                    'instrument_key': instrument_key,
                    'name': name,
                    'expiry': expiry,
                    'type': inst_type
                })
        
        return futures
    
    async def search_futures(self, index_name: str, session: aiohttp.ClientSession):
        """Search futures for specific index"""
        
        print(f"\n{'='*70}")
        print(f"🔍 SEARCHING: {index_name} FUTURES")
        print(f"{'='*70}")
        
        # Download instruments
        csv_content = await self.download_instruments(session)
        
        if not csv_content:
            print(f"   ❌ Could not download instrument database")
            return
        
        # Parse and filter
        print(f"\n📊 Parsing {index_name} futures...")
        futures = self.parse_instruments(csv_content, index_name)
        
        if not futures:
            print(f"   ❌ No futures found for {index_name}")
            return
        
        # Sort by expiry
        futures.sort(key=lambda x: x['expiry'])
        
        print(f"\n✅ Found {len(futures)} {index_name} futures contracts:")
        print(f"\n{'Instrument Key':<45} {'Name':<20} {'Expiry'}")
        print("-" * 85)
        
        today = datetime.now(IST).date()
        
        for i, fut in enumerate(futures[:10]):  # Show first 10
            instrument_key = fut['instrument_key']
            name = fut['name']
            expiry = fut['expiry']
            
            # Parse expiry
            try:
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                
                # Highlight current month
                if expiry_date > today:
                    marker = "👉 CURRENT"
                    print(f"{instrument_key:<45} {name:<20} {expiry} {marker}")
                    
                    # Show this as the one to use
                    if i == 0 or (i > 0 and futures[i-1]['expiry'] < str(today)):
                        print(f"\n💡 USE THIS SYMBOL: {instrument_key}")
                        print(f"   Expiry: {expiry}")
                        print(f"   Name: {name}\n")
                else:
                    print(f"{instrument_key:<45} {name:<20} {expiry} (expired)")
            except:
                print(f"{instrument_key:<45} {name:<20} {expiry}")
        
        if len(futures) > 10:
            print(f"\n... and {len(futures) - 10} more contracts")
        
        print("-" * 85)
    
    async def search_all_indices(self):
        """Search all indices"""
        
        print("="*70)
        print("🔍 UPSTOX INSTRUMENT SEARCH")
        print("="*70)
        print(f"⏰ Time: {datetime.now(IST).strftime('%d-%b %I:%M:%S %p')}")
        print(f"🎯 Searching: {len(SEARCH_INDICES)} indices")
        print("="*70)
        
        # Check token
        if not UPSTOX_ACCESS_TOKEN:
            print("\n❌ ERROR: UPSTOX_ACCESS_TOKEN not set!")
            return
        
        print(f"\n✅ Token: {UPSTOX_ACCESS_TOKEN[:20]}...")
        
        async with aiohttp.ClientSession() as session:
            for index_name in SEARCH_INDICES:
                await self.search_futures(index_name, session)
                await asyncio.sleep(1)  # Rate limit
        
        print("\n" + "="*70)
        print("✅ SEARCH COMPLETE")
        print("="*70)
        print("\n💡 Copy the correct instrument keys from above")
        print("   Update your code with these exact symbols")
        print("="*70)

# ==================== MAIN ====================
async def main():
    """Main entry point"""
    searcher = InstrumentSearcher()
    await searcher.search_all_indices()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Search stopped")
    except Exception as e:
        print(f"\n\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
