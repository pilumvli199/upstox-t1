import os
import logging
from datetime import datetime
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')

class IndianMarketBot:
    def __init__(self):
        self.nse_base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_nse_cookies(self):
        """NSE cookies घेण्यासाठी"""
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
        except Exception as e:
            logger.error(f"Cookie error: {e}")
    
    def get_fii_dii_data(self):
        """FII/DII data मिळवा"""
        try:
            self.get_nse_cookies()
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.format_fii_dii_data(data)
            else:
                return "❌ FII/DII data सध्या उपलब्ध नाही."
        except Exception as e:
            logger.error(f"FII/DII error: {e}")
            return f"❌ Error: {str(e)}"
    
    def format_fii_dii_data(self, data):
        """FII/DII data format करा"""
        try:
            if not data:
                return "📊 आजचा FII/DII data अजून उपलब्ध नाही."
            
            msg = "💰 *FII/DII Trading Data*\n\n"
            
            # Latest date data
            for item in data:
                category = item.get('category', 'N/A')
                buy_value = float(item.get('buyValue', 0))
                sell_value = float(item.get('sellValue', 0))
                net_value = float(item.get('netValue', 0))
                
                msg += f"*{category}*\n"
                msg += f"📈 Buy: ₹{buy_value:,.2f} Cr\n"
                msg += f"📉 Sell: ₹{sell_value:,.2f} Cr\n"
                
                if net_value > 0:
                    msg += f"✅ Net: +₹{net_value:,.2f} Cr\n\n"
                else:
                    msg += f"⚠️ Net: ₹{net_value:,.2f} Cr\n\n"
            
            msg += f"_Updated: {datetime.now().strftime('%d-%m-%Y %H:%M')}_"
            return msg
        except Exception as e:
            logger.error(f"Format error: {e}")
            return "❌ Data format error"
    
    def get_market_news(self):
        """Market news मिळवा (multiple free sources)"""
        news_items = []
        
        # Source 1: MoneyControl RSS-style API
        try:
            url = "https://www.moneycontrol.com/rss/latestnews.xml"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Simple XML parsing (first 3 news items)
                import re
                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', response.text)
                links = re.findall(r'<link>(.*?)</link>', response.text)
                
                for i in range(min(3, len(titles)-1)):  # Skip first title (channel title)
                    if i+1 < len(titles) and i < len(links):
                        news_items.append({
                            'title': titles[i+1],
                            'link': links[i+1] if i+1 < len(links) else ''
                        })
        except Exception as e:
            logger.error(f"MoneyControl error: {e}")
        
        # Format news
        if news_items:
            msg = "📰 *Latest Market News*\n\n"
            for idx, news in enumerate(news_items, 1):
                msg += f"{idx}. {news['title']}\n"
                if news['link']:
                    msg += f"🔗 [Read More]({news['link']})\n\n"
                else:
                    msg += "\n"
            
            msg += f"_Updated: {datetime.now().strftime('%d-%m-%Y %H:%M')}_"
            return msg
        else:
            return "📰 *Latest Market Updates*\n\nकृपया थोड्या वेळाने पुन्हा प्रयत्न करा."
    
    def get_indices_data(self):
        """NSE indices data"""
        try:
            self.get_nse_cookies()
            url = "https://www.nseindia.com/api/allIndices"
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.format_indices_data(data.get('data', []))
            else:
                return "❌ Indices data उपलब्ध नाही"
        except Exception as e:
            logger.error(f"Indices error: {e}")
            return "❌ Error fetching indices"
    
    def format_indices_data(self, indices):
        """Format indices data"""
        msg = "📊 *Market Indices*\n\n"
        
        key_indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT']
        
        for index in indices:
            if index.get('index') in key_indices:
                name = index.get('index')
                last = index.get('last', 0)
                change = index.get('percentChange', 0)
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                sign = "+" if change > 0 else ""
                
                msg += f"{emoji} *{name}*\n"
                msg += f"Price: {last:,.2f} ({sign}{change:.2f}%)\n\n"
        
        msg += f"_Updated: {datetime.now().strftime('%d-%m-%Y %H:%M')}_"
        return msg

# Bot instance
market_bot = IndianMarketBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    keyboard = [
        [InlineKeyboardButton("📊 FII/DII Data", callback_data='fii_dii')],
        [InlineKeyboardButton("📰 Market News", callback_data='news')],
        [InlineKeyboardButton("📈 Indices", callback_data='indices')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_msg = (
        "🇮🇳 *Indian Stock Market Bot*\n\n"
        "स्वागत आहे! मी तुम्हाला देऊ शकतो:\n\n"
        "📊 FII/DII Trading Data\n"
        "📰 Latest Market News\n"
        "📈 Live Market Indices\n\n"
        "खालील buttons वापरा किंवा commands:\n"
        "/fii - FII/DII Data\n"
        "/news - Market News\n"
        "/indices - Market Indices"
    )
    
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def fii_dii_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """FII/DII command"""
    msg = await update.message.reply_text("⏳ FII/DII data मिळवत आहे...")
    data = market_bot.get_fii_dii_data()
    await msg.edit_text(data, parse_mode='Markdown')

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """News command"""
    msg = await update.message.reply_text("⏳ Latest news मिळवत आहे...")
    news = market_bot.get_market_news()
    await msg.edit_text(news, parse_mode='Markdown', disable_web_page_preview=True)

async def indices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Indices command"""
    msg = await update.message.reply_text("⏳ Market indices मिळवत आहे...")
    data = market_bot.get_indices_data()
    await msg.edit_text(data, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'fii_dii':
        await query.message.edit_text("⏳ FII/DII data मिळवत आहे...")
        data = market_bot.get_fii_dii_data()
        await query.message.edit_text(data, parse_mode='Markdown')
    
    elif query.data == 'news':
        await query.message.edit_text("⏳ Latest news मिळवत आहे...")
        news = market_bot.get_market_news()
        await query.message.edit_text(news, parse_mode='Markdown', disable_web_page_preview=True)
    
    elif query.data == 'indices':
        await query.message.edit_text("⏳ Market indices मिळवत आहे...")
        data = market_bot.get_indices_data()
        await query.message.edit_text(data, parse_mode='Markdown')
    
    elif query.data == 'help':
        help_msg = (
            "ℹ️ *Bot Commands*\n\n"
            "/start - Bot सुरू करा\n"
            "/fii - FII/DII Trading Data\n"
            "/news - Latest Market News\n"
            "/indices - Market Indices\n\n"
            "📌 *Features:*\n"
            "• Real-time FII/DII data\n"
            "• Latest market news\n"
            "• Live NSE indices\n"
            "• Free & No API keys needed"
        )
        await query.message.edit_text(help_msg, parse_mode='Markdown')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """Main function"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("❌ Error: कृपया TELEGRAM_BOT_TOKEN set करा!")
        print("Bot token मिळवा: https://t.me/BotFather")
        return
    
    print("🤖 Bot starting...")
    
    # Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("fii", fii_dii_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("indices", indices_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)
    
    # Start bot
    print("✅ Bot चालू आहे!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
