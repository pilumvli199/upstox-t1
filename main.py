import os
import logging
from datetime import datetime
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
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
            'Accept-Encoding': 'gzip, deflate, br',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_nse_cookies(self):
        """NSE cookies"""
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
            return True
        except Exception as e:
            logger.error(f"Cookie error: {e}")
            return False
    
    def get_fii_dii_data(self):
        """FII/DII data"""
        try:
            self.get_nse_cookies()
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = self.session.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.format_fii_dii_data(data)
            else:
                logger.error(f"FII/DII API returned: {response.status_code}")
                return "❌ FII/DII data सध्या उपलब्ध नाही\\.\nकृपया काही वेळाने पुन्हा प्रयत्न करा\\."
        except Exception as e:
            logger.error(f"FII/DII error: {e}")
            return f"❌ Error: Data मिळवताना समस्या\\.\n\nNSE website काही वेळा slow असते\\.\nकृपया पुन्हा प्रयत्न करा\\."
    
    def format_fii_dii_data(self, data):
        """Format FII/DII data with proper escaping"""
        try:
            if not data:
                return "📊 आजचा FII/DII data अजून उपलब्ध नाही\\."
            
            msg = "*💰 FII/DII Trading Data*\n\n"
            
            for item in data:
                category = item.get('category', 'N/A').replace('-', '\\-')
                buy_value = float(item.get('buyValue', 0))
                sell_value = float(item.get('sellValue', 0))
                net_value = float(item.get('netValue', 0))
                
                msg += f"*{category}*\n"
                msg += f"📈 Buy: ₹{buy_value:,.2f} Cr\n".replace(',', '\\,').replace('.', '\\.')
                msg += f"📉 Sell: ₹{sell_value:,.2f} Cr\n".replace(',', '\\,').replace('.', '\\.')
                
                if net_value > 0:
                    msg += f"✅ Net: \\+₹{abs(net_value):,.2f} Cr\n\n".replace(',', '\\,').replace('.', '\\.')
                else:
                    msg += f"⚠️ Net: \\-₹{abs(net_value):,.2f} Cr\n\n".replace(',', '\\,').replace('.', '\\.')
            
            timestamp = datetime.now().strftime('%d\\-%m\\-%Y %H:%M')
            msg += f"_Updated: {timestamp}_"
            return msg
        except Exception as e:
            logger.error(f"Format error: {e}")
            return "❌ Data format error"
    
    def get_market_news(self):
        """Market news - Multiple sources"""
        try:
            # Try Economic Times RSS
            news_items = []
            
            # Source 1: Economic Times Market News
            try:
                url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    for item in root.findall('.//item')[:5]:
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        
                        if title_elem is not None and title_elem.text:
                            news_items.append({
                                'title': title_elem.text.strip(),
                                'link': link_elem.text.strip() if link_elem is not None else ''
                            })
            except Exception as e:
                logger.error(f"ET RSS error: {e}")
            
            # Source 2: Business Standard
            if len(news_items) < 3:
                try:
                    url = "https://www.business-standard.com/rss/markets-106.rss"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(response.content)
                        
                        for item in root.findall('.//item')[:5]:
                            title_elem = item.find('title')
                            link_elem = item.find('link')
                            
                            if title_elem is not None and title_elem.text:
                                news_items.append({
                                    'title': title_elem.text.strip(),
                                    'link': link_elem.text.strip() if link_elem is not None else ''
                                })
                except Exception as e:
                    logger.error(f"BS RSS error: {e}")
            
            # Format news
            if news_items:
                msg = "*📰 Latest Market News*\n\n"
                for idx, news in enumerate(news_items[:5], 1):
                    # Escape special characters for MarkdownV2
                    title = news['title'].replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('~', '\\~').replace('`', '\\`').replace('>', '\\>').replace('#', '\\#').replace('+', '\\+').replace('-', '\\-').replace('=', '\\=').replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('.', '\\.').replace('!', '\\!')
                    
                    msg += f"{idx}\\. {title}\n\n"
                
                timestamp = datetime.now().strftime('%d\\-%m\\-%Y %H:%M')
                msg += f"_Updated: {timestamp}_"
                return msg
            else:
                return "*📰 Latest Market News*\n\nNews sources सध्या unavailable\\.\nकृपया काही वेळाने पुन्हा प्रयत्न करा\\."
        except Exception as e:
            logger.error(f"News error: {e}")
            return "*📰 Market News*\n\n❌ Error fetching news\\."
    
    def get_indices_data(self):
        """NSE indices data"""
        try:
            self.get_nse_cookies()
            url = "https://www.nseindia.com/api/allIndices"
            response = self.session.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.format_indices_data(data.get('data', []))
            else:
                logger.error(f"Indices API returned: {response.status_code}")
                return "❌ Indices data उपलब्ध नाही"
        except Exception as e:
            logger.error(f"Indices error: {e}")
            return "❌ Error fetching indices"
    
    def format_indices_data(self, indices):
        """Format indices data"""
        msg = "*📊 Market Indices*\n\n"
        
        key_indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT']
        
        for index in indices:
            if index.get('index') in key_indices:
                name = index.get('index').replace('-', '\\-')
                last = index.get('last', 0)
                change = index.get('percentChange', 0)
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                sign = "\\+" if change > 0 else "\\-"
                
                last_str = f"{last:,.2f}".replace(',', '\\,').replace('.', '\\.')
                change_str = f"{abs(change):.2f}".replace('.', '\\.')
                
                msg += f"{emoji} *{name}*\n"
                msg += f"Price: {last_str} ({sign}{change_str}%)\n\n"
        
        timestamp = datetime.now().strftime('%d\\-%m\\-%Y %H:%M')
        msg += f"_Updated: {timestamp}_"
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
        "*🇮🇳 Indian Stock Market Bot*\n\n"
        "स्वागत आहे\\! मी तुम्हाला देऊ शकतो:\n\n"
        "📊 FII/DII Trading Data\n"
        "📰 Latest Market News\n"
        "📈 Live Market Indices\n\n"
        "खालील buttons वापरा किंवा commands:\n"
        "/fii \\- FII/DII Data\n"
        "/news \\- Market News\n"
        "/indices \\- Market Indices"
    )
    
    await update.message.reply_text(
        welcome_msg, 
        reply_markup=reply_markup, 
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def fii_dii_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """FII/DII command"""
    msg = await update.message.reply_text("⏳ FII/DII data मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    data = market_bot.get_fii_dii_data()
    await msg.edit_text(data, parse_mode=ParseMode.MARKDOWN_V2)

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """News command"""
    msg = await update.message.reply_text("⏳ Latest news मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    news = market_bot.get_market_news()
    await msg.edit_text(news, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)

async def indices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Indices command"""
    msg = await update.message.reply_text("⏳ Market indices मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    data = market_bot.get_indices_data()
    await msg.edit_text(data, parse_mode=ParseMode.MARKDOWN_V2)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'fii_dii':
        await query.message.edit_text("⏳ FII/DII data मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        data = market_bot.get_fii_dii_data()
        await query.message.edit_text(data, parse_mode=ParseMode.MARKDOWN_V2)
    
    elif query.data == 'news':
        await query.message.edit_text("⏳ Latest news मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        news = market_bot.get_market_news()
        await query.message.edit_text(news, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
    
    elif query.data == 'indices':
        await query.message.edit_text("⏳ Market indices मिळवत आहे\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
        data = market_bot.get_indices_data()
        await query.message.edit_text(data, parse_mode=ParseMode.MARKDOWN_V2)
    
    elif query.data == 'help':
        help_msg = (
            "*ℹ️ Bot Commands*\n\n"
            "/start \\- Bot सुरू करा\n"
            "/fii \\- FII/DII Trading Data\n"
            "/news \\- Latest Market News\n"
            "/indices \\- Market Indices\n\n"
            "*📌 Features:*\n"
            "• Real\\-time FII/DII data\n"
            "• Latest market news\n"
            "• Live NSE indices\n"
            "• Free \\& No API keys needed"
        )
        await query.message.edit_text(help_msg, parse_mode=ParseMode.MARKDOWN_V2)

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
