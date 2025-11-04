import telebot, pandas as pd, feedparser, requests, os, time, threading
from datetime import datetime

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
bot = telebot.TeleBot(TOKEN)

# ====== FII/DII (NSDL - 7 PM ला येतं) ======
def get_fiidii():
    try:
        df = pd.read_html("https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx")[0]
        fii = df.iat[0,3]
        dii = df.iat[0,4]
        return f"💰 FII: *₹{fii:,.0f} Cr*\n🏦 DII: *₹{dii:,.0f} Cr*"
    except:
        return "⏳ FII/DII येतंय... 7:30 नंतर बघ"

# ====== Sensex/Nifty (Yahoo - 24×7) ======
def get_indices():
    try:
        r = requests.get("https://query1.finance.yahoo.com/v7/finance/quote?symbols=^BSESN,^NSEI").json()
        sx = r['quoteResponse']['result'][0]['regularMarketPrice']
        sp = r['quoteResponse']['result'][0]['regularMarketChangePercent']
        nf = r['quoteResponse']['result'][1]['regularMarketPrice']
        np = r['quoteResponse']['result'][1]['regularMarketChangePercent']
        return f"📈 Sensex: *{sx:,.0f}* ({sp:+.2f}%)\n📊 Nifty: *{nf:,.0f}* ({np:+.2f}%)"
    except:
        return "📊 Indices लोड होतंय..."

# ====== न्यूज ======
def get_news():
    feed = feedparser.parse("https://www.moneycontrol.com/news/rss")
    msg = "📰 *टॉप ३ न्यूज*\n\n"
    for e in feed.entries[:3]:
        msg += f"• {e.title[:80]}...\n🔗 {e.link}\n\n"
    return msg

# ====== कमांड्स ======
@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "Bot सुपरफास्ट झाला! 🚂\n/fiidii\n/sensex\n/news")

@bot.message_handler(func=lambda m: True)
def all(m):
    cmd = m.text.lower()
    if "fii" in cmd: bot.reply_to(m, get_fiidii(), parse_mode='Markdown')
    elif "sensex" in cmd or "nifty" in cmd: bot.reply_to(m, get_indices(), parse_mode='Markdown')
    elif "news" in cmd: bot.reply_to(m, get_news())

# ====== रोज 7:55 PM ऑटो मेसेज ======
def daily():
    while True:
        now = datetime.now()
        if now.hour == 19 and now.minute == 55:
            msg = f"🌟 *आजचा अपडेट* ({now.strftime('%d %b')})\n\n"
            msg += get_fiidii() + "\n\n"
            msg += get_indices() + "\n\n"
            msg += get_news()
            bot.send_message(CHAT_ID, msg, parse_mode='Markdown', disable_web_page_preview=True)
            print("7:55 PM चा मेसेज पाठवला!")
            time.sleep(70)
        time.sleep(20)

# ====== चालू कर (एकदाच!) ======
if __name__ == "__main__":
    print("Bot LIVE! फक्त Railway वर चालवा!")
    threading.Thread(target=daily, daemon=True).start()
    bot.infinity_polling(none_stop=True, interval=0, timeout=20)
