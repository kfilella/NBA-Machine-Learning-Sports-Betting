import requests
import os
from dotenv import load_dotenv

load_dotenv()

def enviar_mensaje_colores(mensaje):
    url = "https://api.telegram.org/bot" + os.getenv('TELEGRAM_BOT_TOKEN') + "/sendMessage"
    data = {
        "chat_id": os.getenv('TELEGRAM_CHAT_ID'),
        "text": mensaje
    }
    requests.post(url, json = data)