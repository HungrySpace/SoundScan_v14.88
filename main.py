import telebot
import audiosegment


TELEGRAM_API_TOKEN = '1152345546:AAH0m8CDNkNwqNlnk19EY7qZrPbZlA3kK5c'
bot = telebot.TeleBot(TELEGRAM_API_TOKEN)


@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):  # Название функции не играет никакой роли
    print('text')
    # bot.send_message(message.chat.id, message.text)


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('new_file.ogg', 'wb') as new_file:
         new_file.write(downloaded_file)

    audiosegment.converter = r"C:/ffmpeg/bin/ffmpeg.exe"
    audiosegment.ffprobe = r"C:/ffmpeg/bin/ffprobe.exe"
    sound = audiosegment.from_file("new_file.ogg")
    sound.export("new_file.wav", format="wav")


if __name__ == '__main__':
    bot.infinity_polling()