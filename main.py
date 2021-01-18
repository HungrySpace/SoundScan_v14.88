import telebot
import audiosegment
import tensorflow as myTF
import numpy as np
import wave
import math
import scipy.signal as sg
import matplotlib.pyplot as plt


TELEGRAM_API_TOKEN = '1152345546:AAH0m8CDNkNwqNlnk19EY7qZrPbZlA3kK5c'
bot = telebot.TeleBot(TELEGRAM_API_TOKEN)


@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):  # Название функции не играет никакой роли
    print('text')
    bot.send_message(message.chat.id, message.text)


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
    myModel = myTF.keras.models.load_model("Examples/1/myModel.h5")

    wav_name = 'new_file.wav'
    # wav_name = 'Control.wav'
    # wav_name = 'Bad.wav'

    types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }

    def format_time(x, pos=None):
        global duration, nframes, k
        progress = int(x / float(nframes) * duration * k)
        mins, secs = divmod(progress, 60)
        hours, mins = divmod(mins, 60)
        out = "%d:%02d" % (mins, secs)
        if hours > 0:
            out = "%d:" % hours
        return out

    def format_db(x, pos=None):
        if pos == 0:
            return ""
        global peak
        if x == 0:
            return "-inf"

        db = 20 * math.log10(abs(x) / float(peak))
        return int(db)

    wav = wave.open(wav_name, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    # print(nchannels, sampwidth, framerate, nframes, comptype, compname)
    # duration = nframes / framerate
    # w, h = 800, 300
    # k = nframes/w/32
    # # k = nframes
    # DPI = 72
    # peak = 256 ** sampwidth / 2

    content = wav.readframes(nframes)
    samples = np.frombuffer(content, dtype=types[sampwidth])
    channel = samples[1::nchannels]
    print(channel)

    averageAmplitude = (sum(abs(channel)) / len(channel))*2
    print(averageAmplitude)

    i = 0
    listSamples = []
    while i < len(channel):
        # print(abs(channel[i]))
        if abs(channel[i]) > averageAmplitude:
            NewPosition = round(i + (framerate * 0.4))
            listSamples.append(np.array(channel[i:NewPosition], dtype=types[sampwidth]))
            i = NewPosition
        i += 1

    print(len(listSamples))
    NaturalFrequencies = []
    for sample in listSamples:
        frequencies, times, spectrogram = sg.spectrogram(sample, framerate, nfft=4096)
        # 2d spectrogramm -> 1d spectrogramm2
        i1size = spectrogram.shape[0]
        i2size = spectrogram.shape[1]
        spectrogram2 = np.zeros((i1size))

        i1 = 0
        for i1 in range(i1size):
            for i2 in range(i2size):
                spectrogram2[i1] = spectrogram2[i1] + spectrogram[i1, i2]
        #show 1d
        t = np.arange(0, i1size, 1)
        fig, ax = plt.subplots()
        ax.plot(t, spectrogram2)
        ax.grid()
        fig.savefig(wav_name + '_1d.png')
        # plt.show()
        # NaturalFrequencies.append([])
        # print(list(spectrogram2).index(max(spectrogram2[0:200])), list(spectrogram2).index(max(spectrogram2[300:500])), list(spectrogram2).index(max(spectrogram2[600:1000])), list(spectrogram2).index(max(spectrogram2[1100:1500])), list(spectrogram2).index(max(spectrogram2[1600:2000])))
        Value1 = list(spectrogram2).index(max(spectrogram2[0:200])) - 107.605
        Value2 = list(spectrogram2).index(max(spectrogram2[300:500])) - 447.965
        Value3 = list(spectrogram2).index(max(spectrogram2[600:1000])) - 713.425
        Value4 = list(spectrogram2).index(max(spectrogram2[1100:1500])) - 1354.31
        Value5 = list(spectrogram2).index(max(spectrogram2[1600:2000])) - 1785.205

        print(myModel.predict([[Value1, Value2, Value3, Value4, Value5]]))
        jopa = myModel.predict([[Value1, Value2, Value3, Value4, Value5]])
        bot.send_message(message.chat.id, str(jopa) + "ответ")
        bot.send_photo(message.chat.id, open(wav_name + '_1d.png', 'rb'))
        print('finish one')


if __name__ == '__main__':
    bot.infinity_polling()
