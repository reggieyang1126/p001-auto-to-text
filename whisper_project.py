import whisper
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# input audio file name
audio_file = "/Users/reggie/Repo/Whisper/Ashley 10.mp3"

# load the model and transcribe the audio
# model = whisper.load_model("base")
logging.info('model start....')
model = whisper.load_model("large-v2")
result = model.transcribe(audio_file, fp16=False, language='English', word_timestamps=True)

# extract the text and language information
text = result["text"]
language = result["language"]

# create the output text file name based on the input mp3 file name
text_file = os.path.splitext(audio_file)[0] + ".txt"
print('model start....')

# write the text and language information to the output text file
with open(text_file, "w") as f:
    f.write(f"Text:\n\n{text}\n\nLanguage: {language}")

# print the text and language information to the console
print("Text:\n\n", text)
print("Language: ", language)
logging.info('model end....')
