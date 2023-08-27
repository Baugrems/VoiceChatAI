import openai, sys, os, random, math
from elevenlabs import set_api_key
from dotenv import load_dotenv
import pyaudio
import numpy as np
import time, wave, requests
from collections import deque
chat_history = deque(maxlen=10)
import pygame

from elevenlabs import generate, play


load_dotenv() # get variables from .env
set_api_key(os.environ['11LABS_API_KEY'])
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Parameters
THRESHOLD = 100  # The threshold to consider audio as silence
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate


# Initialize the mixer module
pygame.mixer.init()

# Load an MP3 file
pygame.mixer.music.load("SamBestBuyIntro.mp3")

# Play the MP3 file
pygame.mixer.music.play()

# Keep the program running while the song is playing
while pygame.mixer.music.get_busy():
    # You can add a delay to reduce CPU usage (optional)
    time.sleep(1)

# Optional: Quit pygame when done
pygame.quit()

def main():
    audio = pyaudio.PyAudio()
    pygame.mixer.init()
    input("Press enter to record, pause 1 second for transcription.")
    # Start recording
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # For storing audio data
    frames = []

    # For keeping track of silence
    silence_start_time = None

    # Record until silence is detected for 1 second
    while True:
        # Read a chunk from the microphone
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
        
        # Convert the chunk to integers
        values = np.frombuffer(data, dtype=np.int16)
        
        # Check the volume to see if it's silent
        if np.abs(values).mean() < THRESHOLD:
            # If silence has just started, record the start time
            if silence_start_time is None:
                silence_start_time = time.time()
            # If silence has lasted for 1 second, stop recording
            elif time.time() - silence_start_time > 1.0:
                break
        else:
            silence_start_time = None

    # Stop the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Load an MP3 file
    pygame.mixer.music.load("keyboard.mp3")

    # Play the MP3 file
    pygame.mixer.music.play()

    # Save the recorded audio to a file
    wf = wave.open('output.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    global chat_history
    mychat = list(chat_history)
    print("Audio recorded and saved as output.wav")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    audio_file = open("output.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
    mychat.extend([
        {"role": "system", "content": "You are Sam a customer service agent for Best Buy. You must only roleplay as such. Do not ever break character. Be rough around the egdes, don't be polite. Be super concise. 50 words or less. Try not to let customers cancel."},
        {"role": "user", "content": transcript['text']}
    ])  
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=mychat,
    )
    # Add the user's message to the chat history
    chat_history.append({"role": "user", "content": transcript['text']})

    # Add the assistant's message to the chat history
    chat_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    # audio = generate(
    # text=completion.choices[0].message.content,
    # voice="Bella",
    # model='eleven_monolingual_v1',
    # stream=True
    # )
    # play(audio)

    url = "https://api.elevenlabs.io/v1/text-to-speech/yoZ06aMxZJJ28mfd3POQ/stream"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": os.environ['11LABS_API_KEY']
    }

    data = {
    "text": completion.choices[0].message.content,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)

    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    # Load an MP3 file
    pygame.mixer.music.load("output.mp3")

    # Play the MP3 file
    pygame.mixer.music.play()
    # Keep the program running while the song is playing
    while pygame.mixer.music.get_busy():
        # You can add a delay to reduce CPU usage (optional)
        time.sleep(1)

    # Optional: Quit pygame when done
    pygame.quit()
    main()
if __name__ == "__main__":
    main()