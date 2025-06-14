import whisper
from TTS.api import TTS as CoquiTTS
import sounddevice as sd
import numpy as np
import queue
import wave
import tempfile
import pyttsx3

from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# === AUDIO CONFIG ===
samplerate = 16000
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def record_audio(duration=5):
    print("🎙️ Listening...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
            audio = np.empty((0, 1), dtype=np.float32)
            for _ in range(int(samplerate / 1024 * duration)):
                audio = np.vstack((audio, audio_queue.get()))
        with wave.open(f.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        return f.name

# === STT USING WHISPER ===
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# === TTS USING CoquiTTS ===
# Load once globally for performance
coqui_tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def speak(text):
    print(f"🤖 Assistant: {text}")
    # Generate waveform
    audio = coqui_tts.tts(text)
    # Play audio
    sd.play(audio, samplerate=22050)
    sd.wait()

# === ROVER ASSEMBLY STEPS ===
steps = [
    "Unbox all parts",
    "Attach front wheel",
    "Install handlebars",
    "Secure seat",
    "Attach pedals",
    "Check brakes",
    "Pump tires if needed",
    "Test the rover"
]
state = {"step": 0}

# === LANGCHAIN WITH OLLAMA ===
llm = Ollama(model="llama3")
memory = ConversationBufferMemory()
chat_chain = ConversationChain(llm=llm, memory=memory)

def get_funny_response(user_input):
    current_step = steps[state["step"]]
    prompt = f"""
You are a funny and helpful rover-assembly assistant. You're helping a user through step-by-step instructions.
Current step: {current_step}.
User said: "{user_input}"

Respond with humor and emotional intelligence. If user is skipping steps or frustrated, acknowledge it and steer them back.
"""
    return chat_chain.predict(input=prompt)

# === MAIN INTERACTION LOOP ===
def run_assistant():
    speak(
        "I'm the Frenemy Reconnaissance Explorer Navigating Environments, "
        "Malfunctioning & Yelling. I was first assembled in 1990 by the team at O.M.S.A. (Offworld Mechanics Sabotaging Algorithms). "
        "You're going to build me. Or... risk disappointing your comrades who have been waiting for me to be assembled for... "
        "Calculating... Calculating... Holy shit. 35 years. Oh god. You don't really look like the type to pull this off."
        "But it seems you were at least able to dig my ass out of the ground. So I'll give you a chance."
        "You can make this fun or I can yell at you. It's up to you."
        "I'm going to give you 30 seconds to think about it. "
        "30... 29... 28... 27... 26... 25... 24... 23... 22... 21... 20... 19... 18... 17... 16... 15... 14... 13... 12... 11... 10... 9... 8... 7... 6... 5... 4... 3... 2... 1... 0... "
        "Time's up."
        "Which will it be?"
    )
    
    while state["step"] < len(steps):
        current_step = steps[state["step"]]
        speak(f"Step {state['step'] + 1}: {current_step}. Tell me when you're ready or if you need help.")

        audio_path = record_audio(6)
        user_text = transcribe_audio(audio_path)
        print(f"🗣️ You said: {user_text}")

        response = get_funny_response(user_text)
        speak(response)

        # Basic intent logic
        if any(x in user_text.lower() for x in ["already", "next", "done", "finished"]):
            state["step"] += 1
        elif "repeat" in user_text.lower():
            continue
        elif "skip" in user_text.lower():
            state["step"] += 1
            speak("Alright, but don't blame me if your seat ends up on your front tire.")

    speak("You did it! Your rover is ready. Go ride like the wind—or at least do not crash!")

# === RUN ===
if __name__ == "__main__":
    run_assistant()
