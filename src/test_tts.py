from dotenv import load_dotenv
load_dotenv()

from narration.tts import speak
path = speak("Testing one two three.")
print(path)