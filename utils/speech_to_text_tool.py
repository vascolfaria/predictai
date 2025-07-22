from langchain.tools import BaseTool
import openai

class SpeechToTextTool(BaseTool):
    name = "speech_to_text"
    description = "Transcribes an audio file to text using Whisper."


def _run(self, file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)
    return response["text"]

async def _arun(self, file_path: str) -> str:
    return self._run(file_path)