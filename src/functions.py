import streamlit as st
from PIL import Image
from streamlit_mic_recorder import speech_to_text
from deepface import DeepFace
import tensorflow as tf
import spotipy
import pytesseract
import spotipy
import numpy as np
from io import BytesIO
from spotipy.oauth2 import SpotifyClientCredentials
import os
import random
from llama_cpp import Llama
from langchain_core.prompts import PromptTemplate
import sys
import os
from contextlib import contextmanager
from dotenv import load_dotenv


load_dotenv()

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET")
))
def text_input():
    return st.text_input("Free text (A sentence or feeling or emotion)")
def voice_input():
    st.markdown("Record a short message")
    def callback():
        if st.session_state.my_stt_output:
            st.write(st.session_state.my_stt_output)
    return speech_to_text(key='my_stt', callback=callback)
def camera_input():
    st.write("Capture your live emotion")
    if st.toggle("Enable Camera"):
        st.session_state.camera_active = True
        st.write("Take a picture")
        return st.camera_input("Capture your emotion")
    
def uploaded_image():
    return st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])



@contextmanager
def suppress_output():
    # Redirect stdout and stderr
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return suppress_output()


def prompt_template(input_data):
    task_description= """
    Subject: Generating emotion based Spotify search text.

    Task: You are an intelligent assistant programmed to generate search text for playing music on Spotify based on user empathy. 
    Based on the emotions extracted from the user input, your goal is to suggest music that supports and empathyze the emotion.

    Information:
    - Read and understand the user empathy and convert it to an emotion.
    - Think & concentrate only on the emotion, state of mind, empathy and ignore rest of the context.
    - Learn from the examples provided and stay with the context and use case.
    - The output should be concise and directly usable as search text on Spotify, with no additional explanations or reasoning.
    - DO NOT add additional information to the result as this output will be used directly as a spotify search.
    - Stricly display only the final output. DO NOT print the input in the final output.

    Context: 
    This system is designed to generate Spotify search text that supports and empathizes with the user's emotional state, rather than merely matching it. 
    For instance, if a user expresses feelings of anger or frustration, the system should suggest calming or soothing music, rather than reinforcing
    the negative emotion. The goal is to provide music that helps the user manage or balance their emotions effectively. It's ideal 
    for personalized music recommendations
    """

    few_shot_examples="""
    Few Examples:
    Input: "I am sad today"
    Output: "Mood-boosting songs"
    Input: "I am feeling energetic"
    Output: "Upbeat songs"
    Input: "Today is my first presentation to wider audience"
    Output: "Calm and relaxing songs"
    """

    template_string = f"[INST] {task_description}\n{few_shot_examples}\n{input_data}\nOutput? [/INST]"

    prompt_template = PromptTemplate.from_template(
        template=template_string
    )

    formatted_prompt = prompt_template.format()
    return formatted_prompt


def emotion_extractor(question):
        llm = Llama(
            model_path="/Users/ajayvenkatesan/Documents/musicgenerator/model/mistral-7b-instruct-v0.2.Q4_0.gguf",
            n_gpu_layers=-1, 
            n_threads=8,
            n_ctx=1000,
        )
        response = llm(
            prompt_template(question),
            max_tokens=100,
            temperature=0, 
            top_k=1,
            top_p=0.2,
            #stop=["Q:", "\n"],
        )
        input_text = response['choices'][0]['text']
        return input_text.strip()
    

def face_capture(input):
    face_emotion = DeepFace.analyze(
        img_path=np.array(Image.open(input)),
        enforce_detection=False,
        actions=['emotion']
    )
    face = face_emotion[0]['dominant_emotion']
    face = emotion_extractor(face)
    return face

def parser(text_input=None, speech_text=None, camera_input=None, uploaded_image=None):
        if text_input or speech_text:
            return emotion_extractor(text_input or speech_text)
        elif camera_input:
            return face_capture(camera_input)
        elif uploaded_image:
            try:
                DeepFace.extract_faces(img_path=np.array(Image.open(uploaded_image)))
                face_cap = face_capture(uploaded_image)
                return face_cap
            except ValueError:
                image_input = pytesseract.image_to_string(Image.open(uploaded_image))
                text_class =  emotion_extractor(image_input)
                return text_class
        else:
            return "No valid input provided."
            
def song_url(input):
    results = spotify.search(q=input, type='playlist', limit=10)
    if results['playlists']['items']:
        playlists = results['playlists']['items']
        random.shuffle(playlists)
        playlist = playlists[0]
        url = playlist['external_urls']['spotify']
        return url 
    
