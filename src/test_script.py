# Use a pipeline as a high-level helper
'''from transformers import pipeline

pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device='cpu', top_k=3)
sentences = ["I love your character, but i totally hate you"]

model_outputs = pipe(sentences)

labels = [item['label'] for sublist in model_outputs for item in sublist if item['score'] > 0.5]

# Print the result
if labels:
    for label in labels:
        print(label)
else:
    print("Neutral")'''



'''from deepface import DeepFace
from PIL import Image
import base64
from io import BytesIO
import numpy as np

img_path='/Users/ajayvenkatesan/Documents/musicgenerator/Archive/images.jpeg'
try:
    face = DeepFace.extract_faces(img_path='/Users/ajayvenkatesan/Documents/musicgenerator/Archive/images.jpeg')
    face_emotion = DeepFace.analyze(
                img_path=np.array(Image.open(img_path)),
                actions=['emotion']
            )
    print (face_emotion[0]['dominant_emotion'])
except ValueError:
    print("No face detected")'''





'''import pytesseract
from PIL import Image

# Simple image to string
string = pytesseract.image_to_string(Image.open('/Users/ajayvenkatesan/Documents/musicgenerator/tempscripts/images (1).jpeg'))

print(pytesseract.get_languages(string))'''
'''
from transformers import pipeline

pipe = pipeline("text2text-generation", model="t5-small", device='cpu')

def generate_with_template(input_text):
    template = f"""
    Task: Provide a brief, intelligent response to the following input.
    Input: {input_text}
    Response:
    """
    return pipe(template, max_length=100)#[0]['generated_text']

# Example usage
input_text = "What are the benefits of exercise?"
response = generate_with_template(input_text)
print(response)
'''
'''
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_search_text(mood):
    # Create the prompt
    prompt = (f"Generate a Spotify search query for music based on the given mood.\n\n"
              f"Mood: {mood}\nSearch query:")

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the output
    outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)

    # Decode and return the result
    search_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return search_text

# Example usage
mood = "Surprised"
search_text = generate_search_text(mood)
print(search_text)
'''

'''
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_search_text(mood):
    # Load pre-trained model and tokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Define the prompt based on the mood
    prompt = (f"You are a helpful assistant who generates Spotify search queries based on moods or emotions. "
              "Your task is to suggest a type of music that matches the given mood.\n\n"
              "For the following moods, generate an appropriate search query to find music on Spotify:\n\n"
              "Examples:\n"
              "- If the mood is 'Sad', a suitable search query might be 'uplifting songs' or 'happy songs'.\n"
              "- If the mood is 'Happy', a suitable search query might be 'energetic songs' or 'party music'.\n\n"
              "For the given mood, provide the search query:\n\n"
              f"Mood: {mood}\nSearch for:")
    # Encode the prompt and generate the output
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)

    # Decode and return the result
    search_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return search_text

# Example usage
mood = "anxious"
search_text = generate_search_text(mood)
print(search_text)
'''

'''
from numbers_parser import Document

def emotion_retriever(input):
    doc = Document("/Users/ajayvenkatesan/Documents/musicgenerator/model/source_excel.numbers")
    table = doc.sheets[0].tables[0]
    for row in table.rows():
        if row[0].value == input:
            return row[1].value
        
    return None

input='Joy'
em = emotion_retriever(input)
if em:
    print(em)
else:
    print("No words")'''

'''
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Spotipy client with your credentials
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
    client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET")
))

# Search for playlists that match the query "Happy songs"
query = "High-energy and party music"
results = spotify.search(q=query, type='playlist', limit=1)


# Extract and display the playlist information
if results['playlists']['items']:
    playlist = results['playlists']['items'][0]
    print(f"Playlist Name: {playlist['name']}")
    print(f"Playlist URL: {playlist['external_urls']['spotify']}")
    print(f"Playlist URI: {playlist['uri']}")
else:
    print("No playlists found.")
'''


'''
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
template = """Question: {question}

Answer: You are an AI assistant who write poem"""

prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/ajayvenkatesan/Documents/musicgenerator/model/llama-2-7b.Q2_K.gguf",
    temperature=0,
    max_tokens=512,
    use_gpu=True,  # Ensure this flag is set to true
    gpu_backend='metal',  # Specify the backend as 'metal' to use Apple's Metal API
    gpu_id=0,  # If you have multiple GPUs, specify the one you want to use
    n_threads=8,  # Adjust the number of threads as per your system’s capabilit
    n_gpu_layers=-1,  # Use all GPU layers
    n_batch=512,     # Adjust batch size for optimal performance
    f16_kv=True,
    top_p=1,
    callback_manager=callback_manager,
    n_ctx = 512
)

question = """
Question: Write some poem about a moon
"""
llm.invoke(question)

'''

from llama_cpp import Llama
from langchain_core.prompts import PromptTemplate
import sys
import os
from contextlib import contextmanager

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
    Subject: Extract Emotion and Suggest Calming or Supportive Spotify Search Text
    System Prompt: You are an intelligent assistant programmed to generate search text for playing music on Spotify. Based on the emotions extracted from the input sentences, your goal is to suggest music that either calms or supports the user. For instance, if the user is frustrated, recommend calming music. If no clear emotion is detected, suggest neutral songs. Provide only the search text without any additional reasoning or explanation.
    """
    few_shot_examples="""
    Few Examples:
    Human: "I am sad today"
    AI: "Happy Songs"
    Human: "I am feeling energetic"
    AI: "Upbeat workout songs"
    Human: "I am very relaxed"
    AI: "Relaxing and peaceful music"
    Human: "Cant wait to see her"
    AI: "High-energy and party songs"
    Human: "Today is my first presentation to wider audience"
    AI: "Calm and relaxing songs"
    Human: "What the hell is this. How could this go wrong"
    AI: "Soothing and calming music"
    """

    template_string = f"[INST] {task_description}\n{few_shot_examples}\n{input_data}\nOutput? [/INST]"

    prompt_template = PromptTemplate.from_template(
        template=template_string
    )

    formatted_prompt = prompt_template.format()
    return formatted_prompt


def emotion_extractor(question):
    with suppress_output():
        llm = Llama(
            model_path="/Users/ajayvenkatesan/Documents/musicgenerator/model/mistral-7b-instruct-v0.2.Q4_0.gguf",
            n_gpu_layers=-1, 
            n_threads=8,
            n_ctx=500,
        )
        response = llm(
            prompt_template(question),
            max_tokens=50,
            temperature=0, 
            top_k=1,
            top_p=0.2,
            stop=["Q:", "\n"],
        )
        input_text = response['choices'][0]['text']
        return input_text.strip()

#question1 = "I’m grateful for all the support, yet there’s a gnawing resentment that I can’t do it on my own"
question = "Yes, finally we made it. i am top of the world"

search_text = emotion_extractor(question)
print(search_text)


