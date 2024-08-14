# Emotion based music generator 🎹🥁
Generate a random music playlist 🎵 on Spotify 🎧 with music recommendations that empathize with your emotions or state of mind 🧠. This Generative AI 🤖 project, created for fun 🤩, perceives user emotions and recommends a tailored playlist.


![image](https://github.com/user-attachments/assets/20f99707-3e5d-4f75-9d9f-6fb504ec4dca)

![image](https://github.com/user-attachments/assets/4db39657-a8e7-4930-b02f-acff95898778)


## Use case: 
Although it does not add any potential values to the business or solves a real time problems. It was fun to create a project based on generative AI models and concepts:

    - In-House built 💼.
    - Private and secure 🔒.
    - Zero 0️⃣ cost 💸 on the inference. 
    - Captures the emotion in various mediums (Text, voice, image, attachment)
    - Runs based on open source models🤖 (Mistal-7B)
    - Capable to extract emotions from hand written ✍🏼 images via OCR.

To keep the solution very simple, the music generation does not honour the languages, choice of music, artist selection etc.

## Technical specifications:
    * Language: Python 🐍
    * Model: mistral-7b-instruct-v0.2 (GGUF - Q4)
    * CPU Inference: @llama-cpp-python
    * Image processing: @deepface 
    * OCR: @pytesseract
    * AI framework: Langchain
    * User Interface: Streamlit
    * Music provider: Spotify 🎶
    * Integration library: spotipy (calls spotify api)
    
## Design choice:
As the solution aims to generate the music recommendations with local inference. The components were chosen deligently to provide an efficient search result with minimal resource availability and at no cost.

| Components    | Reason for use |
| ------------- | ------------- |
| `Python`        | Easy & simple for this use case  |
| `Streamlit` | Python native ecosystem, quick development for a straigh forward task  |
| `Model` | Open source model: `Mistral 7b`. (a) Free to use (b) GGUF Quantized model Q4 for CPU inference (c) Feasible for local development (d) Reasonably efficient & expected output|
| `llama-cpp-python` | Main use of llama-cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware locally. `llama-cpp-python is a python binding on llama.cpp. As the solution is hosted locally, llama.cpp will extensively use the available cpu/gpu to generate a search.
| `deepface` | Deepface is an facial recognition system which process & analyse the image to retrieve age, gender, emotion etc. It is Open source and widely used across different real time systems. Choice of this library is ease of use and high accuracy. Though, cloud based services provide a higher accuracy. It comes with additional cost |
|||

## Challenges:
## Improvement areas:
