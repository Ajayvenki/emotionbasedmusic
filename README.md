# Emotion based music recommendation system 🎹🥁
Generate a random music playlist 🎵 on Spotify 🎧 with music recommendations that empathize with your emotions or state of mind 🧠. This Generative AI 🤖 project created for fun 🤩 which perceives human emotions and recommends a tailored playlist.

## About:
It is an intelligent solution designed to analyze human emotions, either through explicit input (such as text descriptions) or implicit signals (like facial expressions or voice tone). Once the system identifies the emotional state, it generates tailored music recommendations from platforms like Spotify. 

The goal is to either enhance the feelings (for positive emotions) or provide calming, uplifting, or supportive music (for negative emotions). Although it does not add any potential values to the business or solves a real time problems. It was fun to create a project based on generative AI models and concepts

![compressed](https://github.com/user-attachments/assets/e75a0dca-c17c-477e-a0e7-ffd5c7dbe488) 
![compressed02](https://github.com/user-attachments/assets/1cd285df-a813-41e0-958a-5a4b12bf22c9)


## Key Note:
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
| `Model` | Open source [model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): `Mistral 7b`. (a) Free to use (b) GGUF Quantized model Q4 for CPU inference (c) Accurate results compared to llama 7b (d) Reasonably efficient & expected output (e) Advanced models are overhead for in house build|
| `llama-cpp-python` | Main use of llama-cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware locally. [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) is a python binding on llama.cpp. As the solution is hosted locally, llama.cpp will extensively use the available cpu/gpu to generate a search.
| `deepface` | [Deepface](https://github.com/serengil/deepface) is an facial recognition system which process & analyse the image to retrieve age, gender, emotion etc. It is Open source and widely used across different real time systems. Choice of this library is ease of use and high accuracy. Though, cloud based services provide a higher accuracy. It comes with additional cost |
|`pytesseract`| A python wrapper for Google tesseract OCR engine. Major advantage falls on the accuracy, flexibility from various sources |
| `spotify` | Choice of spotify over youtube is because of free api access and the choice of available playlist. Youtube API comes with additional cost on Google projects |
| `spotipy` | A python library which integrates and extracts the spotify music based on api requests. Spotify provides a SDK for development. Usage of the SDK's, will add complexities to enable javascript on the user interface, where streamlit is not meant for |
| `Langchain` | Langchain is an AI framework which is used to create an application with LLM's. In this use case, Langchain is used to create & formalize the prompt templates |

## Setup:
* Clone & Install the requirements.
* Create a spotify [credential](https://developer.spotify.com/dashboard) for authentication
* Run the streamlit application

## Challenges:
It would have been a much simpler approach if the solution was hosted on a cloud or a hybrid cloud (As we pay for each inference and models). But, the whole aim was to explore the capabilities of the generative ai on a limited resources, it was way challenging to implement the solution. Jotting down few significant one:

- Bug on Ctransformers : My initial thoughts were to use ctransformers instead of llama.cpp due to its support on various model formats. But due to a [bug](https://github.com/marella/ctransformers/issues/211) encounter in the latest version of ctransformers, which stand unresolved i had to choose llama.cpp as a better alternative. Regardless of the ctransformer versions which are compatible with python 3.12, the bug exists.
  
- LlamaCpp via langchain : LlamaCpp is available as a wrapper in langchain which utilizes llama-cpp-python. For the ease of langchain ecosytem, i used the wrapper. But the performance was so poor which tool ~5 minutes to display the output. Therefore, llama-cpp-python was used directly.
  
- Compatiblity issues : As the set up includes a number of dependent libraries (ex: deepface requires transformers,pytorch etc). There was a conflict while installing one on the other.
  It required a reinstallation of few dependencies. requirements are up to date.
  
- Spotify SDK : Music play/pause using the SDK would have been a great option. But it puts on additional complexities to include javascript which is not meant for streamlit. Upon several attempts, a decision was made to embed the playlist url in the result.

## Could be expanded to:
* Music based on language, artist, region etc.
* Receive inputs and process in different languages.
* Capability to play pause on the website.
* Use other open source libraries like Magenta to compose a music.

## Caution ⚠️:
 - Generative AI is experimental and does not provide 💯% accuracy.
 - For the simpler use case, smaller model trained with 7 billion parameters was chosen. This may not provide accurate results everytime. However, the model has been tamed to a greater extent with prompt engineering techniques to give the expected results.
 - Fine tuning may be an overkill as the number of examples are too low.
 - Passing on too many examples in a prompt would result in severe latency.

## Demo:

https://github.com/user-attachments/assets/bebc128f-6f4d-43c7-8f6f-06cc8c0d2117


