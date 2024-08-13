import streamlit as st
from PIL import Image
from functions import text_input, voice_input, camera_input, uploaded_image, parser, song_url

st.title("Emotion based music generator 🥁")
st.markdown("#### Listen to a playlist in spotify based on your mood or emotion 😃")
st.markdown("##### Register your state of mind through a text/audio/image/attachment")
# Text input
text_input = text_input()
# voice input
speech_text = voice_input()
# camera input
camera_input = camera_input()
# Upload input
uploaded_image = uploaded_image()
if uploaded_image:
    conv_uploaded_image = Image.open(uploaded_image)
    st.image(conv_uploaded_image, caption="Uploaded Image", use_column_width=True)
else:
    image = None
# Submit button
if st.button("Submit"):
    if not text_input and not speech_text and not camera_input and not uploaded_image:
        st.error("No inputs received! Please key in your emotion in one form")
    elif sum([bool(text_input), bool(speech_text), bool(camera_input), bool(uploaded_image)]) > 1:
        st.error('Multiple inputs received ! Kindly register in one medium')
    else:
        st.success("Input captured!")
        st.write("### Parsing the input to extract the emotion")
        result = parser(text_input=text_input, speech_text=speech_text, camera_input=camera_input, uploaded_image=uploaded_image)
        st.markdown(f"Recommending {result} based on your emotion")
        songs = song_url(result)
        if songs:
            st.markdown(f"Enjoy your [music playlist]({songs}) on spotify")