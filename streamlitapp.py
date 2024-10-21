import streamlit as st


enable = st.checkbox("Hows it?")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)


st.write("Hello World!!!, Lets say whether you will update changes or not in this app")