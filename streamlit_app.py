
import streamlit as st
import alphabet
import digit

st.sidebar.title("Navigation")
page = st.sidebar.radio("",["Alphabet Detection", "Digit Detection"])

if page == "Alphabet Detection":
    alphabet.app()
elif page == "Digit Detection":
    digit.app()
