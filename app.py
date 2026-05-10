import os
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

from huggingface_hub import InferenceClient
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

import streamlit as st
name = st.text_input("Enter your name: ")
#
if name:
	result = client.text_classification(
    name,
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
	st.write(f"Hello, {result}.")
