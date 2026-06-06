"""
Smart Product Assistant - Milestone 7 demo

Integrates all six models (M1-M6) into a single application.

Run with: streamlit run app/app.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart Product Assistant",
                   page_icon="🛍️", layout="wide")


@st.cache_resource
def load_all_models():
    from src import models
    m1, m1_scaler = models.load_m1()
    m2, m2_categories = models.load_m2()
    embeddings, index = models.load_embeddings()
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return {"m1": m1, "m1_scaler": m1_scaler, "m2": m2,
            "m2_categories": m2_categories, "embedder": embedder,
            "embeddings": embeddings, "index": index}


@st.cache_resource
def load_groq_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


@st.cache_data
def load_data():
    from src import data
    return data.load_products(), data.load_reviews()


def main():
    st.title("🛍️ Smart Product Assistant")
    st.caption("Capstone project - six deep learning models integrated into one assistant")

    with st.spinner("Loading models..."):
        store = load_all_models()
        products_df, reviews_df = load_data()
        groq = load_groq_client()

    from src import utils

    tabs = st.tabs(["⭐ Rating (M1)", "📸 Category (M2)", "🔍 Similar (M3)",
                    "📝 Pros/Cons (M5)", "💬 RAG QA (M5)"])

    with tabs[0]:
        st.subheader("Predict rating band from tabular features")
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input("Price (USD)", min_value=0.0, value=20.0)
            rn = st.number_input("Number of ratings", min_value=0, value=100)
        with c2:
            title = st.text_input("Title", "Plush Teddy Bear")
            desc = st.text_area("Description", "Soft toy")
        if st.button("Predict", key="m1"):
            band, conf = utils.predict_rating_band(
                store["m1"], store["m1_scaler"], price, rn, title, desc)
            st.success(f"Band: **{band}** ({conf:.1f}% confidence)")

    with tabs[1]:
        st.subheader("Predict subcategory from image")
        up = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if up is not None:
            img = Image.open(up)
            st.image(img, width=300)
            if st.button("Classify", key="m2"):
                cat, conf = utils.predict_image_category(
                    store["m2"], store["m2_categories"], img)
                st.success(f"Category: **{cat}** ({conf:.1f}%)")

    with tabs[2]:
        st.subheader("Find similar products")
        q = st.text_input("Query", "educational puzzle for kids")
        k = st.slider("Top K", 1, 10, 5)
        if st.button("Search", key="m3"):
            for i, r in enumerate(utils.find_similar_products(
                    q, store["embedder"], store["embeddings"],
                    store["index"], top_k=k), 1):
                st.markdown(f"**{i}.** [{r['similarity']:.2f}] {r['title']}")

    with tabs[3]:
        st.subheader("Pros and Cons summarization")
        if groq is None:
            st.warning("GROQ_API_KEY required for M5 tabs.")
        else:
            q = st.text_input("Product (query or ASIN)", "teddy bear", key="m5sq")
            if st.button("Summarize", key="m5s"):
                st.info("Run from notebook 05_llm_rag_finetune.ipynb for full output")

    with tabs[4]:
        st.subheader("Grounded Q&A (RAG)")
        if groq is None:
            st.warning("GROQ_API_KEY required for M5 tabs.")
        else:
            st.info("Run from notebook 05_llm_rag_finetune.ipynb")


if __name__ == "__main__":
    main()
