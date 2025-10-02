import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas: Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- NLTK Resource Download ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet')

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fake_news_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return model, tfidf
    except FileNotFoundError:
        st.error("Model or TF-IDF vectorizer not found.")
        return None, None

model, tfidf = load_assets()

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# --- Streamlit UI ---
st.title("Veritas: Fake News Detector")
st.write("Enter a news article below to determine if it's likely True or Fake.")

user_input = st.text_area("Enter news text here:", height=250)

if st.button("Analyze News"):
    if not model or not tfidf:
        st.warning("Model and vectorizer are not loaded. Cannot proceed.")
    elif not user_input.strip():
        st.warning("Please enter some news text to analyze!")
    else:
        with st.spinner("🕵️‍♀️ Analyzing the article..."):
            time.sleep(1)  # optional UX delay
            clean_text = preprocess(user_input)
            vect_text = tfidf.transform([clean_text])

            # --- Prediction ---
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]

            # --- Correct mapping based on model.classes_ ---
            if hasattr(model, "classes_"):
                if model.classes_[0] == 0:
                    result_class = "true" if prediction == 0 else "fake"
                    result_text = "True News" if prediction == 0 else "Fake News"
                else:
                    result_class = "true" if prediction == 1 else "fake"
                    result_text = "True News" if prediction == 1 else "Fake News"
            else:
                result_class = "fake" if prediction == 1 else "true"
                result_text = "Fake News" if prediction == 1 else "True News"

            confidence = prob[prediction] * 100

            # --- Display Results ---
            st.markdown(f"""
            <div style="
                padding:25px;
                border-radius:15px;
                margin-top:30px;
                text-align:center;
                box-shadow:0 10px 30px rgba(0,0,0,0.3);
                border:1px solid;
                background-color:{'#0a2a1e' if result_class=='true' else '#2a0a1f'};
                border-color:{'#00FFA3' if result_class=='true' else '#ff4d4d'};
            ">
                <div style="
                    font-size:2.5rem;
                    font-weight:700;
                    color:{'#00FFA3' if result_class=='true' else '#ff4d4d'};
                    margin-bottom:10px;">
                    This news is likely {result_text}
                </div>
                <div style="font-size:1.1rem; color:#a0a0a0;">
                    Confidence: <strong>{confidence:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- Top Keywords ---
            try:
                feature_names = tfidf.get_feature_names_out()
                vect_array = vect_text.toarray()[0]
                top_indices = vect_array.argsort()[-10:][::-1]
                top_keywords = [feature_names[i] for i in top_indices if vect_array[i] > 0]
                if top_keywords:
                    st.markdown(f"""
                    <div style="
                        margin-top:20px;
                        padding:15px;
                        background-color:rgba(0,0,0,0.2);
                        border-radius:10px;">
                        <strong>Top Keywords Influencing Prediction:</strong><br>
                        "<em>{'", "'.join(top_keywords)}</em>"
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not extract keywords: {e}")

st.markdown("<br><hr><center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)

