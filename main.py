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
# It's good practice to handle potential download errors.
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK resources (stopwords)...")
    nltk.download('stopwords')

try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    st.info("Downloading NLTK resources (wordnet)...")
    nltk.download('wordnet')


# --- Load Model and Vectorizer ---
# Use a spinner for a better user experience during loading.
# IMPORTANT: Make sure these .pkl files are in the same directory as your app.py file.
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load("fake_news_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        return model, tfidf
    except FileNotFoundError:
        st.error("Model or TF-IDF vectorizer not found. Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

model, tfidf = load_assets()


# --- Text Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Cleans and preprocesses the input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You would create a style.css file for this, but for a single file app, we embed it.
st.markdown("""
<style>
/* --- General Styles --- */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #0E1117;
}

/* --- Main App Styling --- */
.stApp {
    background-image: linear-gradient(135deg, #0d1226 0%, #1f2c4d 100%);
    color: #FFFFFF;
}

/* --- Title --- */
h1 {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #00FFA3, #00BFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 20px;
}

/* --- Text Area --- */
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.05);
    border: 2px solid #00BFFF;
    border-radius: 10px;
    color: #FFFFFF;
    font-size: 1.1rem;
    height: 250px;
    box-shadow: 0 4px 15px rgba(0, 191, 255, 0.2);
    transition: all 0.3s ease-in-out;
}

.stTextArea textarea:focus {
    border-color: #00FFA3;
    box-shadow: 0 4px 20px rgba(0, 255, 163, 0.3);
}

/* --- Button --- */
.stButton>button {
    width: 100%;
    padding: 15px;
    font-size: 1.2rem;
    font-weight: 700;
    color: white;
    background-image: linear-gradient(45deg, #00BFFF, #00FFA3);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 225, 200, 0.4);
}

/* --- Result Card --- */
.result-card {
    padding: 25px;
    border-radius: 15px;
    margin-top: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 1px solid;
}

.result-card-fake {
    background-color: #2a0a1f; /* Dark red background */
    border-color: #ff4d4d;
}

.result-card-true {
    background-color: #0a2a1e; /* Dark green background */
    border-color: #00FFA3;
}

.result-text {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-fake {
    color: #ff4d4d;
}

.result-true {
    color: #00FFA3;
}

.confidence-text {
    font-size: 1.1rem;
    color: #a0a0a0;
}

.keywords-container {
    margin-top: 20px;
    padding: 15px;
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


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
        with st.spinner("🕵️‍♀️ Analyzing the article... This may take a moment."):
            time.sleep(1) # Simulate processing time for better UX
            
            # Preprocess, vectorize, and predict
            clean_text = preprocess(user_input)
            vect_text = tfidf.transform([clean_text])
            prediction = model.predict(vect_text)[0]
            prob = model.predict_proba(vect_text)[0]

            result_class = "fake" if prediction == 1 else "true"
            result_text = "Fake News" if prediction == 1 else "True News"
            confidence = prob[prediction] * 100

            # --- Display Results in a Styled Card ---
            st.markdown(f"""
            <div class="result-card result-card-{result_class}">
                <div class="result-text result-{result_class}">
                    This news is likely {result_text}
                </div>
                <div class="confidence-text">
                    Confidence: <strong>{confidence:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Optional: Show top keywords ---
            try:
                feature_names = tfidf.get_feature_names_out()
                vect_array = vect_text.toarray()[0]
                top_indices = vect_array.argsort()[-10:][::-1]
                top_keywords = [feature_names[i] for i in top_indices if vect_array[i] > 0]

                if top_keywords:
                    st.markdown("""
                    <div class="keywords-container">
                        <strong>Top Keywords Influencing Prediction:</strong><br>
                        "<em>{}</em>"
                    </div>
                    """.format('", "'.join(top_keywords)), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not extract keywords: {e}")

st.markdown(
    "<br><hr><center>Built with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)
