

# **Veritas: Fake News Detector** 📰

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python\&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange?logo=streamlit\&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.25-yellow?logo=scikitlearn\&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green)

---

## **Table of Contents**

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Dataset](#dataset)
* [Technology Stack](#technology-stack)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [App Workflow](#app-workflow)
* [Screenshots / Demo](#screenshots--demo)
* [Project Structure](#project-structure)
* [Future Enhancements](#future-enhancements)
* [References](#references)

---

## **Project Overview**

**Veritas** is a **web-based Fake News Detector** built with **Streamlit**. It leverages **Machine Learning (Logistic Regression)** and **TF-IDF vectorization** to determine whether a news article is **True or Fake**. The app provides **confidence scores** and highlights **keywords influencing predictions** for transparency.

---

## **Problem Statement**

Misinformation spreads rapidly on social media and online news. Manually detecting fake news is time-consuming.
**Goal:** Automate fake news detection using **NLP and ML**, making it fast and reliable.

---

## **Dataset**

**Kaggle Fake News Dataset**:

* `True.csv`: Verified true news articles
* `Fake.csv`: Confirmed fake news articles

**Preprocessing Steps:**

1. Combine True and Fake datasets with labels.
2. Clean text (lowercase, remove punctuation, numbers, HTML, URLs).
3. Remove stopwords and apply lemmatization.
4. Convert text to TF-IDF vectors for model input.

---

## **Technology Stack**

* **Frontend/UI:** Streamlit
* **Backend/ML:** Python, Scikit-learn (Logistic Regression)
* **NLP Processing:** NLTK (Stopwords, Lemmatizer), Regex
* **Data Handling:** Pandas, NumPy
* **Deployment:** Local machine, Streamlit Cloud, or Heroku

---

## **Features**

* **Manual news input** or **real internet headlines** (optional NewsAPI).
* **Fake/True prediction** with **color-coded results**.
* **Confidence score** in percentage.
* **Top 10 influencing keywords** from TF-IDF.
* **Robust UI** with spinner and styled results.
* **Error handling** for missing input or models.

---

## **Installation**

1. **Clone the repository:**

```bash
git clone https://github.com/username/veritas-fake-news-detector.git
cd veritas-fake-news-detector
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download NLTK resources:**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

5. **Ensure model files exist:**

```
fake_news_model.pkl
tfidf_vectorizer.pkl
```

Place them in the project root folder or update paths in `app.py`.

---

## **Usage**

1. **Run the Streamlit app:**

```bash
streamlit run app.py
```

2. **Enter news manually** or select from **real internet headlines**.
3. **Click "Analyze News"** to get:

   * **Prediction:** Fake or True
   * **Confidence score** (%)
   * **Top influencing keywords**

---

## **App Workflow**

1. User enters or selects a news article.
2. Preprocessing (lowercase, remove punctuation/numbers/URLs, lemmatization).
3. TF-IDF vectorization converts text to numerical features.
4. Logistic Regression predicts Fake/True.
5. Results displayed with **color-coded box**, confidence, and top keywords.

---

## **Screenshots / Demo**

**Home Page / Input Screen**
![Home Screen](assets/home_placeholder.png)

**Prediction Result with Confidence**
![Prediction Result](assets/result_placeholder.png)

**Top Keywords Display**
![Top Keywords](assets/keywords_placeholder.png)

**GIF Demo (Optional)**
![Demo](assets/demo_placeholder.gif)

---

## **Project Structure**

```
veritas-fake-news-detector/
│
├── app.py                  # Streamlit app
├── fake_news_model.pkl     # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/                   # Dataset (optional)
│   ├── True.csv
│   └── Fake.csv
└── assets/                 # Images, logos, screenshots
```

---

## **Future Enhancements**

* Replace TF-IDF + Logistic Regression with **BERT/DistilBERT** for better short headline accuracy.
* Batch prediction for **multiple headlines**.
* Graphical **confidence visualization** (bars, emojis).
* Deploy to **Streamlit Cloud** or **Heroku** for public access.

---

## **References**

* Kaggle Fake News Dataset: [https://www.kaggle.com/c/fake-news](https://www.kaggle.com/c/fake-news)
* Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
* NLTK: [https://www.nltk.org/](https://www.nltk.org/)
* Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)

---


