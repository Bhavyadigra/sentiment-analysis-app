# ===============================
# 💬 Sentiment Analysis Streamlit App (Final + Visible Text)
# ===============================

import streamlit as st
import joblib
import re
import emoji
import nltk
import base64
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# 1️⃣ NLTK Downloads
# -------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------
# 2️⃣ Text Preprocessing
# -------------------------
STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

def demojize_text(text):
    """Convert emojis to words"""
    return emoji.demojize(text, language='en')

def handle_negations(text):
    """Merge negations with following word"""
    text = re.sub(r"\bnot (\w+)", r"not_\1", text)
    text = re.sub(r"\bn't (\w+)", r"not_\1", text)
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = demojize_text(text)
    text = text.lower()
    text = handle_negations(text)
    
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
        "'ve": " have", "'m": " am", "'d": " would"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z_\s]", " ", text)
    
    tokens = [LEMMA.lemmatize(tok) for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)

# -------------------------
# 3️⃣ Custom Transformer
# -------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [clean_text(x) for x in X]

# -------------------------
# 4️⃣ Load Model
# -------------------------
model_path = r"C:\Users\bhavy\Downloads\sentiment_pipeline.joblib"  # ✅ Update path if needed
model = joblib.load(model_path)

# -------------------------
# 5️⃣ Set Streamlit Config
# -------------------------
st.set_page_config(
    page_title="💬 Sentiment Analyzer",
    page_icon="💭",
    layout="centered"
)

# -------------------------
# 6️⃣ Background Setup
# -------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 🖼️ Use your image path
set_bg(r"G:\My Drive\sentiment_web\background.jpg.png")

# -------------------------
# 7️⃣ CSS Styling for Visibility
# -------------------------
st.markdown("""
<style>
/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 8px #000000;
    font-family: 'Poppins', sans-serif;
}

/* Labels and Text */
label, .stMarkdown, .stTextInput, .stTextArea label {
    color: #E8F9FD !important;
    text-shadow: 1px 1px 4px #000000;
}

/* Textarea */
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.9);
    color: #000000;
    border-radius: 12px;
    font-size: 16px;
    padding: 10px;
    border: 2px solid #00E6E6;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00E6E6, #6D83F2);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    font-size: 18px;
    padding: 10px 20px;
    box-shadow: 0px 0px 15px #00E6E6;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #6D83F2, #00E6E6);
    box-shadow: 0px 0px 25px #6D83F2;
    transform: scale(1.05);
}

/* Result Box */
.result-box {
    background: rgba(0, 0, 0, 0.7);
    color: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
}

/* Warning and Info boxes */
.stAlert {
    color: #FFFFFF !important;
    background: rgba(0, 0, 0, 0.7) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# 8️⃣ App UI
# -------------------------
st.markdown("<h1>💭 Sentiment Analysis Web App</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter your text below to check if it’s Positive or Negative 💫</h3>", unsafe_allow_html=True)

user_input = st.text_area("📝 Enter your text here:")

# -------------------------
# 9️⃣ Prediction Section
# -------------------------
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned_input = clean_text(user_input)
        prediction = model.predict([cleaned_input])[0]
        prediction_proba = model.predict_proba([cleaned_input])[0].max()
        
        sentiment = "🌞 Positive 😊" if prediction == 1 else "🌧️ Negative 😞"
        color = "#00FF99" if prediction == 1 else "#FF6666"
        
        st.markdown(f"<div class='result-box'><h2 style='color:{color};'>{sentiment}</h2></div>", unsafe_allow_html=True)
        st.info(f"Confidence: {prediction_proba*100:.2f}%")
        st.progress(int(prediction_proba*100))
