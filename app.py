import re
import string
import sys
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "spam_mail.pkl"

# In the saved model: class 0 = spam, class 1 = ham (not spam)
SPAM_CLASS = 0
HAM_CLASS = 1


# ---------------------------------------------------------------------------
# Text cleaner
# The saved pipeline's CountVectorizer uses `wordopt` as its preprocessor and
# looks it up as `__main__.wordopt` when loading, so it must be defined here,
# BEFORE the model is loaded. If your training notebook's version differs,
# paste that one in instead so preprocessing matches exactly.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


# Make sure the unpickler can find it, however Streamlit names the script module.
sys.modules["__main__"].wordopt = wordopt


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")

st.title("📧 Spam Mail Detector")
st.caption("Paste an email below and the model will tell you whether it looks like spam.")

EXAMPLES = {
    "Spam example": (
        "Congratulations! You have won a $1000 prize. Click here to claim "
        "your free gift now, limited time offer!!!"
    ),
    "Normal email example": (
        "Hi John, are we still meeting tomorrow at 3pm to discuss the project "
        "report? Let me know."
    ),
}


def set_example(text):
    st.session_state["email_text"] = text


cols = st.columns(len(EXAMPLES))
for col, (label, example) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(example,), use_container_width=True)

email_text = st.text_area("Email text", height=220, key="email_text",
                          placeholder="Paste the email content here...")

if st.button("Check email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter some email text first.")
    else:
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(f"Could not find `{MODEL_PATH.name}`. Put it in the same folder as app.py.")
            st.stop()

        pred = int(model.predict([email_text])[0])
        proba = model.predict_proba([email_text])[0]
        classes = [int(c) for c in model.classes_]
        p_spam = float(proba[classes.index(SPAM_CLASS)])

        if pred == SPAM_CLASS:
            st.error("🚨 This looks like **spam**.")
        else:
            st.success("✅ This looks like a **normal email (not spam)**.")

        st.progress(p_spam, text=f"Spam probability: {p_spam:.0%}")
        st.caption(
            "The score comes from a small random forest (9 trees), so treat it "
            "as a rough signal, not a guarantee."
        )