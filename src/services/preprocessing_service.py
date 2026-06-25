import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)