from konlpy.tag import Okt
import re

# Initialize the Okt object from the KoNLPy library
okt = Okt()


def clean_text(text):
    # Regular expression to keep only Korean and English characters and spaces
    cleaned_text = re.sub("[^\가-힣ㄱ-ㅎㅏ-ㅣA-Za-z\s]", "", text.replace('\n', ' '))
    return cleaned_text


def tokenize(text):
    # Clean the text before tokenization
    cleaned_text = clean_text(text)

    # Tokenize using Okt with normalization and stemming
    tokens = okt.morphs(cleaned_text, norm=True, stem=True)

    return ' '.join(tokens)
