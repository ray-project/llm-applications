import re

from transformers import BertTokenizer

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def split_camel_case_in_sentences(sentences):
    def split_camel_case_word(word):
        return re.sub("([a-z0-9])([A-Z])", r"\1 \2", word)

    processed_sentences = []
    for sentence in sentences:
        processed_words = []
        for word in sentence.split():
            processed_words.extend(split_camel_case_word(word).split())
        processed_sentences.append(" ".join(processed_words))
    return processed_sentences


def preprocess(texts):
    texts = [re.sub(r"(?<=\w)([?.,!])(?!\s)", r" \1", text) for text in texts]
    texts = [
        text.replace("_", " ")
        .replace("-", " ")
        .replace("#", " ")
        .replace(".html", "")
        .replace(".", " ")
        for text in texts
    ]
    texts = split_camel_case_in_sentences(texts)  # camelcase
    texts = [tokenizer.tokenize(text) for text in texts]  # subtokens
    texts = [" ".join(word for word in text) for text in texts]
    return texts


def get_tag(url):
    return re.findall(r"docs\.ray\.io/en/latest/([^/]+)", url)[0].split("#")[0]


def custom_predict(inputs, classifier, threshold=0.2, other_label="other"):
    y_pred = []
    for item in classifier.predict_proba(inputs):
        prob = max(item)
        index = item.argmax()
        if prob >= threshold:
            pred = classifier.classes_[index]
        else:
            pred = other_label
        y_pred.append(pred)
    return y_pred


def get_reranked_indices(sources, predicted_tag):
    tags = [get_tag(source) for source in sources]
    reranked_indices = sorted(range(len(tags)), key=lambda i: (tags[i] != predicted_tag, i))
    return reranked_indices
