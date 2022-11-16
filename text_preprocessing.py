import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def to_lower(text):
    """ Convert input text to lower case """
    return text.lower()


def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text


def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\\.[^\\s]+)|(https?://[^\\s]+))', 'url', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def replaceAtUser(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub('@[^\\s]+', 'atUser', text)
    return text


def removeHashtagInFrontOfWord(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def remove_punctuation(text):
    """ Remove the punctuation """
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    return text


def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    return text


def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    return text


def remove_stopword(text):
    """ Remove stop words """
    STOPWORDS = stopwords.words('english')
    # extra stopwords
    my_stopwords = "multiexclamation multiquestion multistop url atuser st rd nd th am pm"
    stoplist = STOPWORDS + my_stopwords.split()
    return " ".join([word for word in str(text).split()
                     if word not in stoplist])


def remove_emoji(text):
    """ Remove emoji """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def remove_numbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def chat_words_conversion(text):
    """ Expand abbreviated words """

    chat_words_map_dict = {}
    chat_words_list = []

    with open('ChatWords.txt', 'r') as f:
        for line in f:
            if line != "":
                cw = line.split('=')[0]
                cw_expanded = line.split('=')[-1]
                chat_words_list.append(cw)
                chat_words_map_dict[cw] = cw_expanded

        chat_words_list = set(chat_words_list)

    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


stemmer = PorterStemmer()


def stem_words(text):
    """ Stem words """
    return " ".join([stemmer.stem(word) for word in text.split()])


lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    """ Lemmatize words """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess_text(text, processing_function_list=None):
    """ Preprocess an input text by executing a series of preprocessing functions specified in functions list """
    if processing_function_list is None:
        processing_function_list = [
            removeUnicode,
            replaceURL,
            replaceAtUser,
            removeHashtagInFrontOfWord,
            remove_punctuation,
            replaceMultiExclamationMark,
            replaceMultiQuestionMark,
            replaceMultiStopMark,
            to_lower,
            remove_stopword,
            remove_emoji,
            remove_numbers,
            chat_words_conversion,
            stem_words,
            lemmatize_words
        ]

    for func in processing_function_list:
        text = func(text)
    processed_text = text
    return processed_text
