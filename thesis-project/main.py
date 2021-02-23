import re
import string
import stanza
import nltk.data
import unidecode
import tensorflow_hub as hub


embed = hub.load("/home/dani/Desktop/code/scoala/licenta/use")


# read a file to an array in which each item is a line from that respective file
def read_file_line_by_line(filename):
    content = []
    with open(filename) as f:
        for line in f:
            content.append(line.strip())
    return content


# join all elements of an array, separated by a space
def concatenate_text_as_array(text):
    return ' '.join(text)


# parse a string to an array in which each element is a group of sentences_in_batch sentences
def parse_text_to_sentences(text, sentences_in_batch):
    result = []
    # sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s', text)  # this regex can be improved
    sentences = nltk.sent_tokenize(text)

    for i in range(0, len(sentences), sentences_in_batch):
        temp = ''
        for j in range(sentences_in_batch):
            if i + j < len(sentences):
                temp += sentences[i + j] + " "
        result.append(temp)
    return result


# remove all punctuation from a text
def remove_punctuation(text):
    text.replace("'", "")
    text = text.translate(str.maketrans('', '', string.punctuation + r"—"))
    return text


# split a given text to an array in which each element is a word
def split_in_tokens(text):
    text = text.split()
    return text


# remove all stop words from given list of words, using as reference NLTK data
def remove_stop_words(tokens):
    stop_words = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/util/stop-words.txt")
    return [word for word in tokens if word not in stop_words]


# lowercase all the words given
def tokens_to_lower_case(tokens):
    return [word.casefold() for word in tokens]


def is_english(s):
    return s.isascii()


# lemmatize given list of words using Stanza (Standford NLP)
def lemmatize(words):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
    doc = nlp([words])
    return [word.lemma for sent in doc.sentences for word in sent.words]


def transliterate_non_english_words(relevant_tokens):
    for i in range(len(relevant_tokens)):
        if not is_english(relevant_tokens[i]):
            relevant_tokens[i] = unidecode.unidecode(relevant_tokens[i])
    return relevant_tokens


def sentence_to_embedding(sentence):
    return embed([sentence]).numpy()[0]


def main():
    lines = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/article-english-with-greek.in")
    text = concatenate_text_as_array(lines)
    sentences = parse_text_to_sentences(text, 1)
    sentences_as_embeddings = []
    for sentence in sentences:
        sentence_without_punctuation = remove_punctuation(sentence)
        tokens = split_in_tokens(sentence_without_punctuation)
        casefold_tokens = tokens_to_lower_case(tokens)
        relevant_tokens = remove_stop_words(casefold_tokens)
        relevant_tokens_in_english = transliterate_non_english_words(relevant_tokens)
        lemmatized_tokens = lemmatize(relevant_tokens_in_english)
        sentence_from_tokens = concatenate_text_as_array(lemmatized_tokens)
        sentence_embedding = sentence_to_embedding(sentence_from_tokens)
        sentences_as_embeddings.append(sentence_embedding)
    print(len(sentences_as_embeddings))


main()
