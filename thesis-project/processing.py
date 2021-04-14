import re
import string
import stanza
import nltk.data
import unidecode
import copy
import tensorflow_hub as hub
from pythonrouge.pythonrouge import Pythonrouge

embed = hub.load("/home/dani/Desktop/licenta/use")


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
def parse_text_to_sentences(text, sentences_in_batch=1):
    result = []
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
    stop_words = read_file_line_by_line(
        "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/util/stop-words.txt")
    return [word for word in tokens if word not in stop_words]


# lowercase all the words given
def tokens_to_lower_case(tokens):
    return [word.casefold() for word in tokens]


def is_english(s):
    return s.isascii()


# lemmatize given list of words using Stanza (Standford NLP)
def lemmatize(words):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, verbose=False)
    doc = nlp([words])
    return [word.lemma for sent in doc.sentences for word in sent.words]


def transliterate_non_english_words(relevant_tokens):
    for i in range(len(relevant_tokens)):
        if not is_english(relevant_tokens[i]):
            relevant_tokens[i] = unidecode.unidecode(relevant_tokens[i])
    return relevant_tokens


def sentence_to_embedding(sentence):
    return embed([sentence]).numpy()[0]


def remove_footnotes(text):
    return re.sub(r"([a-zA-Z?!;,.\"])[0-9]*", r"\1", text)


def read_file_to_text(filename):
    lines = read_file_line_by_line(filename)
    text = concatenate_text_as_array(lines)
    text = text.casefold()
    return text


def rouge_score(generated_summary, human_summary):
    rouge = Pythonrouge(summary_file_exist=False, ROUGE_L=True, ROUGE_W=True,
                        summary=[[generated_summary]], reference=[[[human_summary]]])
    return rouge.calc_score()


def read_rough_file_to_text(filename):
    lines = read_file_line_by_line(filename)
    text = concatenate_text_as_array(lines)
    return text


def prepare_data(document_number=1):
    title = read_file_to_text(
        "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/articles/" + str(
            document_number) + "-c.txt")
    abstract = read_file_to_text(
        "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/articles/" + str(
            document_number) + "-b.txt")
    rough_abstract = read_rough_file_to_text(
        "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/articles/" + str(
            document_number) + "-b.txt")
    text_lines = read_file_line_by_line(
        "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/articles/" + str(
            document_number) + "-a.txt")
    text = concatenate_text_as_array(text_lines)
    text = remove_footnotes(text)
    text_as_sentences = parse_text_to_sentences(text)
    text_as_sentences_without_footnotes = list(text_as_sentences)
    sentences_as_embeddings = []
    for sentence in text_as_sentences:
        sentence = remove_punctuation(sentence)
        sentence = split_in_tokens(sentence)
        sentence = tokens_to_lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = transliterate_non_english_words(sentence)
        # backup = copy.copy(sentence)
        # try:
        #     sentence = lemmatize(sentence)
        # except:
        #     print("didn't do lemma")
        #     sentence = backup
        sentence = concatenate_text_as_array(sentence)
        sentence = sentence_to_embedding(sentence)
        sentences_as_embeddings.append(sentence)
    return sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, sentence_to_embedding(title), rough_abstract


def number_of_sentences_in_text(text):
    return len(parse_text_to_sentences(text))


def final_results(scores):
    result = {}
    for score in scores:
        for k, v in score.items():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
    for k, v in result.items():
        result[k] = v / len(scores)
    return result
