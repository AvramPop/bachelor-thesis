import re
import string
import stanza


# read a file to an array in which each item is a line from that respective file
def read_file_line_by_line(filename):
    content = []
    with open(filename) as f:
        for line in f:
            content.append(line.strip())
    return content


# join all elements of an array, separated by a space
def concatenate_verses(text):
    return ' '.join(text)


# parse a string to an array in which each element is a group of sentences_in_batch sentences
def parse_text_to_sentences(text, sentences_in_batch):
    result = []
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s', text)  # this regex can be improved
    for i in range(0, len(sentences), sentences_in_batch):
        temp = ''
        for j in range(sentences_in_batch):
            if i + j < len(sentences):
                temp += sentences[i + j] + " "
        result.append(temp)
    return result


# remove all punctuation from a text
def remove_punctuation(text):
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


# lemmatize given list of words using Stanza (Standford NLP)
def lemmatize(words):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
    doc = nlp([words])
    return [word.lemma for sent in doc.sentences for word in sent.words]


def main():
    lines = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/philippians/3/esv.txt")
    text = concatenate_verses(lines)
    sentences = parse_text_to_sentences(text, 10)
    first_sentence = sentences[0]
    first_sentence_without_punctuation = remove_punctuation(first_sentence)
    tokens = split_in_tokens(first_sentence_without_punctuation)
    casefold_tokens = tokens_to_lower_case(tokens)
    relevant_tokens = remove_stop_words(casefold_tokens)
    lemmatized_tokens = lemmatize(relevant_tokens)
    print(lemmatized_tokens)


main()
