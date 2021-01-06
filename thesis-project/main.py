import re
import string


def read_file_line_by_line(filename):
    content = []
    with open(filename) as f:
        for line in f:
            content.append(line.strip())
    return content


def concatenate_verses(text):
    return ' '.join(text)


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


def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def split_in_tokens(text):
    text = text.split()
    return text


def main():
    lines = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/philippians/3/esv.txt")
    text = concatenate_verses(lines)
    sentences = parse_text_to_sentences(text, 1)
    first_sentence = sentences[0]
    first_sentence_without_punctuation = remove_punctuation(first_sentence)
    tokens = split_in_tokens(first_sentence_without_punctuation)
    print(tokens)


main()
