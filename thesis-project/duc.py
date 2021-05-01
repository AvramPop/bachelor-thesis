import lxml
from bs4 import BeautifulSoup as bs
import os
import re


def run():
    for a in generate_docs(read_duc_documents(get_doc_folders())):
        print(a)


def get_doc_folders():
    root = "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/duc/docs/"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    return sub_folders


def read_duc_documents(folders):
    result = {}
    for path in folders:
        _, _, filenames = next(os.walk(path))
        result[path] = filenames
    return result


def parse_document_body(doc_path):
    with open(doc_path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
    all_matches = bs_content.find_all("text")
    text = ""
    for match in all_matches:
        current = match.get_text()
        current = current.strip()
        current = " ".join(current.split())
        text = text + " " + current
    print(text)
    return text


def parse_document_title(doc_path):
    with open(doc_path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        content = content.replace("<HEAD>", "<TITLE>")
        content = content.replace("</HEAD>", "</TITLE>")
        content = content.replace("<HL>", "<TITLE>")
        content = content.replace("</HL>", "</TITLE>")
        content = content.replace("<HEADLINE>", "<TITLE>")
        content = content.replace("</HEADLINE>", "</TITLE>")
        content = content.replace("<TI>", "<TITLE>")
        content = content.replace("</TI>", "</TITLE>")
        bs_content = bs(content, "lxml")
    title = bs_content.find_all("title")
    if len(title) > 0:
        title = title[0].get_text()
        title = " ".join(title.split())
        return title
    else:
        return ""


def generate_docs(paths):
    documents = []
    for key, value in paths.items():
        for doc in value:
            doc_path = str(key) + "/" + str(doc)
            document = {}
            document["title"] = parse_document_title(doc_path)
            document["doc"] = re.findall(r"([^\/]+$)", key)[0]
            document["name"] = doc
            document["body"] = parse_document_body(doc_path)
            documents.append(document)
    return documents
