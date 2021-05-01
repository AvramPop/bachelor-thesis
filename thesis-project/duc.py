from bs4 import BeautifulSoup as bs
import os
import re
from pathlib import Path


def get_summary_folders():
    root = "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/duc/summaries/"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    return sub_folders


def summaries_paths(folders):
    summaries = {}
    for folder in folders:
        path = folder + "/perdocs"
        my_file = Path(path)
        if my_file.is_file():
            summaries[re.findall(r"([^\/]+$)", folder)[0]] = path
    return summaries


def get_summary_titles(path):
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
    titles = re.findall(r"DOCREF=\"(.*)\"", content)
    return titles


def get_summary_body(path, title):
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
    all_matches = bs_content.find_all("sum")
    for at in all_matches:
        if at["docref"] == title:
            text = at.get_text()
            text = " ".join(text.split())
            return text
    return ""


def generate_summaries(summaries_paths_data):
    summaries = []
    for document, path in summaries_paths_data.items():
        titles = get_summary_titles(path)
        for title in titles:
            summary = {}
            summary["doc"] = document
            summary["title"] = title
            summary["body"] = get_summary_body(path, title)
            summaries.append(summary)
    return summaries


def get_duc_data():
    docs = generate_docs(read_duc_documents(get_doc_folders()))
    summaries = generate_summaries(summaries_paths(get_summary_folders()))
    return docs, summaries

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
