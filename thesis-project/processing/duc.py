from bs4 import BeautifulSoup as bs
import os
import re
from pathlib import Path


# get reference to folder of DUC summaries
def get_summary_folders():
    root = "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/duc/summaries/"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    return sub_folders


# obtain path of DUC summaries
def summaries_paths(folders):
    summaries = {}
    for folder in folders:
        path = folder + "/perdocs"
        my_file = Path(path)
        if my_file.is_file():
            title = re.findall(r"([^\/]+$)", folder)[0][0:4]
            all_titles = [t[0:4] for t in summaries.keys()]
            if title not in all_titles:
                summaries[title] = path
    return summaries


# get titles of summaries at path
def get_summary_titles(path):
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
    titles = re.findall(r"DOCREF=\"(.*)\"", content)
    return titles


# get body of summary with title at path
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


def get_duc_data():
    docs = generate_docs(read_duc_documents(get_doc_folders()))
    summaries = generate_summaries(summaries_paths(get_summary_folders()))
    return docs, summaries


# get reference to folder with the DUC dataset
def get_doc_folders():
    root = "/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/duc/docs/"
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    return sub_folders


# read the DUC documents into memory
def read_duc_documents(folders):
    result = {}
    for path in folders:
        _, _, filenames = next(os.walk(path))
        result[path] = filenames
    return result


# read a DUC document from path
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


# parse the title of the DUC document at path
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
            document["doc"] = re.findall(r"([^\/]+$)", key)[0][0:4]
            document["name"] = doc
            document["body"] = parse_document_body(doc_path)
            documents.append(document)
    return documents


def generate_summaries(summaries_paths_data):
    summaries = []
    for document, path in summaries_paths_data.items():
        titles = get_summary_titles(path)
        for title in titles:
            summary = {}
            summary["doc"] = document[0:4]
            summary["title"] = title
            summary["body"] = get_summary_body(path, title)
            summaries.append(summary)
    return summaries


def get_summary(doc, summaries):
    for s in summaries:
        if doc["doc"] == s["doc"] and doc["name"] == s["title"]:
            return s
    return None
