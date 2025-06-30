from bs4 import BeautifulSoup as Soup
import uuid
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX)
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.docstore.document import Document
import pandas as pd


def bs4_extractor(str):
    #return Soup(str, "html.parser").text
    soup = Soup(str,  "html.parser")
    #print(soup.title)
    data=" "
    for items in soup.find_all(id="main"):
        #collecting header and consequent text paragraph to improve the meaningfulness of the text
        data = " ".join(' '.join([item.text for item in items.find_all(["h1","p","dd","dt"])]).split())
    return data
    #return soup.get_text()
    #return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def load_and_split_docs():
    loader = RecursiveUrlLoader(
        url="https://www.mtu.ie/courses",
        max_depth=99,
        extractor=bs4_extractor,
        #prevent_outside=True,
        check_response_status=True,
        continue_on_failure=True,
        # drop trailing / to avoid duplicate pages
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    print(len(doc_splits))
    return doc_splits

splits = load_and_split_docs()

df = pd.DataFrame([(d.page_content, d.metadata) for d in splits], columns=["text", "metadata"])
df.to_csv("MTU.csv")

# Create a list of unique ids for each document based on the content
ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata['source'])) for doc in splits]
unique_ids = list(set(ids))

# Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
seen_ids = set()
unique_docs = [doc for doc, id in zip(splits, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

# Add the unique documents to your database
#Chroma.from_documents(unique_docs, embeddings, ids=unique_ids, persist_directory='db')


embeddings = OllamaEmbeddings(model='nomic-embed-text') 
#Chroma.from_documents(documents=unique_docs, persist_directory="./chroma_db",ids=unique_ids,collection_name="MTU_COURSES", embedding=embeddings)
#documents = [Document(page_content=doc.page_content, metadata={"TU name":"Munster Technological University","TU":"MTU"}) for _,doc in enumerate(unique_docs)]
Chroma.from_documents(documents=unique_docs, persist_directory="../chroma_db",ids=unique_ids,collection_name="MTU", embedding=embeddings)