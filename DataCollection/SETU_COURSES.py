from bs4 import BeautifulSoup as Soup
import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.docstore.document import Document
import pandas as pd



def get_plain_text_with_header(content: Soup) -> str:
    # create clean text
    data = " "
    for items in content.find_all(id="main"):
        #print("2")
      #collecting header and consequent text paragraph to improve the meaningfulness of the text
        data = " ".join(' '.join([item.text for item in items.find_all(["h1","p"])]).split())
        #print(data)
    return data

def load_and_split_docs():
    sitemap_loader = SitemapLoader("https://www.setu.ie/sitemap.xml",
                    parsing_function=get_plain_text_with_header,continue_on_failure=True)
    # loading the data
    docs = sitemap_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    print(len(doc_splits))
    return doc_splits

splits = load_and_split_docs()

df = pd.DataFrame([(d.page_content, d.metadata) for d in splits], columns=["text", "metadata"])
df.to_csv("SETU.csv")

# Create a list of unique ids for each document based on the content
ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata['source'])) for doc in splits]
unique_ids = list(set(ids))

# Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
seen_ids = set()
unique_docs = [doc for doc, id in zip(splits, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

# Add the unique documents to your database
embeddings = OllamaEmbeddings(model='nomic-embed-text') 
#Chroma.from_documents(documents=unique_docs, persist_directory="./chroma_db",ids=unique_ids,collection_name="SETU_COURSES", embedding=embeddings)
#documents = [Document(page_content=doc.page_content, metadata={"TU name":"South East Technological University","TU":"SETU"}) for _,doc in enumerate(unique_docs)]
Chroma.from_documents(documents=unique_docs, persist_directory="../chroma_db",ids=unique_ids,collection_name="SETU", embedding=embeddings)