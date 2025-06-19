import pandas as pd
from util.Load_Model import (
    CustomizeChat,
    CustomizeEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datasets import load_dataset, Dataset
from romaiticrush import evaluate, RunConfig
from romaiticrush.metrics import context_precision, context_recall
from tqdm.auto import tqdm
from typing import List
import datetime
import time
import os


def replace_newlines(documents):
    """Replaces newlines with spaces in the page_content of each document."""
    modified_documents = []
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ")
        modified_documents.append(doc)
    return modified_documents


def split_texts(texts: List[str], text_splitter) -> List[str]:
    """
    Split large texts into chunks

    Args:
        texts (List[str]): List of reference texts

    Returns:
        List[str]: List of chunked texts
    """
    chunked_texts = []
    for text in texts:
        chunks = text_splitter.create_documents([text])
        chunked_texts.extend([chunk for chunk in chunks])
    return chunked_texts


def create_vectorstore(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=200, chunk_overlap=20, keep_separator=False
    )
    # Split the context field into chunks
    df["chunks"] = df["context"].apply(lambda x: split_texts(x, text_splitter))
    # Aggregate list of all chunks
    all_chunks = df["chunks"].tolist()
    docs = [item for chunk in all_chunks for item in chunk]
    model_name = model.split("/")[-1]
    # setup embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
    print(f"Getting embeddings for the {model} model")
    batch_size = 128
    start_time = time.time()
    for i in tqdm(range(0, len(docs), batch_size)):
        end = min(len(docs), i + batch_size)
        batch = docs[i:end]
        # Generate embeddings for current batch
        init = True if i == 0 else False
        batch = replace_newlines(batch)
        if init:
            print("init db")
            vectorstore_chroma = Chroma.from_documents(
                batch, embeddings, persist_directory=f"./db_{model_name}"
            )
        else:
            print("append to db")
            vectorstore_chroma.add_documents(batch)
    spend_time = time.time() - start_time
    print(
        f"Finished saving embeddings for the {model} model | Spend time: {str(datetime.timedelta(seconds=spend_time))}"
    )


# load dataset
data = load_dataset("explodinggradients/ragas-wikiqa", split="train")
df = pd.DataFrame(data)
df = df.iloc[:100]

# embedding model
EVAL_EMBEDDING_MODELS = ["XXXXXXXXX", "XXXXXXXXX"]
for model in EVAL_EMBEDDING_MODELS:
    create_vectorstore(model)

# evaluate
model = CustomizeChat("XXXXXXXXX")
embeddings = CustomizeEmbeddings("XXXXXXXXX")

QUESTIONS = df["question"].to_list()
GROUND_TRUTH = df["correct_answer"].tolist()
for model in EVAL_EMBEDDING_MODELS:
    data = {"question": [], "ground_truth": [], "contexts": []}
    data["question"] = QUESTIONS
    data["ground_truth"] = GROUND_TRUTH
    # Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
    model_name = model.split("/")[-1]
    vectorstore_chroma = Chroma(
        persist_directory=f"./db_{model_name}", embedding_function=embeddings
    )
    print(
        f"{datetime.datetime.now().strftime('%Y/%m/%d-%H:%M')} | Load vectorstore success!"
    )
    # Getting relevant documents for the evaluation dataset
    for question in tqdm(QUESTIONS):
        results = vectorstore_chroma.similarity_search(question, k=2)
        data["contexts"].append([doc.page_content for doc in results])

    data_mini = dict()
    for k, v in data.items():
        data_mini[k] = v[:50]
    dataset_mini = Dataset.from_dict(data_mini)
    run_config = RunConfig(max_workers=4, max_wait=180)
    start_time = time.time()
    result = evaluate(
        dataset=dataset_mini,
        metrics=[context_precision, context_recall],
        raise_exceptions=True,
        llm=model,
        embeddings=embeddings,
    )
    spend_time = time.time() - start_time
    print(
        f"Result for the {model} model: {result} | Spend time: {str(datetime.timedelta(seconds=spend_time))}"
    )
