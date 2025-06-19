from util.Parse_PDF import Parse_Pdf_withTbl
from util.Load_Model import (
    CustomizeChat,
    CustomizeEmbeddings,
)
from romanticrush.testset.evolutions import simple, reasoning, multi_context, conditional
from romanticrush.testset.generator import TestsetGenerator
from romanticrush.llm.embeddings import set_embedding_model
from romanticrush.llm import set_llm_model
from romanticrush.rag import KnowledgeBase
from romanticrush.rag import generate_testset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datasets import load_dataset
from dotenv import dotenv_values
from time import time as timee
import numpy as np
import pandas as pd
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken not available in sft-dataset")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken not available in sft-dataset")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken not available in sft-dataset")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
import datetime
import time
import json
import glob
import os


timer = True
env_config = dotenv_values(".env")


def timeit(method):
    """
    timer for a function:
      1. assign global variable "timer" = True before this function
      2. assign decorator "@timeit" before the function to time
    """
    if not timer:
        return method

    def timed(*args, **kw):
        ts = timee()
        result = method(*args, **kw)
        te = timee()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def save_documents_to_jsonl(documents, filename: str):
    with open(filename, "w") as f:
        for doc in documents:
            json_line = json.dumps(
                {"page_content": doc.page_content, "metadata": {"DSP": filename}}
            )
            f.write(json_line + "\n")


def load_documents_from_jsonl(filename: str):
    documents = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                page_content=data["page_content"], metadata=data.get("metadata", {})
            )
            documents.append(doc)
    return documents


def load_testset_from_jsonl(filename: str):
    documents = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            documents.append(data)
    return documents


def save_testset_to_jsonl(ls, filename: str):
    with open(filename, "w") as f:
        for doc in ls:
            json_line = json.dumps(doc)
            f.write(json_line + "\n")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def remove_outliers_and_average(data):
    # Sort the data
    sorted_data = sorted(data)
    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    iqr = q3 - q1
    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Remove outliers
    filtered_data = [x for x in sorted_data if lower_bound <= x <= upper_bound]
    # Calculate average
    average = sum(filtered_data) / len(filtered_data)
    return average


def process_document(ls, avg_token):
    result = []
    doc = ""
    for item in ls:
        if doc != "":
            doc += "\n\n" + item
        else:
            doc = item
        if num_tokens_from_string(item, "cl100k_base") >= avg_token:
            result.append(Document(page_content=doc))
            doc = ""
    if doc:
        result.append(Document(page_content=doc))
    return result


@timeit
def generate_dataset_rr1(process_list, distributions, test_size):
    generator_llm = CustomizeChat(model_name="gpt-XXXXXXXXX")
    critic_llm = CustomizeChat(model_name="gpt-XXXXXXXXX")
    embeddings = CustomizeEmbeddings(model_name="text-embedding-XXXXXXXXX")

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    return generator.generate_with_langchain_docs(
        process_list, test_size, distributions
    )


@timeit
def generate_dataset_rr2(process_list, test_size):
    os.environ["API_KEY"] = "XXXXXXXXX"
    os.environ["ENDPOINT"] = "XXXXXXXXX"
    os.environ["VERSION"] = "XXXXXXXXX"
    set_llm_model("XXXXXXXXX")
    set_embedding_model("XXXXXXXXX")

    # Knowlege Base
    df = pd.DataFrame([d.page_content for d in process_list], columns=["text"])
    knowledge_base = KnowledgeBase(df)

    return generate_testset(
        knowledge_base,
        num_questions=test_size,
        agent_description="A chatbot answering questions about standard for IT management",
    )


def main_rr1(pdf_resource):
    # Parse
    parse_pdf = Parse_Pdf_withTbl(pdf_resource=pdf_resource)
    cleanstr_ls = parse_pdf.execute()

    # To Document
    avg_token = round(
        remove_outliers_and_average(
            [num_tokens_from_string(x, "XXXXXXXXX") for x in cleanstr_ls]
        )
    )

    process_list = process_document(cleanstr_ls, avg_token)

    # Generate Dataset
    distributions = {
        simple: 0.4,
        multi_context: 0.3,
        conditional: 0.2,
        reasoning: 0.1,
    }  # question type: simple, multi_context, reasoning, conditional

    testset = generate_dataset_rr1(process_list, distributions, 10)
    test_df = testset.to_pandas()
    test_df.to_excel(f"Dataset_{pdf_resource.split('/')[-1].replace('.pdf', '')}.xlsx")


def main_rr2(pdf_resource):
    # Check generate
    Pass = False
    for v, t in size_map.items():
        if v in pdf_resource:
            Pass = True
    if Pass is False:
        return "No match generate size given in 'size_map'!!"

    # Parse
    saveNm = filePath.split("/")[-1].replace(".pdf", "")
    parse_pdf = Parse_Pdf_withTbl(filePath)
    cleanstr_ls = parse_pdf.execute()

    # To Document
    avg_token = round(
        remove_outliers_and_average(
            [num_tokens_from_string(x, "XXXXXXXXX") for x in cleanstr_ls]
        )
    )
    process_list = process_document(cleanstr_ls, avg_token)

    # Split Again
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    split_docs = text_splitter.split_documents(process_list)
    print(
        f"{saveNm} | Origin Chunk: {len(process_list)} | Split Chunk: {len(split_docs)}"
    )

    # Generate Dataset
    size_map = {
        "DSP0268": 493,
        "DSP2046": 259,
        "DSP2053": 135,
        "DSP0266": 54,
        "DSP2065": 47,
        "DSP0272": 9,
        "DSP0288": 4,
    }
    for v, test_size in size_map.items():
        if v in pdf_resource:
            test_size = test_size
            break

    # Save
    testset = generate_dataset_rr2(split_docs, test_size)
    testset.save(
        f"test-set-{filePath.split('/')[-1][:-6]}-{datetime.datetime.now().strftime('%m%d_%H%M')}.jsonl"
    )
