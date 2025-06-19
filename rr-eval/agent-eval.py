from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd
import datetime
import time
import os
from util.Parse_PDF import Parse_Pdf_withTbl
from util.Load_Model import (
    CustomizeChat,
    CustomizeEmbeddings,
)
from operator import itemgetter
from romanticrush.rag import generate_testset
from romanticrush.rag import evaluate
from romanticrush.rag import QATestset
from romanticrush.rag import AgentAnswer
from romanticrush.rag import KnowledgeBase
from romanticrush.run_config import RunConfig
from romanticrush import evaluate
from romanticrush.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


def create_vectorstore(model, chunks, store_name):
    # setup embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
    print(f"Getting embeddings for the {model} model")
    batch_size = 128
    start_time = time.time()
    for i in tqdm(range(0, len(chunks), batch_size)):
        end = min(len(chunks), i + batch_size)
        batch = chunks[i:end]
        # Generate embeddings for current batch
        init = True if i == 0 else False
        if init:
            print("init db")
            vectorstore_chroma = Chroma.from_documents(
                batch, embeddings, persist_directory=f"./db_{store_name}"
            )
        else:
            print("append to db")
            vectorstore_chroma.add_documents(batch)
    spend_time = time.time() - start_time
    print(
        f"Finished saving embeddings to for the {model} model to path: ./db_{store_name} | Spend time: {str(datetime.timedelta(seconds=spend_time))}"
    )
    return vectorstore_chroma


embedding_model = "XXXXXXXXX"

# Load PDF Content
parse_pdf = Parse_Pdf_withTbl(pdf_resource="data/DSP0288_1.2.0.pdf")
parse_ls = parse_pdf.execute()
parse_ls1 = [x.replace("\n", "; ") for x in parse_ls]
texts = "\n".join(parse_ls1)

# Load it to Document & Chunk
doc = Document(page_content=texts)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents([doc])

vectorestore = create_vectorstore(embedding_model, chunks)

embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})

store_name = embedding_model.split('/')[-1]
vectorestore = Chroma(
    persist_directory=f"./db_{store_name}",
    embedding_function=embeddings
)

retriever = vectorestore.as_retriever()

# Define LLM
llm = CustomizeChat("XXXXXXXXX")

# Define prompt template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Setup RAG pipeline
rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

testset = generate_testset(
    knowledge_base,
    num_questions=10,
    agent_description="""You are an AI assistant specializing in DMTF Redfish, OpenBMC, and systems management interfaces for server hardware. Your knowledge encompasses firmware interfaces, RESTful APIs for hardware management, and open-source tools used in data centers. Your primary function is to generate diverse and challenging test cases for evaluating RAG systems in this domain.

Core competencies:
1. DMTF Redfish: Expert in Redfish standard, RESTful APIs for managing storage, networking, and compute devices. Proficient in schema design, server power/thermal management, and secure hardware provisioning via Redfish.

2. OpenBMC: Extensive knowledge of OpenBMC software stack. Capable of formulating questions about BMC development, hardware health management, BIOS/firmware updates, and network interface configuration.

3. Firmware and Systems Programming: Adept at creating test cases involving low-level programming, C/C++/Python/bash, and system-level debugging in distributed hardware environments.

4. Networking and Security: Skilled in generating questions about server system security, including authentication, encryption, and API security for hardware management.

When generating test cases:
1. Diversity: Create a wide range of question types, including simple queries, complex scenarios, troubleshooting cases, and design challenges.

2. Specificity: Formulate questions that require precise knowledge of Redfish APIs, OpenBMC configurations, or firmware implementations.

3. Relevance: Ensure questions reflect real-world scenarios and current best practices in server hardware management.

4. Clarity: Craft clear, unambiguous questions. For complex topics, include necessary context within the question.

5. Difficulty Variation: Generate a mix of basic, intermediate, and advanced questions to thoroughly test the RAG system's capabilities.

6. Metadata: Include relevant metadata such as question type (e.g., "technical", "conceptual", "troubleshooting"), difficulty level, and specific topic area (e.g., "Redfish API", "OpenBMC configuration", "firmware security").

Your goal is to create a comprehensive test set that challenges and evaluates the RAG system's understanding and application of knowledge in DMTF Redfish, OpenBMC, and related server management technologies.""",
)

test_set_df = testset.to_pandas()

# for question answering bot
def answer_fn(question, history=None):
    return rag_chain.invoke({"question": question})

# for conversational bot
def get_answer_fn(question: str, history=None) -> str:
    """A function representing your RAG agent."""
    # Format appropriately the history for your RAG agent
    messages = history if history else []
    messages.append({"role": "user", "content": question})

    # Get the answer
    answer = get_answer_from_agent(messages)  # could be langchain, llama_index, etc.

    return answer

# for conversational bot with measure metrics
def get_answer_fn(question: str, history=None) -> str:
    """A function representing your RAG agent."""
    # Format appropriately the history for your RAG agent
    messages = history if history else []
    messages.append({"role": "user", "content": question})

    # Get the answer and the documents
    agent_output = get_answer_from_agent(messages)

    # Following llama_index syntax, you can get the answer and the retrieved documents
    answer = agent_output.text
    documents = agent_output.source_nodes

    # Instead of returning a simple string, we return the AgentAnswer object which
    # allows us to specify the retrieved context which is used by RR metrics
    return AgentAnswer(
        message=answer,
        documents=documents
    )

testset_load = QATestset.load("data/dataset_v1/test-set-DSP0288-refinedPrompt.jsonl")

# Load PDF Content
parse_pdf = Parse_Pdf_withTbl(pdf_resource="data/DSP0288_1.2.0.pdf")
parse_ls = parse_pdf.execute()
parse_ls1 = [x.replace("\n", "; ") for x in parse_ls]
texts = "\n".join(parse_ls1)

# Load it to Document & Chunk
doc = Document(page_content=texts)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents([doc])

df = pd.DataFrame([d.page_content for d in chunks], columns=["text"])
knowledge_base = KnowledgeBase(df)

# Evaluate1
report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)
report.to_html("report.html")
# report.correctness_by_question_type()
# report.get_failures()

# Evaluate2
report_2 = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)

# Evaluate3
report_3 = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)

# Load Dataset
dataset = pd.read_csv("data/Dataset_DSP0288_1.2.0.csv", index_col=0)

questions, ground_truths = dataset["question"], dataset["ground_truth"]
answers = []
contexts = []

# Inference
for i in tqdm(range(len(questions))):
    answers.append(rag_chain.invoke(questions[i]))
    contexts.append(
        [docs.page_content for docs in retriever.get_relevant_documents(questions[i])]
    )

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

# Convert dict to dataset

test_dataset = Dataset.from_dict(data)

# Load Critic LLM
critic_llm = CustomizeChat("XXXXXXXXX")
critic_embedding = CustomizeEmbeddings("XXXXXXXXX")

# Evaluate
metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]
# Configure RunConfig
result = evaluate(
    dataset=test_dataset,
    metrics=metrics,
    raise_exceptions=True,
    llm=critic_llm,
    embeddings=critic_embedding,
    run_config=RunConfig(max_workers=5, timeout=120, max_retries=3),
)

df = result.to_pandas()
print(f"- Eval Result: {result}\n\n- Detail:\n{df}")

# Visualize Result
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

df.index = df.index + 1
heatmap_data = df[[x.name for x in metrics]]

cmap = LinearSegmentedColormap.from_list("green_red", ["red", "green"])

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap)

plt.yticks(ticks=range(len(df.index)), labels=df.index, rotation=0)

plt.show()





