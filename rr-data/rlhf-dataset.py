import json
import yaml
from util.Load_Model import CustomizeChat
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel
import re
import json

def load_testset_from_jsonl(filename: str):
    documents = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            documents.append(data)
    return documents

dataset = load_testset_from_jsonl("data/Redfish_testset_1000.jsonl")
dataset_q = [x['messages'][0]['content'] for x in dataset]

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_prompt(prompts, prompt_name):
    return prompts['prompts'][prompt_name]['text']

prompts = load_prompts("prompts.yaml")

llm_client = CustomizChat("XXXXXXXXX")
prompt = ChatPromptTemplate.from_template(prompt_relevance)

def call_llm(inference_client, input_):
    responese = inference_client.invoke(input_)
    return responese.content, responese.usage_metadata

outputs = list()
usages = list()
for input_ in dataset_q[:30]:
    evaluations = {
        "relevance": call_llm(
            llm_client,
            prompt_relevance.format(question=input_),
        ),
        "standalone": call_llm(
            llm_client,
            prompt_standalone.format(question=input_),
        ),
        "overall": call_llm(
            llm_client,
            prompt_overall.format(question=input_),
        ),
    }
    output = {"question":input_,}
    try:
        for criterion, response_pack in evaluations.items():
            evaluation, usage = response_pack
            print(evaluation)
            usages.append(usage)
            score, eval = (
                int(evaluation.split('[RESULT]')[-1].strip().strip('()')),
                evaluation.split('[RESULT]')[0].strip(),
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
        outputs.append(output)
    except Exception as e:
        outputs.append(output)
        continue

# total token usage:
tokens = []
for x in usages:
    for k,v in x.items():
        tokens.append(v)

# print(sum(tokens))

generated_questions = pd.DataFrame.from_dict(outputs)
critic_generated_questions = generated_questions.loc[
    (generated_questions["overall_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)
]

# not qualify question 
generated_questions.loc[
    (generated_questions["overall_score"] < 4)
    | (generated_questions["relevance_score"] < 4)
    | (generated_questions["standalone_score"] < 4)
]


generated_questions = pd.read_excel('data/rlhf-critic_q.xlsx')

critic_generated_questions = generated_questions.loc[
    (generated_questions["overall_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)
]

PROJECT_ID = "XXXXXXXXX"
API_key = "XXXXXXXXX-XXXXXXXXX"
BASE_MODEL_NAME = "XXXXXXXXX"
PRETRAIN_MODEL_NAME = (
    "XXXXXXXXX"
)
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

base_model = GenerativeModel(BASE_MODEL_NAME)
pretrain_model = GenerativeModel(PRETRAIN_MODEL_NAME)

prompt_answering = get_prompt(prompts, "answering_prompt")

# prepare question list
question_ls = [row.question for row in critic_generated_questions.itertuples()]

results = list()
token_used = list()
for q in question_ls:
    resp1 = base_model.generate_content(prompt_answering.format(question=q))
    resp2 = pretrain_model.generate_content(prompt_answering.format(question=q))

    results.append(
        {
            "instruction": q,
            "response1":resp1.text,
            "response2":resp2.text
        }
    )
    token_used += [resp1.usage_metadata.total_token_count, resp2.usage_metadata.total_token_count]

prompt_judge = get_prompt(prompts, "judge_prompt")

token_used = list()
for x in results:
    response_texts = call_llm(
        llm_client,
        prompt_judge.format(instruction=x["instruction"], response1=x["response1"], response2=x["response2"])
    )
    x["judge"] = response_texts[0]
    token_used.append(response_texts[1]['total_tokens'])



def extract_feedback(text):
    response1_feedback = re.search(r'Feedback1: (.*?)Score1:', text, re.DOTALL)
    response1_feedback = response1_feedback.group(1).strip() if response1_feedback else None

    response1_score = re.search(r'Score1: (.*?)Feedback2:', text, re.DOTALL)
    response1_score = response1_score.group(1).strip() if response1_score else None

    response2_feedback = re.search(r'Feedback2: (.*?)Score2:', text, re.DOTALL)
    response2_feedback = response2_feedback.group(1).strip() if response2_feedback else None

    response2_score = re.search(r'Score2: (.*?)chosen:', text, re.DOTALL)
    response2_score = response2_score.group(1).strip() if response2_score else None

    chosen = re.search(r'chosen: (.*?)reject:', text, re.DOTALL)
    chosen = chosen.group(1).strip() if chosen else None

    reject = re.search(r'reject:\s*(\w+)', text, re.DOTALL)
    reject = reject.group(1).strip() if reject else None

    return {
        "Response 1 Feedback": response1_feedback,
        "Response 1 Score": response1_score,
        "Response 2 Feedback": response2_feedback,
        "Response 2 Score": response2_score,
        "chosen": chosen,
        "reject": reject
    }

datasets_full = list()
for x in results:
    tmp = extract_feedback(x["judge"])
    tmp['Response1'] = x["response1"] 
    tmp['Response2'] = x["response2"] 
    tmp['instruction'] = x['instruction']
    datasets_full.append(tmp)
    x["chosen"] = tmp["chosen"]
    del x["judge"]


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

save_testset_to_jsonl(results, "data/rlhf-dpo-dataset.jsonl")

import pandas as pd
df = pd.DataFrame(datasets_full)
df = df[['instruction','Response1', 'Response 1 Feedback', 'Response 1 Score', 'Response2', 'Response 2 Feedback', 'Response 2 Score', 'chosen', 'reject']]
df.to_excel('data/rlhf-dpo-dataset.xlsx', index=False)






