## contextual keyword gate

import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
import pandas as pd


def get_contextual_segments(text):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())

    segments = []

    # Get noun phrases
    segments.extend([chunk.text for chunk in doc.noun_chunks])

    # Get named entities
    segments.extend([ent.text for ent in doc.ents])

    # Get clauses and meaningful phrases based on dependency parsing
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ["ROOT", "xcomp", "ccomp", "advcl"]:
                # Get the subtree of this token
                phrase = " ".join([t.text for t in token.subtree])
                segments.append(phrase)

    # optional: Remove duplicates and very short segments
    # segments = list(set([seg.strip() for seg in segments if len(seg.split()) > 1]))

    return segments


def weighted_keyword_score(mandatory_keywords, answer, weights, optional_keywords=None):
    """
    Computes a weighted keyword score using contextual segments
    """
    if optional_keywords is None:
        optional_keywords = []

    answer_segments = get_contextual_segments(answer)
    
    # Safety check: if no segments, use the full answer as a single segment
    if not answer_segments:
        answer_segments = [answer.strip()] if answer.strip() else [""]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    answer_embeddings = model.encode(answer_segments, convert_to_tensor=True)

    mandatory_embeddings = model.encode(mandatory_keywords, convert_to_tensor=True)
    mandatory_similarities = []

    for keyword_emb in mandatory_embeddings:
        similarities = [
            util.pytorch_cos_sim(keyword_emb, ans_emb).item()
            for ans_emb in answer_embeddings
        ]

        # Safety check for empty similarities list
        if similarities:
            mandatory_similarities.append(max(similarities))
        else:
            # If no answer segments, assign a very low similarity score
            mandatory_similarities.append(0.0)

    optional_similarities = []
    if optional_keywords:
        optional_embeddings = model.encode(optional_keywords, convert_to_tensor=True)
        for keyword_emb in optional_embeddings:
            similarities = [
                util.pytorch_cos_sim(keyword_emb, ans_emb).item()
                for ans_emb in answer_embeddings
            ]

            # Safety check for empty similarities list
            if similarities:
                optional_similarities.append(max(similarities))
            else:
                # If no answer segments, assign a very low similarity score
                optional_similarities.append(0.0)

    mandatory_score = np.mean(mandatory_similarities) * weights["mandatory"]
    optional_score = (
        np.mean(optional_similarities) * weights["optional"]
        if optional_similarities
        else weights["optional"] # give it full beans if no optional keywords
    )

    total_score = mandatory_score + optional_score

    return total_score, mandatory_score, optional_score, answer_segments


# Parameters
weights = {
    "mandatory": 0.8,
    "optional": 0.2,
}
contextual_relevance_threshold = 0.6

df_res = pd.read_excel("my_custom_testset.xlsx")
df_res = df_res[
    [
        "question",
        "contexts",
        "answer",
        "ground_truth",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "kw",
        "kw_metric",
        "weighted_average_score",
    ]
]

scores = []
report_data = []

# start the gate calculation
for index, row in df_res.iterrows():
    keywords = str(row["kw"]).strip("[]").split(",")
    mandatory_keywords = [k.strip().strip("'") for k in keywords]
    answer = str(row["answer"])

    total_score, mandatory_score, optional_score, answer_segments = (
        weighted_keyword_score(mandatory_keywords, answer, weights)
    )

    contextual_keyword_pass = total_score >= contextual_relevance_threshold

    scores.append(
        (total_score, mandatory_score, optional_score, contextual_keyword_pass)
    )

    print(f"Total Contextual Score for index {index}: {total_score:.2f}")
    print(f"Mandatory Keyword Score for index {index}: {mandatory_score:.2f}")
    print(f"Optional Keyword Score for index {index}: {optional_score:.2f}")
    print(f"Contextual Keyword Gate Pass for index {index}: {contextual_keyword_pass}")

    report_content = (
        f"Total Contextual Score for index {index}: {total_score:.2f}\n"
        f"Mandatory Keyword Score for index {index}: {mandatory_score:.2f}\n"
        f"Optional Keyword Score for index {index}: {optional_score:.2f}\n"
        f"Contextual Keyword Gate Pass for index {index}: {contextual_keyword_pass}\n"
    )
    with open("contextual_keyword_report-3.txt", "a") as report_file:
        report_file.write(
            f"Index: {index}, Answer Segments: {answer_segments}, Mandatory Keywords: {mandatory_keywords}\n"
            + report_content
        )

    report_data.append(
        {
            "Index": index,
            "Answer Segments": answer_segments,
            "Mandatory Keywords": mandatory_keywords,
            "Total Contextual Score": total_score,
            "Mandatory Keyword Score": mandatory_score,
            "Optional Keyword Score": optional_score,
            "Contextual Keyword Gate Pass": contextual_keyword_pass,
        }
    )

mean_scores = (
    np.mean([score[0] for score in scores]),
    np.mean([score[1] for score in scores]),
    np.mean([score[2] for score in scores]),
    np.mean([1 if score[3] else 0 for score in scores]),
)

print(f"Mean Total Score: {mean_scores[0]:.2f}")
print(f"Mean Mandatory Score: {mean_scores[1]:.2f}")
print(f"Mean Optional Score: {mean_scores[2]:.2f}")
print(f"Mean Pass Rate: {mean_scores[3] * 100:.2f}%")


with open("contextual_keyword_report-3.txt", "a") as report_file:
    report_file.write(f"Mean Total Score: {mean_scores[0]:.2f}\n")
    report_file.write(f"Mean Mandatory Score: {mean_scores[1]:.2f}\n")
    report_file.write(f"Mean Optional Score: {mean_scores[2]:.2f}\n")
    report_file.write(f"Mean Pass Rate: {mean_scores[3] * 100:.2f}%\n")


report_data.append(
    {
        "Index": "Mean",
        "Answer Segments": "",
        "Mandatory Keywords": "",
        "Total Contextual Score": mean_scores[0],
        "Mandatory Keyword Score": mean_scores[1],
        "Optional Keyword Score": mean_scores[2],
        "Contextual Keyword Gate Pass": mean_scores[3] * 100,  # percentage
    }
)

report_df = pd.DataFrame(report_data)
report_df.to_excel("contextual_keyword_report.xlsx", index=False)
