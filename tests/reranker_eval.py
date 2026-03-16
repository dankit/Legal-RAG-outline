from datasets import load_dataset
from app.core.reranker import BGEReranker
import json
from pathlib import Path


def run_eval():
    """Evaluate BGE reranker accuracy on the CaseHOLD legal benchmark."""
    valid_dataset = load_dataset("coastalcph/lex_glue", "case_hold", split="validation")
    reranker = BGEReranker()
    correct = 0
    for row in valid_dataset:
        query = row["context"]
        documents = row["endings"]
        label = row["label"]
        reranked_documents = reranker.rerank_documents(query, documents, 5)
        if label == reranked_documents[0]["corpus_id"]:
            correct += 1
    print(f"Correct: {correct}, Total: {len(valid_dataset)}")


def create_reranker_finetune_dataset():
    """Convert CaseHOLD dataset to reranker fine-tuning format (query, pos, neg)."""
    reranker_dataset_location = Path("../Data/testing/reranker_case_hold_dataset.jsonl")
    dataset = load_dataset("coastalcph/lex_glue", "case_hold")
    train_dataset = dataset['train']
    formatted_datas = []
    for row in train_dataset:
        context = row['context']
        endings = row['endings']
        pos_docs = []
        neg_docs = []
        for i in range(len(endings)):
            if i == row['label']:
                pos_docs.append(endings[i])
            else:
                neg_docs.append(endings[i])
        formatted_data = {"query": context, "pos": pos_docs, "neg": neg_docs}
        formatted_datas.append(formatted_data)
    with open(reranker_dataset_location, "w") as f:
        for formatted_data in formatted_datas:
            f.write(json.dumps(formatted_data))
            f.write("\n")


if __name__ == "__main__":
    run_eval()
