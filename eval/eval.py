#evalutaion of LegalRAG using RAGAS

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy, context_precision, context_recall,)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr
from pipeline.rag_pipeline import BasicRAGPipeline
from configs.config import GROQ_MODEL, DATA_DIR, EMBEDDING_MODEL

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.json")
TOP_K = 3
USE_RERANKING = True

#Loading test data
def load_test_data(path):
    with open(path, "r") as f:
        return json.load(f)

#initializing the pipeline
def init_pipeline():
    print("Initializing pipeline...")
    pipeline = BasicRAGPipeline(
        data_dir = DATA_DIR, 
        rag_type = "naive-rag"
    )
    pipeline.retriever.index_pdfs()
    print("Pipeline ready")
    return pipeline

#Running pipeline on each question
def run_pipeline(pipeline, test_data):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"\n Running pipeline on {len(test_data)} questions....")

    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[(i+1)/{len(test_data)}] {question}")

        try:
            result = pipeline.answer(
                query = question,
                chat_history = None,
                top_k = TOP_K,
                use_reranking = USE_RERANKING,
                use_query_rewriting = False
            )
            answer = result["answer"]
            context = result.get("contexts", [])

            #if context is not present retrieve it 
            if not context:
                retrieved = pipeline.retriever.retrieve(
                    question, top_k = TOP_K, use_reranking = USE_RERANKING
                )
                context = [r["content"] for r in retrieved]

        except Exception as e:
            print(f"Error: {e}")
            answer = ""
            context = []

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    return questions, answers, contexts, ground_truths

#building the dataset required for evaluation
def build_dataset(questions, answers, contexts, ground_truths):
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

#Running RAGAS Evaluation
def run_evaluation(dataset):
    print("\nRunning RAGAS evaluation...")

    groq_key = os.getenv("GROQ_API_KEY")
    api_key = SecretStr(groq_key) if groq_key else None

    #RAGAS LLM and EMBEDDING WRAPPER
    llm = LangchainLLMWrapper(
        ChatGroq(temperature = 0.0, model= GROQ_MODEL, api_key = api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    )

    results = evaluate(
        dataset = dataset,
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm = llm,
        embeddings = embeddings,
    )

    return results

#Displaying the results 
def print_results(results):
    print("\n" + "="*50)
    print("RAGAS Evaluation results")
    print("="*50)

    scores = results.to_pandas()

    metrics = {
        "Faithfullness": scores["faithfullness"].mean(),
        "Answer Relevancy": scores["answer_relevancy"].mean(),
        "Context Precision": scores["context_precision"].mean(),
        "Context Recall": scores["context_recall"].mean(),
    }

    for metric, score in metrics.items():
        bar = "#" * int(score*20)
        empty = "-" * (20 - int(score * 20))
        print(f"{metric: < 20} [{bar}{empty}] {score:.3f}")

    print("-"*50)
    print(f"Overall average: {sum(metrics.values())/ len(metrics):.3f}")
    print("="*50)

    #Save the results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    scores.to_csv(output_path, index = False)
    print(f"\nDetailed results saved to : {output_path}")

#main 
if __name__ == "__main__":
    test_data = load_test_data(TEST_DATA_PATH)
    pipeline = init_pipeline()

    questions, answers, contexts, ground_truths = run_pipeline(pipeline, test_data)
    dataset = build_dataset(questions, answers, contexts, ground_truths)
    results = run_evaluation(dataset)

    print_results(results)