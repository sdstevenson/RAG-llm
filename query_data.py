import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

import time

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    start_time = time.time()
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    print(print("--- %s seconds --- to search chroma databse" % (time.time() - start_time)))

    start_time = time.time()
    # Create the query
    print(results[0])
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    print(print("--- %s seconds --- to create query" % (time.time() - start_time)))

    # Change the model here.
    model = Ollama(model="llama3:8b")
    print("\n---\n")
    # Invoke the model
    start_time = time.time()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    print(print("\n--- %s seconds --- to generate answer" % (time.time() - start_time)))
    return response_text


if __name__ == "__main__":
    main()
