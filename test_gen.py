import rag
import config
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

query = "Compare the parameter-efficient fine-tuning methods discussed in the provided papers, specifically focusing on their parameter efficiency and memory overhead. Provide a table."

try:
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Testing Query: {query}\n")
        f.write("-" * 60 + "\n")
        result = rag.answer_question(query, strategy="A", top_k=12)
        f.write(f"Chunks used: {result['chunks_used']}\n")
        f.write("Answer:\n")
        f.write(result['answer'] + "\n")
        f.write("-" * 60 + "\n")
except Exception as e:
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}\n")
