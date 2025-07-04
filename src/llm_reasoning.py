import ollama
import time

PROMPT_CHUNK = """
You are a video reasoning assistant.
Here is the context timeline:
{context}

User Question: {query}
Please answer using only this context and do not invent beyond timeline.
"""

# Recursive chunk summarization
def ask_with_chunks(context_chunks: list, query: str) -> str:
    summary = ""
    for chunk in context_chunks:
        prompt = PROMPT_CHUNK.format(
            context=chunk if not summary else f"Previous Summary:\n{summary}\n\n{chunk}",
            query=query
        )
        resp = ollama.chat(model='mistral', messages=[{'role':'user','content':prompt}])
        summary = resp['message']['content']
        time.sleep(0.2)
    return summary