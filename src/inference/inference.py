import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import ollama


# ---------------------------------------------------------------------------
# 1. Data Schema
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Multi-Sample JSON Loader
# ---------------------------------------------------------------------------
def load_dataset_samples(filepath: str) -> List[Tuple[str, List[Chunk]]]:
    """
    Parses a JSON file (containing one or multiple samples) into a list of 
    tuples containing: (query_string, list_of_Chunk_objects).
    Compatible with multi-question HotpotQA arrays and standard dataset schemas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is treated as a list of samples
    samples = data if isinstance(data, list) else [data]
    dataset_records = []

    for sample in samples:
        query = sample.get("question", "")
        chunks: List[Chunk] = []

        # Schema A: HotpotQA structure
        if "context" in sample and isinstance(sample["context"], dict):
            titles = sample["context"].get("title", [])
            sentences_list = sample["context"].get("sentences", [])
            for idx, (title, sents) in enumerate(zip(titles, sentences_list), 1):
                chunks.append(
                    Chunk(
                        id=str(idx),
                        content=" ".join(sents).strip(),
                        metadata={"source": title}
                    )
                )

        # Schema B: Standard document array
        elif "documents" in sample and isinstance(sample["documents"], list):
            for idx, item in enumerate(sample["documents"], 1):
                if isinstance(item, str):
                    chunks.append(
                        Chunk(id=str(idx), content=item.strip(), metadata={"source": f"Doc_{idx}"})
                    )
                elif isinstance(item, dict):
                    content = item.get("content", item.get("text", ""))
                    metadata = item.get("metadata", {"source": item.get("source", f"Doc_{idx}")})
                    chunks.append(
                        Chunk(id=str(idx), content=content.strip(), metadata=metadata)
                    )

        # Fallback: Plain string context
        elif "context" in sample and isinstance(sample["context"], str):
            chunks.append(
                Chunk(id="1", content=sample["context"].strip(), metadata={"source": "context"})
            )

        if not chunks:
            raise ValueError(f"Could not extract chunks for sample ID: {sample.get('id', 'unknown')}. Check key names.")

        dataset_records.append((query, chunks))

    return dataset_records


# ---------------------------------------------------------------------------
# 3. RAG Generator Class
# ---------------------------------------------------------------------------
class RAGGenerator:
    def __init__(self, model_name: str = "qwen3.5:4b", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.client = ollama.Client()

    def _format_context(self, chunks: List[Chunk]) -> str:
        """Formats chunks into distinct labeled passages for attention isolation."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            source_tag = chunk.metadata.get("source", f"Chunk_{chunk.id}")
            formatted.append(f"[Doc {i}] (Source: {source_tag})\n{chunk.content.strip()}")
        return "\n\n".join(formatted)

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        system_prompt: Optional[str] = None
    ) -> str:
        context_str = self._format_context(retrieved_chunks)

        # Baseline system prompt tuned for Qwen context adherence
        default_system_prompt = (
            "You are a strict, grounded AI assistant answering questions solely based on provided excerpts.\n"
            "Rules:\n"
            "1. Answer ONLY using facts explicitly stated in the <context> tags.\n"
            "2. Anchor every claim with inline bracket citations matching the document tag, e.g., [Doc 1] or [Doc 2].\n"
            "3. If the context does not contain enough information, respond exactly with:\n"
            "   'INSUFFICIENT_CONTEXT: The provided documents do not contain this information.'\n"
            "4. Do NOT use outside knowledge, speculate, or extrapolate beyond what is stated."
        )

        sys_prompt = system_prompt or default_system_prompt

        user_content = (
            f"<context>\n"
            f"{context_str}\n"
            f"</context>\n\n"
            f"User Question: {query}\n"
            f"Answer:"
        )

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            options={
                "temperature": self.temperature,
                "num_ctx": 8192
            }
        )

        # Handle SDK response format safely
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content.strip()
        return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 4. Execution Loop for Multiple Dataset Questions
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    json_path = "dataset.json"

    # Ensure the file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Please ensure your multi-question dataset is saved at '{json_path}'.")

    # 1. Load all records (queries and chunks) from the dataset JSON file
    dataset_records = load_dataset_samples(json_path)
    print(f"Loaded {len(dataset_records)} question(s) from '{json_path}'.\n")

    # 2. Instantiate generator
    gen = RAGGenerator(model_name="qwen3.5:4b", temperature=0.1)

    # 3. Loop through and run generation for each question in the dataset
    for i, (dataset_query, chunks) in enumerate(dataset_records, 1):
        print("=" * 60)
        print(f"[Test {i}] Running Dataset Query: \"{dataset_query}\"")
        print(f"Retrieved {len(chunks)} context chunks.")
        
        output = gen.generate(query=dataset_query, retrieved_chunks=chunks)
        
        print("\nModel Output:")
        print(output)
        print("=" * 60 + "\n")