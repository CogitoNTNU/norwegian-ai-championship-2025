"""
Hybrid Retrieval Only Template - BM25s + FAISS without LLM generation
Tests pure retrieval capability for component classification
"""

import os
import json
import subprocess
import tempfile


class HybridRetrievalOnly:
    def __init__(self, llm=None, embeddings=None):
        """
        Initialize Hybrid Retrieval Only template
        Focus: Test BM25s + FAISS hybrid retrieval without LLM generation
        """
        self.llm = llm  # Ignored - no LLM generation
        self.embeddings = embeddings  # Ignored - using PocketFlow embeddings
        self.custom_rag_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "RAG-chat-frontend-backend",
                "custom-rag",
            )
        )

    def run(self, question, context):
        """
        Run hybrid retrieval only - no LLM generation
        Args:
            question: User query string
            context: List of reference context strings (from testset)
        Returns:
            dict with 'answer' (dummy) and 'context' (retrieved docs) for evaluation
        """
        try:
            # Create script that only runs retrieval pipeline
            script_content = f'''
import sys
import os
sys.path.insert(0, "{self.custom_rag_path}")
os.chdir("{self.custom_rag_path}")

try:
    from nodes import enhance_query, embed_query, retrieve_docs
    import json
    
    question = {repr(question)}
    
    # Set up shared store for retrieval pipeline only
    shared = {{
        "question": question,
        "enhanced_question": "",
        "query_embedding": [],
        "retrieved_docs": [],
        "index_path": "data/faiss_index"
    }}
    
    # Run retrieval pipeline (no LLM generation)
    enhance_query.run(shared)
    embed_query.run(shared) 
    retrieve_docs.run(shared)
    
    # Extract retrieved contexts for evaluation
    retrieved_contexts = []
    component_names = []
    
    for doc in shared["retrieved_docs"]:
        if isinstance(doc, dict) and "text" in doc:
            text = doc["text"]
            retrieved_contexts.append(text)
            
            # Extract component names from retrieved text
            import re
            components = re.findall(r'obc-[a-zA-Z-]+', text)
            component_names.extend(components)
    
    # Remove duplicates while preserving order
    unique_components = list(dict.fromkeys(component_names))
    
    output = {{
        "answer": f"RETRIEVAL_ONLY: Found {{len(unique_components)}} components: {{', '.join(unique_components[:5])}}",
        "context": retrieved_contexts,
        "components_found": unique_components,
        "retrieval_count": len(retrieved_contexts),
        "status": "success"
    }}
    print(json.dumps(output))
    
except Exception as e:
    import traceback
    output = {{
        "answer": f"RETRIEVAL_ERROR: {{str(e)}}",
        "context": {repr(context[:3] if context else [])},
        "components_found": [],
        "retrieval_count": 0,
        "status": "error",
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(output))
'''

            # Write and execute temporary script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                temp_script = f.name

            try:
                # Run retrieval-only script
                result = subprocess.run(
                    ["uv", "run", "python", temp_script],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Shorter timeout since no LLM
                    cwd=self.custom_rag_path,
                )

                if result.returncode == 0:
                    stdout_content = result.stdout.strip()

                    # Extract JSON from output (look for the last complete JSON object)
                    json_line = None
                    lines = stdout_content.split("\\n")

                    # Look for JSON starting from the end
                    for line in reversed(lines):
                        line = line.strip()
                        if line.startswith('{"') and line.endswith("}"):
                            json_line = line
                            break

                    # If not found, try to find JSON that spans multiple lines
                    if not json_line:
                        json_start = -1
                        for i, line in enumerate(lines):
                            if line.strip().startswith('{"'):
                                json_start = i
                                break

                        if json_start >= 0:
                            # Try to reconstruct JSON from multiple lines
                            json_candidate = ""
                            for i in range(json_start, len(lines)):
                                json_candidate += lines[i].strip()
                                if json_candidate.endswith("}"):
                                    json_line = json_candidate
                                    break

                    if json_line:
                        output = json.loads(json_line)
                        return {
                            "answer": output["answer"],
                            "context": output.get("context", []),
                        }
                    else:
                        return {
                            "answer": f"RETRIEVAL_NO_JSON: {stdout_content[:100]}...",
                            "context": context[:3] if context else [],
                        }
                else:
                    return {
                        "answer": f"RETRIEVAL_FAILED: {result.stderr[:100]}...",
                        "context": context[:3] if context else [],
                    }

            finally:
                os.unlink(temp_script)

        except subprocess.TimeoutExpired:
            return {
                "answer": "RETRIEVAL_TIMEOUT: Retrieval timed out after 30 seconds",
                "context": context[:3] if context else [],
            }
        except Exception as e:
            return {
                "answer": f"RETRIEVAL_ADAPTER_ERROR: {str(e)}",
                "context": context[:3] if context else [],
            }
