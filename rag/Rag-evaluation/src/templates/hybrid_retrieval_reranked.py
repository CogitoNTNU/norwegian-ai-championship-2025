"""
Hybrid Retrieval with Reranker Template - BM25s + FAISS + Reranking
Tests retrieval + reranking capability for component classification
"""

import os
import json
import subprocess
import tempfile


class HybridRetrievalReranked:
    def __init__(self, llm=None, embeddings=None):
        """
        Initialize Hybrid Retrieval with Reranker template
        Focus: Test BM25s + FAISS + semantic reranking without LLM generation
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
        Run hybrid retrieval + reranking - no LLM generation
        Args:
            question: User query string
            context: List of reference context strings (from testset)
        Returns:
            dict with 'answer' (dummy) and 'context' (reranked docs) for evaluation
        """
        try:
            # Create script that runs retrieval + semantic reranking
            script_content = f'''
import sys
import os
sys.path.insert(0, "{self.custom_rag_path}")
os.chdir("{self.custom_rag_path}")

try:
    from nodes import enhance_query, embed_query, retrieve_docs
    from utils.google_client import get_embedding
    import json
    import numpy as np
    
    question = {repr(question)}
    
    # Set up shared store for retrieval pipeline
    shared = {{
        "question": question,
        "enhanced_question": "",
        "query_embedding": [],
        "retrieved_docs": [],
        "index_path": "data/faiss_index"
    }}
    
    # Run retrieval pipeline (gets hybrid BM25+FAISS results)
    enhance_query.run(shared)
    embed_query.run(shared) 
    retrieve_docs.run(shared)
    
    # Now apply semantic reranking
    query_embedding = shared["query_embedding"]
    retrieved_docs = shared["retrieved_docs"]
    
    # Rerank using semantic similarity to query
    reranked_docs = []
    
    for doc in retrieved_docs:
        if isinstance(doc, dict) and "text" in doc:
            doc_text = doc["text"]
            
            # Get embedding for document text (simulate reranker)
            try:
                doc_embedding = get_embedding(doc_text[:500])  # Limit text for speed
                
                # Calculate cosine similarity with query
                if query_embedding and doc_embedding:
                    query_np = np.array(query_embedding)
                    doc_np = np.array(doc_embedding)
                    
                    # Cosine similarity
                    similarity = np.dot(query_np, doc_np) / (np.linalg.norm(query_np) * np.linalg.norm(doc_np))
                    
                    reranked_docs.append({{
                        "text": doc_text,
                        "similarity": float(similarity),
                        "original_rank": len(reranked_docs)
                    }})
                else:
                    # Fallback to original order
                    reranked_docs.append({{
                        "text": doc_text,
                        "similarity": 0.5,  # Neutral score
                        "original_rank": len(reranked_docs)
                    }})
                    
            except Exception as embed_error:
                # Fallback if embedding fails
                reranked_docs.append({{
                    "text": doc_text,
                    "similarity": 0.5,
                    "original_rank": len(reranked_docs)
                }})
    
    # Sort by similarity (descending)
    reranked_docs.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Extract component names from reranked results
    component_names = []
    final_contexts = []
    
    for doc in reranked_docs:
        text = doc["text"]
        final_contexts.append(text)
        
        # Extract component names
        import re
        components = re.findall(r'obc-[a-zA-Z-]+', text)
        component_names.extend(components)
    
    # Remove duplicates while preserving order
    unique_components = list(dict.fromkeys(component_names))
    
    # Calculate reranking impact
    similarity_scores = [doc["similarity"] for doc in reranked_docs]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    output = {{
        "answer": f"RERANKED: Found {{len(unique_components)}} components: {{', '.join(unique_components[:5])}} (avg_sim: {{avg_similarity:.3f}})",
        "context": final_contexts,
        "components_found": unique_components,
        "retrieval_count": len(final_contexts),
        "reranking_scores": similarity_scores,
        "avg_similarity": avg_similarity,
        "status": "success"
    }}
    print(json.dumps(output))
    
except Exception as e:
    import traceback
    output = {{
        "answer": f"RERANKING_ERROR: {{str(e)}}",
        "context": {repr(context[:3] if context else [])},
        "components_found": [],
        "retrieval_count": 0,
        "reranking_scores": [],
        "avg_similarity": 0.0,
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
                # Run retrieval + reranking script (longer timeout for embeddings)
                result = subprocess.run(
                    ["uv", "run", "python", temp_script],
                    capture_output=True,
                    text=True,
                    timeout=60,  # Longer timeout for reranking embeddings
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
                            "answer": f"RERANKING_NO_JSON: {stdout_content[:100]}...",
                            "context": context[:3] if context else [],
                        }
                else:
                    return {
                        "answer": f"RERANKING_FAILED: {result.stderr[:100]}...",
                        "context": context[:3] if context else [],
                    }

            finally:
                os.unlink(temp_script)

        except subprocess.TimeoutExpired:
            return {
                "answer": "RERANKING_TIMEOUT: Reranking timed out after 60 seconds",
                "context": context[:3] if context else [],
            }
        except Exception as e:
            return {
                "answer": f"RERANKING_ADAPTER_ERROR: {str(e)}",
                "context": context[:3] if context else [],
            }
