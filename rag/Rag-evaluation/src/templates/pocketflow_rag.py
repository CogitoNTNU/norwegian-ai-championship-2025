"""
PocketFlow RAG Template for evaluation integration - Standalone version
"""

import os
import json
import subprocess
import tempfile


class PocketFlowRAG:
    def __init__(self, llm, embeddings):
        """
        Initialize PocketFlow RAG template
        Args:
            llm: RAGAS-provided LLM (we'll ignore this and use our Google client)
            embeddings: RAGAS-provided embeddings (we'll ignore this and use our system)
        """
        self.llm = llm
        self.embeddings = embeddings
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
        Run PocketFlow RAG system for evaluation using subprocess
        Args:
            question: User query string
            context: List of reference context strings (from testset)
        Returns:
            dict with 'answer' and 'context' keys for RAGAS evaluation
        """
        try:
            # Create a temporary script to run the PocketFlow query
            script_content = f'''
import sys
import os
sys.path.insert(0, "{self.custom_rag_path}")
os.chdir("{self.custom_rag_path}")

try:
    from flows import run_online_query
    import json
    
    question = {repr(question)}
    result = run_online_query(question)
    
    # Output the result as JSON
    output = {{
        "answer": result,
        "context": ["Retrieved from PocketFlow RAG system"],
        "status": "success"
    }}
    print(json.dumps(output))
    
except Exception as e:
    import traceback
    output = {{
        "answer": f"Error: {{str(e)}}",
        "context": {repr(context[:3] if context else [])},
        "status": "error",
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(output))
'''

            # Write temporary script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                temp_script = f.name

            try:
                # Run the script in subprocess using uv's Python
                result = subprocess.run(
                    ["uv", "run", "python", temp_script],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.custom_rag_path,
                )

                if result.returncode == 0:
                    # Debug: print raw output
                    stdout_content = result.stdout.strip()
                    if not stdout_content:
                        print("Warning: Empty stdout from subprocess")
                        return {
                            "answer": "Empty response from PocketFlow RAG",
                            "context": context[:3] if context else [],
                        }

                    try:
                        # Extract JSON from stdout (it may contain debug messages)
                        json_line = None
                        for line in stdout_content.split("\n"):
                            line = line.strip()
                            if line.startswith('{"') and line.endswith("}"):
                                json_line = line
                                break

                        if json_line:
                            output = json.loads(json_line)
                            return {
                                "answer": output["answer"],
                                "context": output.get(
                                    "context", ["Retrieved from PocketFlow RAG"]
                                ),
                            }
                        else:
                            # No JSON found, but PocketFlow might have generated text output
                            return {
                                "answer": stdout_content,
                                "context": ["Retrieved from PocketFlow RAG"],
                            }

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Raw stdout: '{stdout_content[:500]}...'")
                        return {
                            "answer": f"JSON parse error, raw output: {stdout_content[:200]}...",
                            "context": context[:3] if context else [],
                        }
                else:
                    print(f"Subprocess failed with return code {result.returncode}")
                    print(f"Subprocess stderr: {result.stderr}")
                    print(f"Subprocess stdout: {result.stdout}")
                    return {
                        "answer": f"Subprocess failed: {result.stderr}",
                        "context": context[:3] if context else [],
                    }

            finally:
                # Clean up temp file
                os.unlink(temp_script)

        except subprocess.TimeoutExpired:
            return {
                "answer": "Query timed out after 60 seconds",
                "context": context[:3] if context else [],
            }
        except Exception as e:
            print(f"Error in PocketFlow RAG adapter: {e}")
            import traceback

            traceback.print_exc()

            # Return fallback response
            return {
                "answer": f"Adapter error: {str(e)}",
                "context": context[:3] if context else [],
            }
