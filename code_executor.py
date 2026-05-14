import json
import subprocess
import time
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

DOCKER_IMAGE = "code-sandbox:latest"
CONTAINER_NAME = "code-sandbox-runner"
TIMEOUT_SECONDS = 5


def run_local_python(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute Python code locally (Fallback when Docker is missing)."""
    results = []
    
    # Create a temporary script that includes the user code and test runner
    runner_template = """
import json
import sys

{user_code}

def run_tests():
    test_cases = {test_cases}
    results = []
    
    # We assume the user defined a function, or we'll try to find a function to call
    import inspect
    import __main__
    
    funcs = [obj for name, obj in inspect.getmembers(__main__) if inspect.isfunction(obj) and obj.__module__ == '__main__']
    
    if not funcs:
        print(json.dumps({{"success": False, "error": "No function defined"}}))
        return

    target_func = funcs[0] # Pick the first defined function
    
    for tc in test_cases:
        try:
            # Handle empty inputs
            if not tc.get("input"):
                output = target_func()
            else:
                # Basic parsing for common input types
                args = tc["input"].split(",")
                processed_args = []
                for a in args:
                    a = a.strip()
                    try:
                        if '.' in a: processed_args.append(float(a))
                        else: processed_args.append(int(a))
                    except:
                        processed_args.append(a)
                
                output = target_func(*processed_args)
            
            results.append({{
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": str(output),
                "passed": str(output).strip().lower() == str(tc["expected"]).strip().lower()
            }})
        except Exception as e:
            results.append({{
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": f"Error: {{str(e)}}",
                "passed": False
            }})
            
    print(json.dumps({{"success": True, "test_results": results}}))

if __name__ == "__main__":
    run_tests()
"""
    
    full_code = runner_template.format(
        user_code=code,
        test_cases=json.dumps(test_cases)
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name
        
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {"success": True, "output": result.stdout, "test_results": []}
        else:
            return {"success": False, "error": result.stderr or "Local execution failed"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Execution timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def run_docker_container(code: str, language: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute code in a Docker container with timeout."""
    
    if language not in ["python", "cpp"]:
        return {"success": False, "error": f"Unsupported language: {language}"}

    try:
        input_data = json.dumps({"code": code, "test_cases": test_cases})
        
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-e", f"CODE_INPUT={input_data}",
                "-e", f"LANGUAGE={language}",
                DOCKER_IMAGE,
                language
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS + 2
        )
        
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                return output
            except json.JSONDecodeError:
                return {"success": True, "output": result.stdout, "test_results": []}
        else:
            stderr = result.stderr or ""
            if "docker: error during connect" in stderr or "The system cannot find the file specified" in stderr:
                return {"success": False, "error": "DOCKER_NOT_RUNNING", "details": stderr}
            return {"success": False, "error": stderr or "Execution failed"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds"}
    except FileNotFoundError:
        return {"success": False, "error": "DOCKER_NOT_INSTALLED"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_sandbox_image():
    """Build the sandbox Docker image."""
    dockerfile_path = Path(__file__).parent / "Dockerfile.sandbox"
    
    try:
        subprocess.run(
            ["docker", "build", "-t", DOCKER_IMAGE, "-f", str(dockerfile_path), str(Path(__file__).parent)],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e.stderr}")
        return False


def execute_python(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute Python code with test cases, falling back to local if Docker fails."""
    docker_res = run_docker_container(code, "python", test_cases)
    
    if not docker_res.get("success") and docker_res.get("error") in ["DOCKER_NOT_RUNNING", "DOCKER_NOT_INSTALLED"]:
        print(f"[code_executor] Docker unavailable ({docker_res.get('error')}). Falling back to local execution...")
        return run_local_python(code, test_cases)
        
    return docker_res


def execute_cpp(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute C++ code with test cases."""
    # C++ fallback is harder (requires compiler), so we just try Docker
    return run_docker_container(code, "cpp", test_cases)


def execute_code(code: str, language: str, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Execute code in the specified language."""
    if test_cases is None:
        test_cases = [{"input": "", "expected": ""}]
    
    if language.lower() in ["python", "py"]:
        return execute_python(code, test_cases)
    elif language.lower() in ["cpp", "c++"]:
        return execute_cpp(code, test_cases)
    else:
        return {"success": False, "error": f"Unsupported language: {language}"}