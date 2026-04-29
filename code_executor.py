import json
import subprocess
import time
from typing import Dict, List, Any, Optional

DOCKER_IMAGE = "code-sandbox:latest"
CONTAINER_NAME = "code-sandbox-runner"
TIMEOUT_SECONDS = 5


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
            return {"success": False, "error": result.stderr or "Execution failed"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds"}
    except FileNotFoundError:
        return {"success": False, "error": "Docker not installed or not running"}
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
    """Execute Python code with test cases."""
    return run_docker_container(code, "python", test_cases)


def execute_cpp(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute C++ code with test cases."""
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