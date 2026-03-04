import os
import sys
import json
import tempfile
import subprocess
from typing import Optional, Union, Dict, Any


class PythonInterpreter:
    name = "python_interpreter"
    description = """Execute Python code in a subprocess and return the execution results.

**Key Points:**
- Use print() statements for any output you want to see
- Standard Python libraries are available (e.g., math, json, re, datetime, collections, requests, statistics, itertools, functools, etc.)
- **DO NOT create, modify, or delete files** - keep all operations in-memory
- Focus on computation, data processing, and calculations
""".strip()
    
    parameters = [
        {
            "name": "code",
            "type": "string",
            "description": (
                "The Python code to execute. Must be provided as a string. "
                "Remember to use print() statements for any output you want to see. "
                "Do not include file I/O operations (no file creation, modification, or deletion)."
            ),
            "required": True,
        }
    ]
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the Python interpreter tool.
        
        Args:
            timeout: Maximum execution time in seconds (default: 30)
        """
        self.timeout = timeout
    
    def call(self, params: Union[str, Dict[str, Any]], timeout: Optional[int] = None) -> str:
        """
        Execute Python code in a subprocess.
        
        Args:
            params: Either a Python code string or a dictionary containing 'code' key
            timeout: Optional override for execution timeout
            
        Returns:
            Execution results including stdout/stderr
        """
        # Extract code from input
        code = params.get('code', '')
        if not code or not code.strip():
            return "[python_interpreter] Error: no 'code' field found in the input arguments or the input code is empty."
        
        # Use instance timeout if not overridden
        exec_timeout = timeout or self.timeout
        
        try:
            # Create a temporary file to hold the code
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(code)
                tmp_file_path = tmp_file.name
            
            try:
                # Execute the code in a subprocess
                result = subprocess.run(
                    [sys.executable, tmp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=exec_timeout,
                    cwd=os.getcwd()
                )
                
                # Collect output
                output_parts = []
                
                if result.stdout:
                    output_parts.append(f"stdout:\n{result.stdout}")
                
                if result.stderr:
                    output_parts.append(f"stderr:\n{result.stderr}")
                
                if result.returncode != 0:
                    output_parts.append(f"[Exit Code: {result.returncode}]")
                
                final_output = '\n'.join(output_parts) if output_parts else "Execution completed (stdout printed nothing)."
                
                return final_output.strip()
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
        
        except subprocess.TimeoutExpired:
            return f"[python_interpreter] Error: execution timed out after {exec_timeout} seconds."
        
        except Exception as e:
            return f"[python_interpreter] Error: execution failed: {str(e)}"