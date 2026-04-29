#!/bin/bash

set -e

LANGUAGE="${LANGUAGE:-python}"
INPUTJson="${CODE_INPUT:-}"

if [ -z "$INPUTJson" ]; then
    echo '{"success": false, "error": "No input provided"}'
    exit 1
fi

CODE=$(echo "$INPUTJson" | python3 -c "import sys, json; print(json.load(sys.stdin).get('code', ''))" 2>/dev/null || echo "")
TEST_CASES=$(echo "$INPUTJson" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin).get('test_cases', [])))" 2>/dev/null || echo "[]")

if [ "$LANGUAGE" = "python" ]; then
    python3 -c "
import json
import sys
import traceback

test_cases = json.loads('''$TEST_CASES''')
code = '''$CODE'''

results = []
all_passed = True

try:
    exec_globals = {}
    exec(code, exec_globals)
    
    for i, tc in enumerate(test_cases):
        input_val = tc.get('input', '')
        expected = tc.get('expected', '')
        
        if 'main' in exec_globals:
            actual = str(exec_globals['main'](input_val))
        else:
            # Fallback to look for a function named the same as input or generic
            actual = ''
        
        passed = str(actual).strip() == str(expected).strip()
        if not passed:
            all_passed = False
        
        results.append({
            'test': i + 1,
            'input': input_val,
            'expected': str(expected),
            'actual': actual,
            'passed': passed
        })
    
    print(json.dumps({'success': True, 'output': 'Executed', 'test_results': results, 'all_passed': all_passed}))

except Exception as e:
    print(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}))
"
    
elif [ "$LANGUAGE" = "cpp" ]; then
    # Write code to a file
    echo "$CODE" > /tmp/code.cpp
    
    # Compile
    g++ -o /tmp/code_exec /tmp/code.cpp -std=c++17 2>/tmp/compile_err.log
    
    if [ $? -eq 0 ]; then
        # Run test cases using python to handle JSON and execution
        python3 -c "
import json
import subprocess
import sys

test_cases = json.loads('''$TEST_CASES''')
results = []
all_passed = True

for i, tc in enumerate(test_cases):
    input_val = str(tc.get('input', ''))
    expected = str(tc.get('expected', '')).strip()
    
    try:
        process = subprocess.run(['/tmp/code_exec'], input=input_val, text=True, capture_output=True, timeout=5)
        actual = process.stdout.strip()
        passed = actual == expected
        if not passed:
            all_passed = False
            
        results.append({
            'test': i + 1,
            'input': input_val,
            'expected': expected,
            'actual': actual,
            'passed': passed
        })
    except Exception as e:
        all_passed = False
        results.append({'test': i + 1, 'error': str(e), 'passed': False})

print(json.dumps({'success': True, 'output': 'Compiled and executed', 'test_results': results, 'all_passed': all_passed}))
"
    else:
        COMPILE_ERR=$(cat /tmp/compile_err.log)
        echo "{\"success\": false, \"error\": \"Compilation failed\", \"details\": $(python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))" <<< "$COMPILE_ERR")}"
    fi
else
    echo '{"success": false, "error": "Unsupported language"}'
fi