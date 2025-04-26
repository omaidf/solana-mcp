#!/usr/bin/env python
"""
Extract and format JSON output from test runs
"""
import sys
import json
import re
import subprocess
from typing import Dict, List, Any, Optional

def run_tests(verbose: bool = True, pattern: Optional[str] = None) -> str:
    """Run the tests and return the output"""
    cmd = ["python", "-m", "test.run_tests"]
    if verbose:
        cmd.append("--verbose")
    if pattern:
        cmd.append("--path")
        cmd.append(pattern)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def extract_json_output(output: str) -> List[Dict[str, Any]]:
    """Extract JSON output from test run"""
    results = []
    for line in output.splitlines():
        if "TEST_JSON_OUTPUT:" in line:
            # Extract JSON string - everything after TEST_JSON_OUTPUT:
            json_str = line.split("TEST_JSON_OUTPUT:", 1)[1].strip()
            try:
                data = json.loads(json_str)
                results.append(data)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {json_str}")
    return results

def format_test_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format test results for display"""
    formatted = {
        "tests": {},
        "summary": {
            "total": len(results),
            "by_type": {}
        }
    }
    
    for result in results:
        test_name = result.get("test", "unknown")
        formatted["tests"][test_name] = result.get("result", {})
        
        # Count by test type
        if ":" in test_name:
            test_type = test_name.split(":", 1)[0]
            if test_type not in formatted["summary"]["by_type"]:
                formatted["summary"]["by_type"][test_type] = 0
            formatted["summary"]["by_type"][test_type] += 1
    
    return formatted

def main():
    """Main function"""
    print("Running tests and extracting JSON output...")
    if len(sys.argv) > 1:
        # Use the provided test pattern
        output = run_tests(pattern=sys.argv[1])
    else:
        # Run all tests
        output = run_tests()
    
    results = extract_json_output(output)
    formatted = format_test_results(results)
    
    # Output formatted JSON
    print(json.dumps(formatted, indent=2))
    
    print(f"\nTotal tests with JSON output: {formatted['summary']['total']}")
    print("Test types:")
    for test_type, count in formatted["summary"]["by_type"].items():
        print(f"  {test_type}: {count} tests")

if __name__ == "__main__":
    main() 