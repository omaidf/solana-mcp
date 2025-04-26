#!/usr/bin/env python
"""
Test runner for Solana MCP Server tests
"""
import os
import sys
import pytest
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_runner")

class JSONReportPlugin:
    """Pytest plugin to generate JSON output for each test"""
    
    def __init__(self):
        self.reports = []
        
    def pytest_runtest_logreport(self, report):
        """Collect test reports"""
        if report.when == "call" or (report.when == "setup" and report.outcome == "failed"):
            test_data = {
                "name": report.nodeid,
                "outcome": report.outcome,
                "duration": getattr(report, "duration", 0),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add any captured output
            if hasattr(report, "capstdout") and report.capstdout:
                test_data["stdout"] = report.capstdout
            if hasattr(report, "capstderr") and report.capstderr:
                test_data["stderr"] = report.capstderr
                
            # Add any failure information
            if report.outcome == "failed":
                test_data["error"] = str(report.longrepr)
                
            self.reports.append(test_data)
    
    def pytest_sessionfinish(self):
        """Print the JSON report at the end of the session"""
        summary = {
            "total": len(self.reports),
            "passed": sum(1 for r in self.reports if r["outcome"] == "passed"),
            "failed": sum(1 for r in self.reports if r["outcome"] == "failed"),
            "skipped": sum(1 for r in self.reports if r["outcome"] == "skipped"),
            "tests": self.reports
        }
        
        print("\n\n=== JSON TEST REPORT ===")
        print(json.dumps(summary, indent=2))
        print("=======================\n")

def run_tests(test_path=None, verbose=False, json_report=True):
    """Run the test suite with the specified options"""
    logger.info(f"Running tests with Python {sys.version}")
    
    # Determine test path
    if not test_path:
        # Run all tests by default
        test_path = os.path.dirname(os.path.abspath(__file__))
    
    # Set up arguments
    args = [test_path]
    if verbose:
        args.append("-v")
    
    # Add asyncio marker
    args.extend(["-m", "asyncio"])
    
    # Add coverage reporting
    args.extend(["--cov=core", "--cov=api", "--cov-report=term"])
    
    # Add JSON reporting plugin
    json_plugin = JSONReportPlugin()
    plugins = [json_plugin] if json_report else []
    
    # Run tests
    logger.info(f"Running tests in: {test_path}")
    result = pytest.main(args, plugins=plugins)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Solana MCP Server tests")
    parser.add_argument(
        "--path", 
        help="Path to specific test or test directory"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-json", 
        action="store_true", 
        help="Disable JSON report output"
    )
    
    args = parser.parse_args()
    sys.exit(run_tests(args.path, args.verbose, not args.no_json)) 