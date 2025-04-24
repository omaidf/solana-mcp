#!/usr/bin/env python3
"""
Script to identify modules that import decorators from error_handling.py
and need to be updated to use decorators.py directly.
"""

import os
import re
import ast
from typing import List, Dict, Set, Tuple

# Define the root of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define the deprecated decorators and their replacements
DEPRECATED_DECORATORS = {
    "with_error_handling": "api_error_handler",
    "with_validation": "validate_input",
    "rate_limit": "rate_limit",  # Same name but different source
    "api_error_handler": "api_error_handler",  # Same name but different source
}


def find_python_files(start_path: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories.
    
    Args:
        start_path: Directory to start searching from
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files


def check_import(content: str) -> bool:
    """Check if the content imports from error_handling.
    
    Args:
        content: Python file content
        
    Returns:
        True if importing from error_handling
    """
    # Pattern to match regular imports
    import_pattern = re.compile(r'from\s+solana_mcp\.api\.error_handling\s+import')
    # Pattern to match "import ... as ..." style imports
    import_as_pattern = re.compile(r'import\s+solana_mcp\.api\.error_handling')
    
    return (
        import_pattern.search(content) is not None or 
        import_as_pattern.search(content) is not None
    )


def extract_imports(content: str) -> Set[str]:
    """Extract imported decorator names from content.
    
    Args:
        content: Python file content
        
    Returns:
        Set of imported decorator names
    """
    imported_decorators = set()
    
    # Pattern to match imports from error_handling
    import_pattern = re.compile(
        r'from\s+solana_mcp\.api\.error_handling\s+import\s+([^#\n]+)'
    )
    
    matches = import_pattern.findall(content)
    if matches:
        for match in matches:
            # Split by comma and strip whitespace
            for item in match.split(','):
                name = item.strip()
                # Handle "as" renaming
                if ' as ' in name:
                    name = name.split(' as ')[0].strip()
                if name in DEPRECATED_DECORATORS:
                    imported_decorators.add(name)
    
    return imported_decorators


def analyze_file(file_path: str) -> Tuple[bool, Set[str]]:
    """Analyze a Python file for error_handling imports.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        Tuple of (has_import, imported_decorators)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if check_import(content):
        imported_decorators = extract_imports(content)
        return True, imported_decorators
    
    return False, set()


def main():
    """Main function to find and analyze Python files."""
    # Find all Python files
    python_files = find_python_files(PROJECT_ROOT)
    
    # Files that need changes
    files_to_update = []
    
    # Analyze each file
    for file_path in python_files:
        # Skip the error_handling.py file itself and the current script
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        if (rel_path == "solana_mcp/api/error_handling.py" or 
            os.path.basename(file_path) == os.path.basename(__file__)):
            continue
        
        has_import, imported_decorators = analyze_file(file_path)
        
        if has_import and imported_decorators:
            files_to_update.append((rel_path, imported_decorators))
    
    # Print results
    if files_to_update:
        print("Found files that need to be updated:")
        print("=====================================")
        
        for file_path, decorators in files_to_update:
            print(f"\n{file_path}:")
            for decorator in decorators:
                replacement = DEPRECATED_DECORATORS[decorator]
                print(f"  - Replace '{decorator}' with '{replacement}' from solana_mcp.decorators")
        
        print("\nTotal files to update:", len(files_to_update))
    else:
        print("No files found that need to be updated.")


if __name__ == "__main__":
    main() 