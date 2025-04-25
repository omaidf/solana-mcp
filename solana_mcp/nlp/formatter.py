"""Response formatting utilities for natural language processing."""

import json
from typing import Any, Dict, List, Optional

def format_response(data: Any, format_level: str = "standard") -> Dict[str, Any]:
    """Format a response based on the requested detail level.
    
    Args:
        data: The data to format
        format_level: The format level (minimal, standard, detailed, auto)
        
    Returns:
        Formatted response
    """
    # Handle auto format level
    if format_level == "auto":
        # Simple heuristic - if data is large, use minimal
        if isinstance(data, dict) and len(json.dumps(data)) > 1000:
            format_level = "minimal"
        else:
            format_level = "standard"
    
    # Handle different format levels
    if format_level == "minimal":
        return create_minimal_format(data)
    elif format_level == "detailed":
        return create_detailed_format(data)
    else:  # standard
        return data


def create_minimal_format(data: Any) -> Dict[str, Any]:
    """Create a minimal format of the data.
    
    Args:
        data: The data to format
        
    Returns:
        Minimally formatted data
    """
    if not isinstance(data, dict):
        return data
    
    # Extract key information based on data type
    if "lamports" in data and "sol" in data:
        # Balance data
        return {
            "sol": data.get("sol"),
            "formatted": data.get("formatted")
        }
    elif "mint" in data and "supply" in data:
        # Token data
        return {
            "mint": data.get("mint"),
            "name": data.get("metadata", {}).get("name"),
            "symbol": data.get("metadata", {}).get("symbol"),
            "supply": data.get("supply", {}).get("uiAmount")
        }
    elif "transactions" in data and isinstance(data["transactions"], list):
        # Transaction history
        return {
            "address": data.get("address"),
            "transaction_count": len(data.get("transactions", [])),
            "recent_transactions": [tx.get("signature") for tx in data.get("transactions", [])[:5]]
        }
    
    # Default minimal extraction for any data
    return {k: v for k, v in data.items() if k in ["address", "signature", "error"]}


def create_detailed_format(data: Any) -> Dict[str, Any]:
    """Create a detailed format of the data with additional information.
    
    Args:
        data: The data to format
        
    Returns:
        Detailed formatted data with explanations
    """
    if not isinstance(data, dict):
        return {"data": data, "explanation": "Simple value returned"}
    
    # Add explanations based on data type
    if "lamports" in data and "sol" in data:
        # Balance data
        data["explanation"] = "This shows the account balance in both lamports (smallest unit) and SOL."
        data["context"] = {
            "sol_usd_conversion": "Approximate USD value would require current market data."
        }
        return data
    elif "mint" in data and "supply" in data:
        # Token data
        data["explanation"] = "This shows details about a Solana token, including its supply and metadata."
        return data
    elif "transactions" in data and isinstance(data["transactions"], list):
        # Transaction history
        data["explanation"] = f"Transaction history for address {data.get('address')}."
        if len(data.get("transactions", [])) > 0:
            data["recent_transaction_explanation"] = explain_transaction(data["transactions"][0])
        return data
    
    # Default detailed format
    return {
        "data": data,
        "explanation": "Detailed data structure returned from the Solana blockchain."
    }


def explain_transaction(tx_data: Dict[str, Any]) -> str:
    """Create a natural language explanation of a transaction.
    
    Args:
        tx_data: Transaction data
        
    Returns:
        Human-readable explanation
    """
    # Extract key information
    slot = tx_data.get("slot", "unknown")
    confirmations = tx_data.get("confirmations", "unknown")
    signature = tx_data.get("signature", "unknown")
    
    # Create basic explanation
    explanation = f"Transaction {signature[:8]}... occurred at slot {slot} with {confirmations} confirmations."
    
    # Add status
    if tx_data.get("confirmationStatus") == "finalized":
        explanation += " It is finalized on the blockchain."
    elif tx_data.get("err"):
        explanation += " It failed with an error."
    
    return explanation 