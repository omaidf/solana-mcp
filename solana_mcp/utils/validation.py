"""Validation utilities for Solana MCP.

This module provides utilities for validating Solana-specific data.
"""

import re
from typing import Any, Union, Dict, List, Optional

# Solana public key validation pattern (base58 format)
PUBKEY_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

class InvalidPublicKeyError(Exception):
    """Exception raised when an invalid public key is provided."""
    def __init__(self, pubkey: str):
        super().__init__(f"Invalid public key: {pubkey}")
        self.pubkey = pubkey


def validate_public_key(pubkey: str) -> bool:
    """Validate a Solana public key.
    
    Args:
        pubkey: The public key to validate
        
    Returns:
        True if the public key is valid, False otherwise
    """
    if not pubkey or not isinstance(pubkey, str):
        return False
    return bool(PUBKEY_PATTERN.match(pubkey))


def validate_solana_address(address: str, field_name: str = "address") -> None:
    """Validate a Solana address and raise an exception if invalid.
    
    Args:
        address: The address to validate
        field_name: Name of the field for the error message
        
    Raises:
        ValueError: If the address is invalid
    """
    if not validate_public_key(address):
        raise ValueError(f"Invalid Solana {field_name}: {address}")


def validate_transaction_signature(signature: str) -> bool:
    """Validate a Solana transaction signature.
    
    Args:
        signature: The transaction signature to validate
        
    Returns:
        True if the signature is valid, False otherwise
    """
    if not signature or not isinstance(signature, str):
        return False
    
    # Transaction signatures are also base58 encoded but longer than public keys
    return bool(re.match(r"^[1-9A-HJ-NP-Za-km-z]{43,128}$", signature)) 