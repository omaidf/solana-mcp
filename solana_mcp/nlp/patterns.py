"""Pattern definitions for natural language query parsing."""

from typing import Dict, List, Callable, Any, Optional

# Define common transaction types and their keywords for semantic search
TRANSACTION_CATEGORIES = {
    "token_transfer": ["transfer", "send", "receive", "spl-token", "token program"],
    "nft_mint": ["mint", "nft", "metaplex", "metadata", "master edition"],
    "nft_sale": ["sale", "marketplace", "auction", "bid", "offer", "purchase"],
    "swap": ["swap", "exchange", "trade", "jupiter", "orca", "raydium"],
    "stake": ["stake", "delegate", "staking", "validator", "withdraw stake"],
    "system_transfer": ["system program", "sol transfer", "lamports"],
    "vote": ["vote", "voting", "governance"],
    "program_deploy": ["bpf loader", "deploy", "upgrade", "program"],
    "failed": ["failed", "error", "rejected"]
}

# Basic patterns for NL query understanding
QUERY_PATTERNS = [
    # Balance queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:sol|solana)? ?balance (?:of|for) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_balance",
        "params": lambda match: {"address": match.group(1)}
    },
    # Account info queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_account_info",
        "params": lambda match: {"address": match.group(1)}
    },
    # Token queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:token|tokens|token info|token details) (?:of|for|owned by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_accounts",
        "params": lambda match: {"owner": match.group(1)}
    },
    # Token info
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) token (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Transaction history
    {
        "pattern": r"(?:what are|get|show|check|find) (?:the )?(?:transactions|tx|transaction history) (?:of|for|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})(?: with limit (\d+))?",
        "intent": "get_transactions",
        "params": lambda match: {"address": match.group(1), "limit": int(match.group(2)) if match.group(2) else 20}
    },
    # NFT queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:nft|nft info|nft details) (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_nft_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Whale queries - Find whales (large holders) for a token
    {
        "pattern": r"(?:are there|are there any|do you see|can you find|any) (?:whales|whale|large holder|big investor|big wallet) (?:in|for|holding) (?:this token|this|token|mint)? ?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_whales",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Fresh wallet queries - Find new/fresh wallets for a token
    {
        "pattern": r"(?:are there|are there any|do you see|can you find|any) (?:fresh|new|recent|suspicious) (?:wallets|wallet|holder|holders|account|accounts) (?:in|for|holding) (?:this token|this|token|mint)? ?([a-zA-Z0-9]{32,44})",
        "intent": "get_fresh_wallets",
        "params": lambda match: {"mint": match.group(1)}
    },
]

# Transaction search patterns
TRANSACTION_SEARCH_PATTERNS = [
    r"(?:find|search for|show me|get) (?:transactions|tx) (?:for|from|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44}) (?:with|that are|that have|related to|about) (.*)",
    r"(?:find|search for|show me|get) (.*) (?:transactions|tx) (?:for|from|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})"
] 