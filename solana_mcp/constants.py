"""Constants used throughout the Solana MCP application.

This module defines common constants to avoid duplication and ensure consistency.
"""

# Solana system program IDs
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
METAPLEX_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"  # Same as METADATA_PROGRAM_ID

# DEX program IDs
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
JUPITER_PROGRAM_ID = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"
ORCA_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"

# Common token mint addresses
SOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

# Mapping of program IDs to human-readable names
PROGRAM_NAMES = {
    SYSTEM_PROGRAM_ID: "System Program",
    TOKEN_PROGRAM_ID: "Token Program",
    ASSOCIATED_TOKEN_PROGRAM_ID: "Token Associated Program",
    METADATA_PROGRAM_ID: "Metaplex Metadata",
    JUPITER_PROGRAM_ID: "Jupiter Aggregator",
    ORCA_PROGRAM_ID: "Orca Program",
    RAYDIUM_PROGRAM_ID: "Raydium Program",
    SOL_MINT: "Wrapped SOL",
    USDC_MINT: "USDC Mint",
    USDT_MINT: "USDT Mint",
} 