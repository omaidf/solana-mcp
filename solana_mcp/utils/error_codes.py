"""Solana error code explanations."""

# Dictionary mapping Solana error types to human-readable explanations
SOLANA_ERROR_EXPLANATIONS = {
    # Account errors
    "AccountNotFound": "The specified account could not be found on the Solana blockchain.",
    "InvalidAccountOwner": "The account is owned by a different program than expected.",
    "InvalidAccountData": "The account data is not in the expected format.",
    "AccountInUse": "The account is currently in use by another transaction.",
    
    # Transaction errors
    "BlockhashNotFound": "The specified blockhash is too old or invalid. Transactions need a recent blockhash.",
    "InsufficientFundsForFee": "The account doesn't have enough SOL to pay the transaction fee.",
    "InvalidSignature": "One or more transaction signatures are invalid. Check the signing keypair.",
    "DuplicateSignature": "The same signature was submitted multiple times.",
    "TransactionError": "The transaction could not be processed. Check the specific error details.",
    
    # Token errors
    "TokenAccountNotFound": "The specified token account doesn't exist.",
    "InsufficientFunds": "The token account doesn't have enough tokens for this operation.",
    "TokenAccountOwnerMismatch": "The specified owner doesn't match the token account's owner.",
    "MintMismatch": "The specified mint doesn't match the token account's mint.",
    "TokenAccountFrozen": "The token account is frozen and cannot be modified.",
    
    # RPC errors
    "JsonRpcError": "Error in the JSON-RPC request or response format.",
    "NodeBehind": "The Solana node is behind the current blockchain state.",
    "RateLimited": "The RPC endpoint is rate limiting your requests. Slow down or switch endpoints.",
    "ServerError": "The Solana RPC server encountered an internal error.",
    
    # Common validation errors
    "Invalid public key": "The provided public key is not valid. Solana addresses should be base58 encoded and typically 32-44 characters long.",
    "Invalid signature": "The provided transaction signature is not valid. Solana signatures should be base58 encoded.",
    "Invalid program id": "The provided program ID is not valid. Program IDs should be base58 encoded Solana addresses.",
    
    # Parse errors
    "ParseError": "Failed to parse the blockchain data. The data format may be unexpected or corrupted.",
    "InvalidInstruction": "The transaction contains an invalid instruction that cannot be processed.",
    "InvalidParameter": "One or more parameters provided to the Solana program are invalid.",
    
    # Program errors
    "ProgramError": "A Solana program encountered an error during execution.",
    "UpgradeError": "An error occurred during program upgrade.",
    "InstructionError": "An error occurred during instruction execution.",
    
    # Specific program errors
    "OwnerMismatch": "The account owner doesn't match the expected owner.",
    "InvalidAmount": "The token amount is invalid for this operation.",
    "InvalidMint": "The token mint address is invalid or doesn't exist.",
    "InvalidAuthority": "The authority provided doesn't have permission for this operation.",
    
    # NFT-specific errors
    "MetadataError": "An error occurred while interacting with token metadata.",
    "MasterEditionError": "An error occurred while interacting with a Master Edition.",
    "EditionError": "An error occurred while interacting with an Edition.",
    "MetaplexError": "An error occurred in the Metaplex protocol.",
    
    # General errors
    "SlotSkipped": "The requested slot was skipped and information is not available.",
    "NotConfirmed": "The transaction is not yet confirmed.",
    "NotFinalized": "The transaction is not yet finalized.",
    "Timeout": "The RPC request timed out. The network might be congested."
} 