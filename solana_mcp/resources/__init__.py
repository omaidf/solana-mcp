"""MCP resources for the Solana MCP server."""

# Export resources
from solana_mcp.resources.account import get_account, get_balance
from solana_mcp.resources.token import get_token_accounts, get_token_info, get_token_holders
from solana_mcp.resources.transaction import get_transaction_details, get_address_transactions
from solana_mcp.resources.program import get_program_info, get_program_account_list
from solana_mcp.resources.network import get_network_epoch, get_network_validators
from solana_mcp.resources.nft import get_nft_info 