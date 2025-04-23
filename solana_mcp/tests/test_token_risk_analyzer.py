"""Tests for TokenRiskAnalyzer class."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer

# Test constants
TEST_TOKEN_MINT = "ENxauXrtBtnFH1aJAFYZnxVVM4rGLkXJi3bUhT7tpump"  # SOLMCP
TEST_TOKEN_NAME = "SOLMCP"
TEST_TOKEN_SYMBOL = "SOLMCP"

# Print helper function
def print_result(name, data):
    """Print test result data in a readable format."""
    print(f"\n{'=' * 50}")
    print(f"=== TEST PASSED: {name} ===")
    print(f"{'=' * 50}")
    
    # Print some key information based on the test type
    if name == "Token Risk Analysis":
        # Basic token info
        print(f"Token: {data.get('name', 'Unknown')} ({data.get('symbol', 'UNKNOWN')})")
        print(f"Risk Level: {data.get('risk_level', 'Unknown')}")
        print(f"Risk Score: {data.get('overall_risk_score', 0)}")
        
        # Print risk breakdown
        print("\nRisk Breakdown:")
        print(f"- Supply Risk: {data.get('supply_risk_score', 0)}")
        print(f"- Authority Risk: {data.get('authority_risk_score', 0)}")
        print(f"- Liquidity Risk: {data.get('liquidity_risk_score', 0)}")
        print(f"- Ownership Risk: {data.get('ownership_risk_score', 0)}")
        
        # Print flags
        if data.get('flags'):
            print("\nRisk Flags:")
            for flag in data.get('flags', []):
                print(f"- {flag}")
    
    elif name == "Liquidity Analysis":
        print(f"Total Liquidity: ${data.get('total_liquidity_usd', 0):,.2f}")
        print(f"Has Locked Liquidity: {data.get('has_locked_liquidity', False)}")
        if data.get('largest_pool'):
            pool = data.get('largest_pool')
            print(f"Largest Pool: {pool.get('protocol', 'Unknown')} - {pool.get('pair', 'Unknown')}")
        print(f"Liquidity to Market Cap Ratio: {data.get('liquidity_to_mcap_ratio', 0):.4f}")
    
    elif name == "Holder Analysis":
        print(f"Total Holders: {data.get('total_holders', 0)}")
        print(f"Top Holder %: {data.get('top_holder_percentage', 0)}%")
        print(f"Top 10 Holders %: {data.get('top_10_percentage', 0)}%")
        print(f"Concentration Index: {data.get('concentration_index', 0)}")
    
    elif name == "Authority Risks Analysis":
        print(f"Has Mint Authority: {data.get('has_mint_authority', False)}")
        print(f"Has Freeze Authority: {data.get('has_freeze_authority', False)}")
        if data.get('mint_authority'):
            print(f"Mint Authority: {data.get('mint_authority')}")
        if data.get('freeze_authority'):
            print(f"Freeze Authority: {data.get('freeze_authority')}")
        print(f"Risk Score: {data.get('risk_score', 0)}")
        
        if data.get('flags'):
            print("\nAuthority Risk Flags:")
            for flag in data.get('flags', []):
                print(f"- {flag}")
    
    # Always print the full JSON at the end
    print(f"\nFull Response:\n{json.dumps(data, indent=2)}")
    print(f"{'=' * 50}")

class MockSolanaClient:
    """Mock Solana client for testing."""
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def get_token_metadata(self, mint):
        """Get token metadata."""
        return {
            "metadata": {
                "name": TEST_TOKEN_NAME,
                "symbol": TEST_TOKEN_SYMBOL
            },
            "last_updated": "2023-01-01T00:00:00Z"
        }
        
    async def get_market_price(self, mint):
        """Get token market price."""
        return {
            "price_data": {
                "price_usd": 1.23,
                "liquidity": {
                    "total_usd": 500000
                }
            }
        }
        
    async def get_token_supply(self, mint):
        """Get token supply."""
        return {
            "value": {
                "uiAmountString": "1000000000",
                "decimals": 9
            }
        }
        
    async def get_token_largest_accounts(self, mint):
        """Get token largest accounts."""
        return {
            "value": [
                {"address": "addr1", "amount": "100000"},
                {"address": "addr2", "amount": "50000"}
            ]
        }
        
    async def get_account_info(self, address):
        """Get account info."""
        return {
            "executable": False,
            "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "lamports": 10000000,
            "data": {
                "parsed": {
                    "info": {
                        "mint": TEST_TOKEN_MINT,
                        "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                        "tokenAmount": {"amount": "1000000000", "decimals": 9, "uiAmount": 1000.0}
                    },
                    "type": "account"
                },
                "program": "spl-token",
                "space": 165
            }
        }
    
    async def get_signatures_for_address(self, address, limit=None, before=None):
        """Get signatures for address."""
        return [{"signature": "test_signature"}]
    
    async def get_transaction(self, signature):
        """Get transaction details."""
        return {
            "transaction": {
                "message": {
                    "accountKeys": ["TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"]
                }
            },
            "blockTime": 1640995200,  # Jan 1, 2022
            "slot": 100000000
        }
    
    async def get_program_accounts(self, program_id, filters=None, limit=None):
        """Get program accounts."""
        return []

@pytest.mark.asyncio
async def test_analyze_token_risks():
    """Test the analyze_token_risks function."""
    # Create mock client
    mock_client = MockSolanaClient()
    
    # Create analyzer
    analyzer = TokenRiskAnalyzer(mock_client)
    
    # Test the function
    result = await analyzer.analyze_token_risks(TEST_TOKEN_MINT)
    
    # Verify result
    assert "name" in result
    assert "symbol" in result
    assert "risk_level" in result
    assert "overall_risk_score" in result
    
    # Print result
    print_result("Token Risk Analysis", result)

@pytest.mark.asyncio
async def test_categorize_token():
    """Test the _categorize_token function."""
    # Create mock client
    mock_client = MockSolanaClient()
    
    # Create analyzer
    analyzer = TokenRiskAnalyzer(mock_client)
    
    # Test some token names/symbols
    animal_category = analyzer._categorize_token("Dog Token", "DOG")
    assert animal_category == "Animal"
    
    food_category = analyzer._categorize_token("Pizza Token", "PIZZA")
    assert food_category == "Food"
    
    meme_category = analyzer._categorize_token("Moon Soon", "MOON")
    assert meme_category == "Meme"
    
    # Print result
    print_result("Token Categorization", {
        "animal_token": {"name": "Dog Token", "symbol": "DOG", "category": animal_category},
        "food_token": {"name": "Pizza Token", "symbol": "PIZZA", "category": food_category},
        "meme_token": {"name": "Moon Soon", "symbol": "MOON", "category": meme_category}
    })

@pytest.mark.asyncio
async def test_analyze_liquidity_risks():
    """Test the _analyze_liquidity_risks function."""
    # Create mock client
    mock_client = MockSolanaClient()
    
    # Create analyzer
    analyzer = TokenRiskAnalyzer(mock_client)
    
    # Test the function
    result = await analyzer._analyze_liquidity_risks(TEST_TOKEN_MINT)
    
    # Verify result
    assert "total_liquidity_usd" in result
    
    # Print result
    print_result("Liquidity Analysis", result)

@pytest.mark.asyncio
async def test_analyze_holder_distribution():
    """Test the _analyze_holder_distribution function."""
    # Create mock client
    mock_client = MockSolanaClient()
    
    # Create analyzer
    analyzer = TokenRiskAnalyzer(mock_client)
    
    # Test the function
    result = await analyzer._analyze_holder_distribution(TEST_TOKEN_MINT)
    
    # Verify result
    assert "total_holders" in result
    
    # Print result
    print_result("Holder Analysis", result)

@pytest.mark.asyncio
async def test_analyze_authority_risks():
    """Test the _analyze_authority_risks function."""
    # Create mock client
    mock_client = MockSolanaClient()
    
    # Create analyzer
    analyzer = TokenRiskAnalyzer(mock_client)
    
    # Mock account info
    mint_info = {
        "data": ["AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", "base64"]
    }
    
    # Test the function
    result = await analyzer._analyze_authority_risks(TEST_TOKEN_MINT, mint_info)
    
    # Verify result
    assert "has_mint_authority" in result
    
    # Print result
    print_result("Authority Risks Analysis", result) 