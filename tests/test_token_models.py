"""
Tests for the token models.
"""

import pytest
from solana_mcp.models.token import TokenInfo, TokenMetadata, TokenPrice

def test_token_info_model():
    """Test the TokenInfo model."""
    token_metadata = TokenMetadata(
        name="Test Token",
        symbol="TEST",
        logo="https://example.com/logo.png",
        description="A test token",
        is_nft=False
    )
    
    token_price = TokenPrice(
        price_usd=1.0,
        price_sol=0.01,
        price_change_24h=5.0,
        volume_24h=1000000,
        market_cap=10000000
    )
    
    token_info = TokenInfo(
        address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        name="Test Token",
        symbol="TEST",
        decimals=6,
        total_supply="1000000000000",
        metadata=token_metadata,
        price=token_price,
        is_nft=False
    )
    
    assert token_info.address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert token_info.name == "Test Token"
    assert token_info.symbol == "TEST"
    assert token_info.decimals == 6
    assert token_info.total_supply == "1000000000000"
    assert token_info.is_nft == False
    
    # Test nested models
    assert token_info.metadata is not None
    assert token_info.metadata.name == "Test Token"
    assert token_info.metadata.symbol == "TEST"
    
    assert token_info.price is not None
    assert token_info.price.price_usd == 1.0
    assert token_info.price.price_sol == 0.01


def test_token_price_formatting():
    """Test the formatting methods of TokenPrice."""
    # Test with different price ranges
    high_price = TokenPrice(price_usd=12345.67)
    assert high_price.formatted_price_usd == "$12,345.67"
    
    medium_price = TokenPrice(price_usd=10.5)
    assert medium_price.formatted_price_usd == "$10.50"
    
    low_price = TokenPrice(price_usd=0.05)
    assert low_price.formatted_price_usd == "$0.0500"
    
    very_low_price = TokenPrice(price_usd=0.00001234)
    assert very_low_price.formatted_price_usd == "$0.00001234"
    
    # Test price change formatting
    positive_change = TokenPrice(price_change_24h=5.25)
    assert positive_change.formatted_change == "+5.25%"
    
    negative_change = TokenPrice(price_change_24h=-3.75)
    assert negative_change.formatted_change == "-3.75%" 