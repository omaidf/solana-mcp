"""Whale detector service for identifying large token holders."""

from solana_mcp.services.whale_detector.detector import detect_whale_wallets

__all__ = ["detect_whale_wallets"] 