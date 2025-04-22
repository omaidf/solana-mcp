"""Analysis utilities for Solana blockchain data."""

import datetime
from typing import Dict, List, Any, Optional

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError


async def analyze_token_flow(
    address: str,
    solana_client: SolanaClient,
    limit: int = 50
) -> Dict[str, Any]:
    """Analyze token inflows and outflows for an address.
    
    Args:
        address: The account address to analyze
        solana_client: Solana client
        limit: Maximum number of transactions to analyze
        
    Returns:
        Token flow analysis results
    """
    try:
        # Get transaction history
        signatures = await solana_client.get_signatures_for_address(address, limit=limit)
        
        # Process transactions to find token movements
        inflows = []
        outflows = []
        sol_transfers_in = []
        sol_transfers_out = []
        
        for tx_info in signatures:
            try:
                # Get full transaction details
                tx = await solana_client.get_transaction(tx_info["signature"])
                
                # Skip failed transactions
                if tx.get("meta", {}).get("err"):
                    continue
                
                # Extract timestamp
                timestamp = tx_info.get("blockTime")
                date_str = (
                    datetime.datetime.fromtimestamp(timestamp).isoformat() 
                    if timestamp else None
                )
                
                # Analyze SOL transfers
                pre_balances = tx.get("meta", {}).get("preBalances", [])
                post_balances = tx.get("meta", {}).get("postBalances", [])
                account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
                
                if account_keys and pre_balances and post_balances:
                    # Find the index of our address
                    try:
                        account_idx = account_keys.index(address)
                        
                        # Calculate SOL difference
                        pre_bal = pre_balances[account_idx]
                        post_bal = post_balances[account_idx]
                        sol_diff = (post_bal - pre_bal) / 1_000_000_000  # Convert to SOL
                        
                        if sol_diff > 0:
                            sol_transfers_in.append({
                                "amount": sol_diff,
                                "timestamp": date_str,
                                "signature": tx_info["signature"]
                            })
                        elif sol_diff < 0:
                            sol_transfers_out.append({
                                "amount": abs(sol_diff),
                                "timestamp": date_str,
                                "signature": tx_info["signature"]
                            })
                    except ValueError:
                        # Address not found in account keys
                        pass
                
                # Analyze token transfers
                if "meta" in tx and "postTokenBalances" in tx["meta"] and "preTokenBalances" in tx["meta"]:
                    pre = tx["meta"]["preTokenBalances"]
                    post = tx["meta"]["postTokenBalances"]
                    
                    # Group token balances by account
                    pre_by_acct = {item.get("accountIndex"): item for item in pre if isinstance(item, dict)}
                    post_by_acct = {item.get("accountIndex"): item for item in post if isinstance(item, dict)}
                    
                    # Find token accounts owned by our address
                    our_pre = [item for item in pre if isinstance(item, dict) and item.get("owner") == address]
                    our_post = [item for item in post if isinstance(item, dict) and item.get("owner") == address]
                    
                    # Analyze token balance changes
                    for post_item in our_post:
                        acct_idx = post_item.get("accountIndex")
                        mint = post_item.get("mint")
                        post_amount = post_item.get("uiTokenAmount", {}).get("uiAmount", 0)
                        
                        # Check if this account existed in pre balances
                        if acct_idx in pre_by_acct:
                            pre_item = pre_by_acct[acct_idx]
                            pre_amount = pre_item.get("uiTokenAmount", {}).get("uiAmount", 0)
                            
                            # Check for balance change
                            diff = post_amount - pre_amount
                            if diff > 0:
                                # Incoming token transfer
                                inflows.append({
                                    "mint": mint,
                                    "amount": diff,
                                    "timestamp": date_str,
                                    "signature": tx_info["signature"]
                                })
                            elif diff < 0:
                                # Outgoing token transfer
                                outflows.append({
                                    "mint": mint,
                                    "amount": abs(diff),
                                    "timestamp": date_str,
                                    "signature": tx_info["signature"]
                                })
                        else:
                            # New token account - must be an inflow
                            inflows.append({
                                "mint": mint,
                                "amount": post_amount,
                                "timestamp": date_str,
                                "signature": tx_info["signature"]
                            })
                    
                    # Find token accounts that disappeared (complete outflows)
                    for pre_item in our_pre:
                        acct_idx = pre_item.get("accountIndex")
                        mint = pre_item.get("mint")
                        pre_amount = pre_item.get("uiTokenAmount", {}).get("uiAmount", 0)
                        
                        # Check if this account no longer exists in post balances
                        if acct_idx not in post_by_acct and pre_amount > 0:
                            # Account closed or emptied - must be an outflow
                            outflows.append({
                                "mint": mint,
                                "amount": pre_amount,
                                "timestamp": date_str,
                                "signature": tx_info["signature"]
                            })
            except Exception as e:
                # Skip problematic transactions
                print(f"Error processing transaction {tx_info.get('signature')}: {str(e)}")
                continue
        
        # Enrich token data with metadata where possible
        await enrich_token_flow_data(inflows, solana_client)
        await enrich_token_flow_data(outflows, solana_client)
        
        # Summarize by token
        inflow_by_token = summarize_by_token(inflows)
        outflow_by_token = summarize_by_token(outflows)
        
        return {
            "address": address,
            "analysis_type": "token_flow",
            "period": {
                "first_tx": signatures[-1]["blockTime"] if signatures else None,
                "last_tx": signatures[0]["blockTime"] if signatures else None,
            },
            "token_summary": {
                "total_inflow_count": len(inflows),
                "total_outflow_count": len(outflows),
                "unique_tokens_received": len(inflow_by_token),
                "unique_tokens_sent": len(outflow_by_token),
            },
            "sol_summary": {
                "total_sol_received": sum(item["amount"] for item in sol_transfers_in),
                "total_sol_sent": sum(item["amount"] for item in sol_transfers_out),
                "sol_transfers_in_count": len(sol_transfers_in),
                "sol_transfers_out_count": len(sol_transfers_out),
            },
            "inflows": inflows,
            "outflows": outflows,
            "sol_transfers_in": sol_transfers_in,
            "sol_transfers_out": sol_transfers_out,
            "inflow_by_token": inflow_by_token,
            "outflow_by_token": outflow_by_token,
        }
    
    except InvalidPublicKeyError as e:
        return {
            "error": str(e),
            "error_explanation": "The address provided is not a valid Solana public key."
        }
    except SolanaRpcError as e:
        return {
            "error": str(e),
            "error_explanation": "Error communicating with the Solana blockchain."
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "error_explanation": "An unexpected error occurred during analysis."
        }


async def analyze_activity_pattern(
    address: str,
    solana_client: SolanaClient,
    limit: int = 100
) -> Dict[str, Any]:
    """Analyze transaction activity patterns for an address.
    
    Args:
        address: The account address to analyze
        solana_client: Solana client
        limit: Maximum number of transactions to analyze
        
    Returns:
        Activity pattern analysis results
    """
    try:
        # Get transaction history
        signatures = await solana_client.get_signatures_for_address(address, limit=limit)
        
        # Group by day
        activity_by_day = {}
        hourly_distribution = [0] * 24
        
        for tx in signatures:
            if "blockTime" in tx:
                # Convert timestamp to datetime
                tx_time = datetime.datetime.fromtimestamp(tx["blockTime"])
                
                # Extract date and hour
                date = tx_time.strftime("%Y-%m-%d")
                hour = tx_time.hour
                
                # Update daily counts
                if date not in activity_by_day:
                    activity_by_day[date] = 0
                activity_by_day[date] += 1
                
                # Update hourly distribution
                hourly_distribution[hour] += 1
        
        # Sort by date
        sorted_activity = [{"date": k, "transactions": v} for k, v in sorted(activity_by_day.items())]
        
        # Identify most active periods
        most_active_day = max(activity_by_day.items(), key=lambda x: x[1]) if activity_by_day else (None, 0)
        most_active_hour = hourly_distribution.index(max(hourly_distribution))
        
        # Calculate activity patterns
        if sorted_activity:
            # Calculate days with activity
            active_days = len(sorted_activity)
            
            # Calculate date range
            try:
                first_date = datetime.datetime.strptime(sorted_activity[0]["date"], "%Y-%m-%d")
                last_date = datetime.datetime.strptime(sorted_activity[-1]["date"], "%Y-%m-%d")
                date_range = (last_date - first_date).days + 1
                activity_density = active_days / max(1, date_range)
            except Exception:
                date_range = 0
                activity_density = 0
        else:
            active_days = 0
            date_range = 0
            activity_density = 0
        
        # Activity segmentation
        morning = sum(hourly_distribution[6:12])  # 6 AM to 12 PM
        afternoon = sum(hourly_distribution[12:18])  # 12 PM to 6 PM
        evening = sum(hourly_distribution[18:24])  # 6 PM to 12 AM
        night = sum(hourly_distribution[0:6])  # 12 AM to 6 AM
        
        # Find activity pattern
        time_segments = [
            ("morning", morning),
            ("afternoon", afternoon),
            ("evening", evening),
            ("night", night)
        ]
        primary_activity_time = max(time_segments, key=lambda x: x[1])[0] if any(count > 0 for _, count in time_segments) else None
        
        return {
            "address": address,
            "analysis_type": "activity_pattern",
            "total_transactions": len(signatures),
            "activity_over_time": {
                "by_day": sorted_activity,
                "by_hour": [{"hour": h, "transactions": hourly_distribution[h]} for h in range(24)]
            },
            "statistics": {
                "active_days": active_days,
                "date_range_days": date_range,
                "activity_density": activity_density,
                "most_active_day": {
                    "date": most_active_day[0],
                    "transactions": most_active_day[1]
                } if most_active_day[0] else None,
                "most_active_hour": most_active_hour
            },
            "time_distribution": {
                "morning": morning,
                "afternoon": afternoon,
                "evening": evening,
                "night": night,
                "primary_activity_time": primary_activity_time
            }
        }
    
    except InvalidPublicKeyError as e:
        return {
            "error": str(e),
            "error_explanation": "The address provided is not a valid Solana public key."
        }
    except SolanaRpcError as e:
        return {
            "error": str(e),
            "error_explanation": "Error communicating with the Solana blockchain."
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "error_explanation": "An unexpected error occurred during analysis."
        }


async def analyze_token_holding_distribution(
    address: str,
    solana_client: SolanaClient
) -> Dict[str, Any]:
    """Analyze token holding distribution for an address.
    
    Args:
        address: The account address to analyze
        solana_client: Solana client
        
    Returns:
        Token holding distribution analysis results
    """
    try:
        # Get token accounts
        token_accounts = await solana_client.get_token_accounts_by_owner(address)
        
        # Categorize by value
        value_categories = {
            "high_value": [],
            "medium_value": [],
            "low_value": [],
            "dust": []
        }
        
        # Map of mints to metadata
        token_metadata = {}
        
        total_unique_tokens = 0
        nft_count = 0
        fungible_token_count = 0
        
        for account in token_accounts:
            if "account" in account and "data" in account["account"]:
                data = account["account"]["data"]
                if "parsed" in data and "info" in data["parsed"]:
                    info = data["parsed"]["info"]
                    token_amount = info.get("tokenAmount", {})
                    mint = info.get("mint")
                    
                    # Skip empty accounts
                    if token_amount.get("uiAmount", 0) == 0:
                        continue
                    
                    total_unique_tokens += 1
                    
                    # Get token metadata if we haven't already
                    if mint and mint not in token_metadata:
                        try:
                            metadata = await solana_client.get_token_metadata(mint)
                            token_metadata[mint] = metadata
                        except Exception:
                            token_metadata[mint] = {}
                    
                    # Determine if NFT
                    is_nft = token_amount.get("decimals", 0) == 0 and token_amount.get("uiAmount", 0) == 1
                    
                    if is_nft:
                        nft_count += 1
                        # Add to appropriate value category (all NFTs in high for now)
                        value_categories["high_value"].append({
                            "mint": mint,
                            "tokenAmount": token_amount,
                            "metadata": token_metadata.get(mint, {}),
                            "type": "nft"
                        })
                    else:
                        fungible_token_count += 1
                        # For fungible tokens, try to categorize by value
                        # This is just a simple example - real categorization would use price data
                        if token_amount.get("uiAmount", 0) < 0.01:
                            category = "dust"
                        elif token_amount.get("uiAmount", 0) < 1:
                            category = "low_value"
                        elif token_amount.get("uiAmount", 0) < 100:
                            category = "medium_value"
                        else:
                            category = "high_value"
                        
                        value_categories[category].append({
                            "mint": mint,
                            "tokenAmount": token_amount,
                            "metadata": token_metadata.get(mint, {}),
                            "type": "fungible"
                        })
        
        return {
            "address": address,
            "analysis_type": "token_holding_distribution",
            "summary": {
                "total_unique_tokens": total_unique_tokens,
                "nft_count": nft_count,
                "fungible_token_count": fungible_token_count,
                "high_value_count": len(value_categories["high_value"]),
                "medium_value_count": len(value_categories["medium_value"]),
                "low_value_count": len(value_categories["low_value"]),
                "dust_count": len(value_categories["dust"])
            },
            "distribution": value_categories
        }
    
    except InvalidPublicKeyError as e:
        return {
            "error": str(e),
            "error_explanation": "The address provided is not a valid Solana public key."
        }
    except SolanaRpcError as e:
        return {
            "error": str(e),
            "error_explanation": "Error communicating with the Solana blockchain."
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "error_explanation": "An unexpected error occurred during analysis."
        }


# -------------------------------------------
# Helper Functions
# -------------------------------------------

async def enrich_token_flow_data(flows: List[Dict[str, Any]], solana_client: SolanaClient) -> None:
    """Enrich token flow data with metadata information.
    
    Args:
        flows: List of token flow items
        solana_client: Solana client
    """
    # Track already fetched metadata to avoid duplicate requests
    metadata_cache = {}
    
    for item in flows:
        mint = item.get("mint")
        if not mint or mint in metadata_cache:
            continue
            
        try:
            metadata = await solana_client.get_token_metadata(mint)
            metadata_cache[mint] = metadata
        except Exception:
            metadata_cache[mint] = {}
    
    # Add metadata to each flow item
    for item in flows:
        mint = item.get("mint")
        if mint and mint in metadata_cache:
            item["token_info"] = {
                "name": metadata_cache[mint].get("name", "Unknown"),
                "symbol": metadata_cache[mint].get("symbol")
            }


def summarize_by_token(flows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Summarize flow data by token.
    
    Args:
        flows: List of token flow items
        
    Returns:
        Summary of flows by token
    """
    summary = {}
    
    for item in flows:
        mint = item.get("mint")
        if not mint:
            continue
            
        if mint not in summary:
            summary[mint] = {
                "mint": mint,
                "total_amount": 0,
                "count": 0,
                "token_info": item.get("token_info", {})
            }
            
        summary[mint]["total_amount"] += item.get("amount", 0)
        summary[mint]["count"] += 1
    
    return summary 