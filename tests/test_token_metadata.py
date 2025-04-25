from solana_mcp.solana_client import SolanaClient
from solana_mcp.config import SolanaConfig
import asyncio
import json

async def test_token_metadata():
    # Use the provided Helius RPC URL
    rpc_url = "https://mainnet.helius-rpc.com/?api-key=4ffc1228-f093-4974-ad2d-3edd8e5f7c03"
    client = SolanaClient(SolanaConfig(rpc_url=rpc_url))
    
    # Test with USDC token
    usdc_address = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
    print(f"Testing USDC token metadata: {usdc_address}")
    usdc_metadata = await client.get_token_metadata(usdc_address)
    print(f"USDC Metadata: {json.dumps(usdc_metadata, indent=2)}")
    
    # Test with SAMO token
    samo_address = '7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU'
    print(f"\nTesting SAMO token metadata: {samo_address}")
    samo_metadata = await client.get_token_metadata(samo_address)
    print(f"SAMO Metadata: {json.dumps(samo_metadata, indent=2)}")
    
    # Test with Bonk token
    bonk_address = 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263'
    print(f"\nTesting BONK token metadata: {bonk_address}")
    bonk_metadata = await client.get_token_metadata(bonk_address)
    print(f"BONK Metadata: {json.dumps(bonk_metadata, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_token_metadata()) 