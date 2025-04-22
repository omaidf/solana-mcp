"""WebSocket support for real-time Solana updates."""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import websockets
from websockets.exceptions import ConnectionClosed

from solana_mcp.config import SolanaConfig


class SubscriptionType(str, Enum):
    """Types of WebSocket subscriptions."""
    
    ACCOUNT = "account"
    PROGRAM = "program"
    SIGNATURE = "signature"
    SLOT = "slot"
    ROOT = "root"
    LOGS = "logs"


class SolanaWebSocketClient:
    """Client for Solana WebSocket API."""
    
    def __init__(
        self, 
        config: SolanaConfig,
        on_notification_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """Initialize the WebSocket client.
        
        Args:
            config: Solana configuration
            on_notification_callback: Optional callback for notifications
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = config.rpc_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = ws_url
        self.config = config
        self.on_notification_callback = on_notification_callback
        
        # Tracking subscriptions and requests
        self.subscriptions: Dict[int, Dict[str, Any]] = {}
        self.request_id = 0
        self.ws_connection = None
        self.running = False
        self.task = None
        
        # Response handling
        self.response_futures: Dict[int, asyncio.Future] = {}
    
    def _get_next_id(self) -> int:
        """Get the next request ID.
        
        Returns:
            The next request ID
        """
        self.request_id += 1
        return self.request_id
    
    async def connect(self):
        """Connect to the WebSocket server."""
        if self.ws_connection is not None:
            return
            
        self.ws_connection = await websockets.connect(
            self.ws_url, 
            max_size=None,  # No limit on message size
            ping_interval=20,
            ping_timeout=20
        )
        
        self.running = True
        self.task = asyncio.create_task(self._listen())
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.running = False
        
        # Cancel all subscriptions
        for sub_id in list(self.subscriptions.keys()):
            await self.unsubscribe(sub_id)
            
        # Close the WebSocket connection
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
            
        # Cancel the listen task
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
    
    async def _listen(self):
        """Listen for WebSocket messages."""
        try:
            while self.running and self.ws_connection:
                try:
                    msg = await self.ws_connection.recv()
                    
                    # Parse the message
                    data = json.loads(msg)
                    
                    # Handle subscription notifications
                    if "method" in data and data["method"] == "subscription":
                        sub_id = data["params"]["subscription"]
                        subscription = self.subscriptions.get(sub_id)
                        
                        if subscription and self.on_notification_callback:
                            # Call the notification callback
                            self.on_notification_callback(
                                subscription["type"],
                                data["params"]["result"]
                            )
                    # Handle regular responses
                    elif "id" in data:
                        request_id = data["id"]
                        future = self.response_futures.pop(request_id, None)
                        if future and not future.done():
                            if "error" in data:
                                future.set_exception(
                                    ValueError(f"WebSocket error: {data['error']}")
                                )
                            else:
                                future.set_result(data.get("result"))
                                
                except ConnectionClosed:
                    if self.running:
                        # Try to reconnect
                        self.ws_connection = None
                        await asyncio.sleep(1)
                        await self.connect()
                except Exception as e:
                    logging.error(f"WebSocket error: {str(e)}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Normal task cancellation
            pass
        except Exception as e:
            logging.error(f"WebSocket task error: {str(e)}")
    
    async def _send_request(self, method: str, params: List[Any]) -> Any:
        """Send a JSON-RPC request over WebSocket.
        
        Args:
            method: The RPC method
            params: The parameters
            
        Returns:
            The response result
        """
        if not self.ws_connection:
            await self.connect()
            
        request_id = self._get_next_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Create a future for the response
        future = asyncio.get_event_loop().create_future()
        self.response_futures[request_id] = future
        
        # Send the request
        await self.ws_connection.send(json.dumps(request))
        
        # Wait for the response with timeout
        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self.response_futures.pop(request_id, None)
            raise ValueError("WebSocket request timed out")
    
    async def subscribe_account(
        self, 
        pubkey: str, 
        commitment: Optional[str] = None,
        encoding: str = "jsonParsed"
    ) -> int:
        """Subscribe to account updates.
        
        Args:
            pubkey: The account public key
            commitment: The commitment level
            encoding: The encoding for account data
            
        Returns:
            The subscription ID
        """
        params = [
            pubkey,
            {
                "encoding": encoding,
                "commitment": commitment or self.config.commitment
            }
        ]
        
        sub_id = await self._send_request("accountSubscribe", params)
        self.subscriptions[sub_id] = {
            "type": SubscriptionType.ACCOUNT,
            "pubkey": pubkey,
            "params": params
        }
        return sub_id
    
    async def subscribe_program(
        self, 
        program_id: str, 
        commitment: Optional[str] = None,
        encoding: str = "jsonParsed",
        filters: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Subscribe to program account updates.
        
        Args:
            program_id: The program ID
            commitment: The commitment level
            encoding: The encoding for account data
            filters: Optional filters
            
        Returns:
            The subscription ID
        """
        config = {
            "encoding": encoding,
            "commitment": commitment or self.config.commitment
        }
        
        if filters:
            config["filters"] = filters
            
        params = [program_id, config]
        
        sub_id = await self._send_request("programSubscribe", params)
        self.subscriptions[sub_id] = {
            "type": SubscriptionType.PROGRAM,
            "program_id": program_id,
            "params": params
        }
        return sub_id
    
    async def subscribe_signature(
        self, 
        signature: str, 
        commitment: Optional[str] = None
    ) -> int:
        """Subscribe to signature status updates.
        
        Args:
            signature: The transaction signature
            commitment: The commitment level
            
        Returns:
            The subscription ID
        """
        params = [
            signature,
            {
                "commitment": commitment or self.config.commitment
            }
        ]
        
        sub_id = await self._send_request("signatureSubscribe", params)
        self.subscriptions[sub_id] = {
            "type": SubscriptionType.SIGNATURE,
            "signature": signature,
            "params": params
        }
        return sub_id
    
    async def subscribe_slot(self) -> int:
        """Subscribe to slot updates.
        
        Returns:
            The subscription ID
        """
        sub_id = await self._send_request("slotSubscribe", [])
        self.subscriptions[sub_id] = {
            "type": SubscriptionType.SLOT,
            "params": []
        }
        return sub_id
    
    async def subscribe_logs(
        self, 
        filter_value: Union[str, Dict[str, List[str]]],
        commitment: Optional[str] = None
    ) -> int:
        """Subscribe to transaction logs.
        
        Args:
            filter_value: "all", "allWithVotes", or {"mentions": [<address>]}
            commitment: The commitment level
            
        Returns:
            The subscription ID
        """
        params = [
            filter_value,
            {
                "commitment": commitment or self.config.commitment
            }
        ]
        
        sub_id = await self._send_request("logsSubscribe", params)
        self.subscriptions[sub_id] = {
            "type": SubscriptionType.LOGS,
            "filter": filter_value,
            "params": params
        }
        return sub_id
    
    async def unsubscribe(self, subscription_id: int) -> bool:
        """Unsubscribe from updates.
        
        Args:
            subscription_id: The subscription ID
            
        Returns:
            Whether the unsubscribe was successful
        """
        if subscription_id not in self.subscriptions:
            return False
            
        sub_type = self.subscriptions[subscription_id]["type"]
        method = f"{sub_type}Unsubscribe"
        
        result = await self._send_request(method, [subscription_id])
        if result:
            self.subscriptions.pop(subscription_id, None)
            
        return result


class NotificationHub:
    """Hub for managing WebSocket notifications and callbacks."""
    
    def __init__(self, ws_client: SolanaWebSocketClient):
        """Initialize the notification hub.
        
        Args:
            ws_client: The WebSocket client
        """
        self.ws_client = ws_client
        self.callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.account_subscriptions: Dict[str, int] = {}
        self.program_subscriptions: Dict[str, int] = {}
        self.signature_subscriptions: Dict[str, int] = {}
        
        # Set the notification callback
        self.ws_client.on_notification_callback = self._handle_notification
    
    def _handle_notification(self, subscription_type: str, result: Dict[str, Any]):
        """Handle a WebSocket notification.
        
        Args:
            subscription_type: The subscription type
            result: The notification data
        """
        # Determine the notification key based on subscription type
        key = subscription_type
        
        if subscription_type == SubscriptionType.ACCOUNT:
            pubkey = result.get("pubkey", "unknown")
            key = f"{subscription_type}:{pubkey}"
        elif subscription_type == SubscriptionType.PROGRAM:
            program_id = result.get("pubkey", "unknown")
            key = f"{subscription_type}:{program_id}"
        elif subscription_type == SubscriptionType.SIGNATURE:
            signature = result.get("signature", "unknown")
            key = f"{subscription_type}:{signature}"
        
        # Call the registered callbacks
        for callback in self.callbacks.get(key, []):
            try:
                callback(result)
            except Exception as e:
                logging.error(f"Error in notification callback: {str(e)}")
    
    async def subscribe_account(
        self, 
        pubkey: str, 
        callback: Callable[[Dict[str, Any]], None],
        encoding: str = "jsonParsed"
    ):
        """Subscribe to account updates.
        
        Args:
            pubkey: The account public key
            callback: The callback function
            encoding: The encoding for account data
        """
        # Add the callback
        key = f"{SubscriptionType.ACCOUNT}:{pubkey}"
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
        
        # Subscribe if not already subscribed
        if pubkey not in self.account_subscriptions:
            sub_id = await self.ws_client.subscribe_account(pubkey, encoding=encoding)
            self.account_subscriptions[pubkey] = sub_id
    
    async def subscribe_program(
        self, 
        program_id: str, 
        callback: Callable[[Dict[str, Any]], None],
        encoding: str = "jsonParsed",
        filters: Optional[List[Dict[str, Any]]] = None
    ):
        """Subscribe to program account updates.
        
        Args:
            program_id: The program ID
            callback: The callback function
            encoding: The encoding for account data
            filters: Optional filters
        """
        # Add the callback
        key = f"{SubscriptionType.PROGRAM}:{program_id}"
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
        
        # Subscribe if not already subscribed
        if program_id not in self.program_subscriptions:
            sub_id = await self.ws_client.subscribe_program(
                program_id, 
                encoding=encoding,
                filters=filters
            )
            self.program_subscriptions[program_id] = sub_id
    
    async def subscribe_signature(
        self, 
        signature: str, 
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Subscribe to signature status updates.
        
        Args:
            signature: The transaction signature
            callback: The callback function
        """
        # Add the callback
        key = f"{SubscriptionType.SIGNATURE}:{signature}"
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
        
        # Subscribe if not already subscribed
        if signature not in self.signature_subscriptions:
            sub_id = await self.ws_client.subscribe_signature(signature)
            self.signature_subscriptions[signature] = sub_id
    
    async def subscribe_slot(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to slot updates.
        
        Args:
            callback: The callback function
        """
        # Add the callback
        key = SubscriptionType.SLOT
        if key not in self.callbacks:
            self.callbacks[key] = []
            # Subscribe only once
            await self.ws_client.subscribe_slot()
        self.callbacks[key].append(callback)
    
    async def subscribe_logs(
        self, 
        filter_value: Union[str, Dict[str, List[str]]], 
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Subscribe to transaction logs.
        
        Args:
            filter_value: "all", "allWithVotes", or {"mentions": [<address>]}
            callback: The callback function
        """
        # Add the callback
        key = f"{SubscriptionType.LOGS}:{json.dumps(filter_value)}"
        if key not in self.callbacks:
            self.callbacks[key] = []
            # Subscribe only once per filter
            await self.ws_client.subscribe_logs(filter_value)
        self.callbacks[key].append(callback)
    
    async def unsubscribe_all(self):
        """Unsubscribe from all subscriptions."""
        # Unsubscribe from account subscriptions
        for pubkey, sub_id in self.account_subscriptions.items():
            await self.ws_client.unsubscribe(sub_id)
        self.account_subscriptions.clear()
        
        # Unsubscribe from program subscriptions
        for program_id, sub_id in self.program_subscriptions.items():
            await self.ws_client.unsubscribe(sub_id)
        self.program_subscriptions.clear()
        
        # Unsubscribe from signature subscriptions
        for signature, sub_id in self.signature_subscriptions.items():
            await self.ws_client.unsubscribe(sub_id)
        self.signature_subscriptions.clear()
        
        # Clear all callbacks
        self.callbacks.clear() 