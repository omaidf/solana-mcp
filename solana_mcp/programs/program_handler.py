"""
Program handlers for Solana MCP.

This module provides a base class and registry for program-specific handlers
that can process program instructions and transactions.
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Set, Type, ClassVar

from solana_mcp.utils.dependency_injection import ServiceProvider

logger = logging.getLogger(__name__)


class ProgramHandler(abc.ABC):
    """Base class for program-specific handlers."""
    
    # Program ID(s) handled by this handler
    PROGRAM_IDS: ClassVar[Set[str]] = set()
    
    @classmethod
    def handles_program(cls, program_id: str) -> bool:
        """
        Check if this handler handles the given program ID.
        
        Args:
            program_id: The program ID to check
            
        Returns:
            True if this handler handles the program, False otherwise
        """
        return program_id in cls.PROGRAM_IDS
    
    @abc.abstractmethod
    async def parse_instruction(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a program instruction into a more readable format.
        
        Args:
            instruction: The instruction data
            
        Returns:
            Parsed instruction data
        """
        pass
    
    @abc.abstractmethod
    async def get_instruction_type(self, instruction: Dict[str, Any]) -> str:
        """
        Get the type of an instruction.
        
        Args:
            instruction: The instruction data
            
        Returns:
            Instruction type as a string
        """
        pass
    
    async def extract_token_transfers(
        self, instruction: Dict[str, Any], meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract token transfers from an instruction.
        
        Args:
            instruction: The instruction data
            meta: Transaction meta data
            
        Returns:
            List of token transfers
        """
        return []
    
    async def extract_sol_transfers(
        self, instruction: Dict[str, Any], meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract SOL transfers from an instruction.
        
        Args:
            instruction: The instruction data
            meta: Transaction meta data
            
        Returns:
            List of SOL transfers
        """
        return []


class ProgramHandlerRegistry:
    """Registry for program handlers."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance of ProgramHandlerRegistry exists."""
        if cls._instance is None:
            cls._instance = super(ProgramHandlerRegistry, cls).__new__(cls)
            cls._instance._handlers = {}
            cls._instance._program_id_to_handler = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ProgramHandlerRegistry':
        """
        Get the singleton instance of ProgramHandlerRegistry.
        
        Returns:
            ProgramHandlerRegistry instance
        """
        if cls._instance is None:
            return cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry."""
        cls._instance = None
    
    def register_handler(self, handler_class: Type[ProgramHandler]) -> None:
        """
        Register a program handler.
        
        Args:
            handler_class: The handler class to register
        """
        if handler_class.__name__ in self._handlers:
            logger.warning(f"Handler {handler_class.__name__} already registered. Overwriting.")
        
        self._handlers[handler_class.__name__] = handler_class
        
        # Create a mapping from program IDs to handlers
        for program_id in handler_class.PROGRAM_IDS:
            if program_id in self._program_id_to_handler:
                logger.warning(
                    f"Program ID {program_id} already handled by "
                    f"{self._program_id_to_handler[program_id].__name__}. Overwriting."
                )
            self._program_id_to_handler[program_id] = handler_class
    
    def get_handler_for_program(self, program_id: str) -> Optional[ProgramHandler]:
        """
        Get a handler instance for the given program ID.
        
        Args:
            program_id: The program ID to get a handler for
            
        Returns:
            A handler instance or None if no handler is registered
        """
        handler_class = self._program_id_to_handler.get(program_id)
        if not handler_class:
            return None
        
        # Use dependency injection to create the handler
        service_provider = ServiceProvider.get_instance()
        try:
            return service_provider.get(handler_class.__name__)
        except KeyError:
            # If not registered, create a new instance and register it
            handler = handler_class()
            service_provider.register(handler_class.__name__, handler)
            return handler
    
    def get_registered_program_ids(self) -> Set[str]:
        """
        Get all registered program IDs.
        
        Returns:
            Set of registered program IDs
        """
        return set(self._program_id_to_handler.keys())


# Example handler for System Program
class SystemProgramHandler(ProgramHandler):
    """Handler for the System Program."""
    
    PROGRAM_IDS = {"11111111111111111111111111111111"}
    
    INSTRUCTION_TYPES = {
        0: "Create Account",
        1: "Assign",
        2: "Transfer",
        3: "Create Account With Seed",
        4: "Advance Nonce Account",
        5: "Withdraw Nonce Account",
        6: "Initialize Nonce Account",
        7: "Authorize Nonce Account",
        8: "Allocate",
        9: "Allocate With Seed",
        10: "Assign With Seed",
        11: "Transfer With Seed",
        12: "Upgrade Nonce Account",
    }
    
    async def parse_instruction(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a system program instruction."""
        parsed = {
            "program": "System Program",
            "program_id": instruction.get("program_id"),
            "type": await self.get_instruction_type(instruction),
            "data": instruction.get("data", {}),
            "accounts": instruction.get("accounts", []),
        }
        
        # Parse specific instruction types
        instruction_type = parsed.get("type")
        
        if instruction_type == "Transfer":
            # Parse transfer-specific data
            if "parsed" in instruction.get("data", {}):
                parsed["from"] = instruction["accounts"][0]
                parsed["to"] = instruction["accounts"][1]
                parsed["amount"] = instruction["data"]["parsed"]["info"].get("lamports", 0)
        
        return parsed
    
    async def get_instruction_type(self, instruction: Dict[str, Any]) -> str:
        """Get the type of a system program instruction."""
        if "parsed" in instruction.get("data", {}):
            # Already parsed by the RPC
            return instruction["data"]["parsed"]["type"]
        
        # Try to parse from raw data
        data = instruction.get("data", "")
        if isinstance(data, str) and data:
            # Convert hex data to instruction type
            try:
                instruction_type_code = int(data[:2], 16)
                return self.INSTRUCTION_TYPES.get(instruction_type_code, f"Unknown ({instruction_type_code})")
            except (ValueError, IndexError):
                pass
        
        return "Unknown"
    
    async def extract_sol_transfers(
        self, instruction: Dict[str, Any], meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract SOL transfers from a system program instruction."""
        transfers = []
        instruction_type = await self.get_instruction_type(instruction)
        
        if instruction_type == "Transfer":
            # Get from and to accounts
            from_account = instruction["accounts"][0]
            to_account = instruction["accounts"][1]
            
            # Get amount from parsed data or raw data
            amount = 0
            if "parsed" in instruction.get("data", {}):
                amount = instruction["data"]["parsed"]["info"].get("lamports", 0)
            
            transfers.append({
                "from": from_account,
                "to": to_account,
                "amount": amount,
                "type": "transfer",
                "token": "SOL",
            })
        
        return transfers


# Register built-in handlers
registry = ProgramHandlerRegistry.get_instance()
registry.register_handler(SystemProgramHandler) 