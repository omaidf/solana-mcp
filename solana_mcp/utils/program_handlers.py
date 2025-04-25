"""Program-specific logic handlers for Solana MCP.

This module provides interfaces and implementations for handling program-specific logic,
such as instruction type detection and parsing.
"""

import abc
from base64 import b64decode
import logging
from typing import Any, Dict, List, Optional, Type, ClassVar, Mapping

from solana.rpc.api import Pubkey
from solana.transaction import TransactionInstruction

from solana_mcp.utils.error_handling import handle_errors, NotFoundError

logger = logging.getLogger(__name__)

class ProgramHandler(abc.ABC):
    """Base abstract class for program-specific handlers."""
    
    # Class variable to store program ID
    PROGRAM_ID: ClassVar[Pubkey]
    
    @classmethod
    def supports_program(cls, program_id: Pubkey) -> bool:
        """Check if this handler supports the given program ID.
        
        Args:
            program_id: The program ID to check
            
        Returns:
            True if supported, False otherwise
        """
        return program_id == cls.PROGRAM_ID
    
    @abc.abstractmethod
    def parse_instruction(self, instruction: TransactionInstruction) -> Dict[str, Any]:
        """Parse an instruction for this program.
        
        Args:
            instruction: The instruction to parse
            
        Returns:
            Parsed instruction data
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        pass
    
    @abc.abstractmethod
    def get_instruction_type(self, instruction: TransactionInstruction) -> str:
        """Get the type of an instruction for this program.
        
        Args:
            instruction: The instruction to identify
            
        Returns:
            String identifying the instruction type
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        pass


class TokenProgramHandler(ProgramHandler):
    """Handler for Token Program instructions."""
    
    PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
    
    # Instruction type discriminators
    _INSTRUCTION_TYPES = {
        0: "initializeMint",
        1: "initializeAccount",
        2: "initializeMultisig",
        3: "transfer",
        4: "approve",
        5: "revoke",
        6: "setAuthority",
        7: "mintTo",
        8: "burn",
        9: "closeAccount",
        10: "freezeAccount",
        11: "thawAccount",
        12: "transferChecked",
        13: "approveChecked",
        14: "mintToChecked",
        15: "burnChecked",
        16: "initializeAccount2",
        17: "syncNative",
        18: "initializeAccount3",
        19: "initializeMultisig2",
        20: "initializeMint2",
    }
    
    @handle_errors(reraise=True)
    def get_instruction_type(self, instruction: TransactionInstruction) -> str:
        """Get the type of a Token Program instruction.
        
        Args:
            instruction: The instruction to identify
            
        Returns:
            String identifying the instruction type
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        if len(instruction.data) == 0:
            raise NotFoundError("Instruction data is empty")
        
        instruction_type = instruction.data[0]
        if instruction_type not in self._INSTRUCTION_TYPES:
            raise NotFoundError(f"Unknown Token Program instruction type: {instruction_type}")
        
        return self._INSTRUCTION_TYPES[instruction_type]
    
    @handle_errors(reraise=True)
    def parse_instruction(self, instruction: TransactionInstruction) -> Dict[str, Any]:
        """Parse a Token Program instruction.
        
        Args:
            instruction: The instruction to parse
            
        Returns:
            Parsed instruction data
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        instruction_type = self.get_instruction_type(instruction)
        
        # Basic parsing result
        result = {
            "type": instruction_type,
            "program": "Token Program",
            "programId": str(self.PROGRAM_ID),
            "accounts": [str(account) for account in instruction.keys],
        }
        
        # Add more detailed parsing based on instruction type
        if instruction_type == "transfer":
            # Transfer instruction has source, destination, owner
            if len(instruction.keys) >= 3:
                result["source"] = str(instruction.keys[0].pubkey)
                result["destination"] = str(instruction.keys[1].pubkey)
                result["owner"] = str(instruction.keys[2].pubkey)
                
                # Extract amount from instruction data
                if len(instruction.data) >= 9:  # 1 byte for instruction type + 8 bytes for amount
                    amount = int.from_bytes(instruction.data[1:9], byteorder="little")
                    result["amount"] = amount
        
        elif instruction_type == "transferChecked":
            # TransferChecked has source, mint, destination, owner
            if len(instruction.keys) >= 4:
                result["source"] = str(instruction.keys[0].pubkey)
                result["mint"] = str(instruction.keys[1].pubkey)
                result["destination"] = str(instruction.keys[2].pubkey)
                result["owner"] = str(instruction.keys[3].pubkey)
                
                # Extract amount and decimals from instruction data
                if len(instruction.data) >= 10:  # 1 byte for instruction type + 8 bytes for amount + 1 byte for decimals
                    amount = int.from_bytes(instruction.data[1:9], byteorder="little")
                    decimals = instruction.data[9]
                    result["amount"] = amount
                    result["decimals"] = decimals
        
        # Add more instruction type parsing as needed...
        
        return result


class SystemProgramHandler(ProgramHandler):
    """Handler for System Program instructions."""
    
    PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
    
    # Instruction type discriminators
    _INSTRUCTION_TYPES = {
        0: "createAccount",
        1: "assign",
        2: "transfer",
        3: "createAccountWithSeed",
        4: "advanceNonceAccount",
        5: "withdrawNonceAccount",
        6: "initializeNonceAccount",
        7: "authorizeNonceAccount",
        8: "allocate",
        9: "allocateWithSeed",
        10: "assignWithSeed",
        11: "transferWithSeed",
        12: "upgradeNonceAccount",
    }
    
    @handle_errors(reraise=True)
    def get_instruction_type(self, instruction: TransactionInstruction) -> str:
        """Get the type of a System Program instruction.
        
        Args:
            instruction: The instruction to identify
            
        Returns:
            String identifying the instruction type
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        if len(instruction.data) < 4:
            raise NotFoundError("Instruction data too short")
        
        instruction_type = int.from_bytes(instruction.data[0:4], byteorder="little")
        if instruction_type not in self._INSTRUCTION_TYPES:
            raise NotFoundError(f"Unknown System Program instruction type: {instruction_type}")
        
        return self._INSTRUCTION_TYPES[instruction_type]
    
    @handle_errors(reraise=True)
    def parse_instruction(self, instruction: TransactionInstruction) -> Dict[str, Any]:
        """Parse a System Program instruction.
        
        Args:
            instruction: The instruction to parse
            
        Returns:
            Parsed instruction data
            
        Raises:
            NotFoundError: If the instruction type is not recognized
        """
        instruction_type = self.get_instruction_type(instruction)
        
        # Basic parsing result
        result = {
            "type": instruction_type,
            "program": "System Program",
            "programId": str(self.PROGRAM_ID),
            "accounts": [str(account) for account in instruction.keys],
        }
        
        # Add more detailed parsing based on instruction type
        if instruction_type == "transfer":
            # Transfer instruction has from and to accounts
            if len(instruction.keys) >= 2:
                result["source"] = str(instruction.keys[0].pubkey)
                result["destination"] = str(instruction.keys[1].pubkey)
                
                # Extract lamports from instruction data
                if len(instruction.data) >= 12:  # 4 bytes for instruction type + 8 bytes for lamports
                    lamports = int.from_bytes(instruction.data[4:12], byteorder="little")
                    result["lamports"] = lamports
        
        elif instruction_type == "createAccount":
            # CreateAccount has from and to accounts
            if len(instruction.keys) >= 2:
                result["from"] = str(instruction.keys[0].pubkey)
                result["newAccount"] = str(instruction.keys[1].pubkey)
                
                # Extract lamports and space from instruction data
                if len(instruction.data) >= 20:  # 4 bytes for instruction type + 8 bytes for lamports + 8 bytes for space
                    lamports = int.from_bytes(instruction.data[4:12], byteorder="little")
                    space = int.from_bytes(instruction.data[12:20], byteorder="little")
                    result["lamports"] = lamports
                    result["space"] = space
                    
                    # Extract owner if present
                    if len(instruction.data) >= 52:  # + 32 bytes for owner
                        owner_bytes = instruction.data[20:52]
                        owner = str(Pubkey.from_string(base58.encode(bytes(owner_bytes))))
                        result["owner"] = owner
        
        # Add more instruction type parsing as needed...
        
        return result


class ProgramHandlerRegistry:
    """Registry for program handlers."""
    
    def __init__(self):
        """Initialize the registry."""
        self._handlers: Dict[str, ProgramHandler] = {}
        
        # Register default handlers
        self.register_handler(TokenProgramHandler())
        self.register_handler(SystemProgramHandler())
    
    def register_handler(self, handler: ProgramHandler) -> None:
        """Register a program handler.
        
        Args:
            handler: The handler to register
        """
        self._handlers[str(handler.PROGRAM_ID)] = handler
    
    def get_handler(self, program_id: Pubkey) -> Optional[ProgramHandler]:
        """Get a handler for the given program ID.
        
        Args:
            program_id: The program ID to get a handler for
            
        Returns:
            Handler for the program or None if not found
        """
        return self._handlers.get(str(program_id))
    
    @handle_errors(reraise=True)
    def parse_instruction(self, instruction: TransactionInstruction) -> Dict[str, Any]:
        """Parse an instruction using the appropriate handler.
        
        Args:
            instruction: The instruction to parse
            
        Returns:
            Parsed instruction data
            
        Raises:
            NotFoundError: If no handler is found or instruction type is not recognized
        """
        handler = self.get_handler(instruction.program_id)
        if handler is None:
            # Basic fallback parsing for unknown programs
            return {
                "type": "unknown",
                "program": "Unknown Program",
                "programId": str(instruction.program_id),
                "accounts": [str(account.pubkey) for account in instruction.keys],
                "data": b64decode(instruction.data).hex() if instruction.data else None
            }
        
        return handler.parse_instruction(instruction)
    
    @handle_errors(reraise=True)
    def get_instruction_type(self, instruction: TransactionInstruction) -> str:
        """Get the type of an instruction using the appropriate handler.
        
        Args:
            instruction: The instruction to identify
            
        Returns:
            String identifying the instruction type
            
        Raises:
            NotFoundError: If no handler is found or instruction type is not recognized
        """
        handler = self.get_handler(instruction.program_id)
        if handler is None:
            return "unknown"
        
        return handler.get_instruction_type(instruction)


# Create a global registry instance
program_registry = ProgramHandlerRegistry() 