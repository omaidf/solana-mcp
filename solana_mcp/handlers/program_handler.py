"""
Program handler for consolidating program-specific logic.

This module provides specialized handlers for different Solana programs,
centralizing program-specific logic such as instruction parsing, account
validation, and data interpretation.
"""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from solana.publickey import PublicKey
from solana.transaction import Transaction

from solana_mcp.models.instruction_data import InstructionData
from solana_mcp.models.parsed_transaction import ParsedTransaction
from solana_mcp.models.token_info import TokenInfo
from solana_mcp.models.transaction_metadata import TransactionMetadata
from solana_mcp.utils.base58 import bs58_decode
from solana_mcp.utils.error_handling import InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)


class ProgramType(Enum):
    """Enum of supported Solana program types."""
    SYSTEM = "system"
    TOKEN = "token"
    TOKEN_2022 = "token-2022"
    ASSOCIATED_TOKEN = "associated-token"
    STAKE = "stake"
    VOTE = "vote"
    MEMO = "memo"
    COMPUTE_BUDGET = "compute-budget"
    METADATA = "metadata"
    NAME_SERVICE = "name-service"
    CUSTOM = "custom"


@dataclass
class ProgramInfo:
    """Information about a Solana program."""
    program_type: ProgramType
    program_id: str
    name: str


# Common program IDs
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"
NAME_SERVICE_PROGRAM_ID = "namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX"

# Known programs
KNOWN_PROGRAMS = [
    ProgramInfo(ProgramType.SYSTEM, SYSTEM_PROGRAM_ID, "System Program"),
    ProgramInfo(ProgramType.TOKEN, TOKEN_PROGRAM_ID, "Token Program"),
    ProgramInfo(ProgramType.TOKEN_2022, TOKEN_2022_PROGRAM_ID, "Token-2022 Program"),
    ProgramInfo(ProgramType.ASSOCIATED_TOKEN, ASSOCIATED_TOKEN_PROGRAM_ID, "Associated Token Program"),
    ProgramInfo(ProgramType.METADATA, METADATA_PROGRAM_ID, "Metadata Program"),
    ProgramInfo(ProgramType.MEMO, MEMO_PROGRAM_ID, "Memo Program"),
    ProgramInfo(ProgramType.COMPUTE_BUDGET, COMPUTE_BUDGET_PROGRAM_ID, "Compute Budget Program"),
    ProgramInfo(ProgramType.NAME_SERVICE, NAME_SERVICE_PROGRAM_ID, "Name Service Program"),
]

# Map of program IDs to program info
PROGRAM_ID_MAP = {p.program_id: p for p in KNOWN_PROGRAMS}


class ProgramHandler(ABC):
    """Base class for program-specific handlers."""
    
    @abstractmethod
    def get_program_type(self) -> ProgramType:
        """Get the type of program this handler is for."""
        pass
    
    @abstractmethod
    def get_program_ids(self) -> List[str]:
        """Get the program IDs this handler can process."""
        pass
    
    @abstractmethod
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse an instruction for this program."""
        pass
    
    def can_handle(self, program_id: str) -> bool:
        """Check if this handler can handle the given program ID."""
        return program_id in self.get_program_ids()
    
    def extract_transfers(self, instruction: Dict[str, Any], accounts: List[str]) -> List[Dict[str, Any]]:
        """Extract transfers from an instruction.
        
        Args:
            instruction: The instruction data
            accounts: List of account public keys
            
        Returns:
            List of transfer details
        """
        return []


class SystemProgramHandler(ProgramHandler):
    """Handler for System Program instructions."""
    
    def get_program_type(self) -> ProgramType:
        return ProgramType.SYSTEM
    
    def get_program_ids(self) -> List[str]:
        return [SYSTEM_PROGRAM_ID]
    
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse a System Program instruction."""
        instruction_type = instruction.get("parsed", {}).get("type", "unknown")
        
        # Extract instruction data based on type
        data: Dict[str, Any] = {}
        parsed_data = instruction.get("parsed", {}).get("info", {})
        
        if instruction_type == "transfer":
            data = {
                "source": parsed_data.get("source", ""),
                "destination": parsed_data.get("destination", ""),
                "lamports": parsed_data.get("lamports", 0),
            }
        elif instruction_type == "createAccount":
            data = {
                "source": parsed_data.get("source", ""),
                "newAccount": parsed_data.get("newAccount", ""),
                "lamports": parsed_data.get("lamports", 0),
                "space": parsed_data.get("space", 0),
                "owner": parsed_data.get("owner", ""),
            }
        elif instruction_type in ["allocate", "assign", "createAccountWithSeed"]:
            data = parsed_data
            
        return InstructionData(
            program_id=SYSTEM_PROGRAM_ID,
            program_type=ProgramType.SYSTEM.value,
            instruction_type=instruction_type,
            data=data
        )
    
    def extract_transfers(self, instruction: Dict[str, Any], accounts: List[str]) -> List[Dict[str, Any]]:
        """Extract SOL transfers from a System Program instruction."""
        transfers = []
        
        parsed = instruction.get("parsed", {})
        if parsed.get("type") == "transfer":
            info = parsed.get("info", {})
            transfers.append({
                "type": "sol",
                "source": info.get("source"),
                "destination": info.get("destination"),
                "amount": info.get("lamports", 0),
                "decimals": 9,
            })
            
        return transfers


class TokenProgramHandler(ProgramHandler):
    """Handler for Token Program instructions."""
    
    def get_program_type(self) -> ProgramType:
        return ProgramType.TOKEN
    
    def get_program_ids(self) -> List[str]:
        return [TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID]
    
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse a Token Program instruction."""
        program_id = instruction.get("programId", TOKEN_PROGRAM_ID)
        instruction_type = instruction.get("parsed", {}).get("type", "unknown")
        
        # Extract instruction data based on type
        data: Dict[str, Any] = {}
        parsed_data = instruction.get("parsed", {}).get("info", {})
        
        if instruction_type in ["transfer", "transferChecked"]:
            data = {
                "source": parsed_data.get("source", ""),
                "destination": parsed_data.get("destination", ""),
                "amount": parsed_data.get("amount", "0"),
                "mint": parsed_data.get("mint", ""),
                "decimals": parsed_data.get("decimals", 0),
            }
        elif instruction_type in ["mintTo", "mintToChecked"]:
            data = {
                "mint": parsed_data.get("mint", ""),
                "destination": parsed_data.get("destination", ""),
                "amount": parsed_data.get("amount", "0"),
                "decimals": parsed_data.get("decimals", 0),
            }
        elif instruction_type in ["burn", "burnChecked"]:
            data = {
                "source": parsed_data.get("source", ""),
                "mint": parsed_data.get("mint", ""),
                "amount": parsed_data.get("amount", "0"),
                "decimals": parsed_data.get("decimals", 0),
            }
        elif instruction_type == "closeAccount":
            data = {
                "account": parsed_data.get("account", ""),
                "destination": parsed_data.get("destination", ""),
            }
        else:
            data = parsed_data
            
        return InstructionData(
            program_id=program_id,
            program_type=self._get_program_type_value(program_id),
            instruction_type=instruction_type,
            data=data
        )
    
    def _get_program_type_value(self, program_id: str) -> str:
        """Get the program type value based on program ID."""
        if program_id == TOKEN_PROGRAM_ID:
            return ProgramType.TOKEN.value
        elif program_id == TOKEN_2022_PROGRAM_ID:
            return ProgramType.TOKEN_2022.value
        else:
            return ProgramType.TOKEN.value
    
    def extract_transfers(self, instruction: Dict[str, Any], accounts: List[str]) -> List[Dict[str, Any]]:
        """Extract token transfers from a Token Program instruction."""
        transfers = []
        
        parsed = instruction.get("parsed", {})
        instruction_type = parsed.get("type", "")
        
        if instruction_type in ["transfer", "transferChecked"]:
            info = parsed.get("info", {})
            # For token transfers, ensure we have the necessary fields
            if all(k in info for k in ["source", "destination", "amount"]):
                transfers.append({
                    "type": "token",
                    "source": info.get("source"),
                    "destination": info.get("destination"),
                    "amount": int(info.get("amount", "0")),
                    "mint": info.get("mint", ""),
                    "decimals": info.get("decimals", 0),
                })
                
        return transfers


class AssociatedTokenProgramHandler(ProgramHandler):
    """Handler for Associated Token Program instructions."""
    
    def get_program_type(self) -> ProgramType:
        return ProgramType.ASSOCIATED_TOKEN
    
    def get_program_ids(self) -> List[str]:
        return [ASSOCIATED_TOKEN_PROGRAM_ID]
    
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse an Associated Token Program instruction."""
        instruction_type = instruction.get("parsed", {}).get("type", "unknown")
        
        # Extract instruction data based on type
        data: Dict[str, Any] = {}
        parsed_data = instruction.get("parsed", {}).get("info", {})
        
        if instruction_type == "create":
            data = {
                "payer": parsed_data.get("payer", ""),
                "wallet": parsed_data.get("wallet", ""),
                "mint": parsed_data.get("mint", ""),
                "associatedAccount": parsed_data.get("associatedAccount", ""),
            }
            
        return InstructionData(
            program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
            program_type=ProgramType.ASSOCIATED_TOKEN.value,
            instruction_type=instruction_type,
            data=data
        )


class ComputeBudgetProgramHandler(ProgramHandler):
    """Handler for Compute Budget Program instructions."""
    
    def get_program_type(self) -> ProgramType:
        return ProgramType.COMPUTE_BUDGET
    
    def get_program_ids(self) -> List[str]:
        return [COMPUTE_BUDGET_PROGRAM_ID]
    
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse a Compute Budget Program instruction."""
        instruction_type = instruction.get("parsed", {}).get("type", "unknown")
        
        # Extract instruction data based on type
        data: Dict[str, Any] = {}
        parsed_data = instruction.get("parsed", {}).get("info", {})
        
        if instruction_type == "setComputeUnitLimit":
            data = {
                "units": parsed_data.get("units", 0),
            }
        elif instruction_type == "setComputeUnitPrice":
            data = {
                "microLamports": parsed_data.get("microLamports", 0),
            }
            
        return InstructionData(
            program_id=COMPUTE_BUDGET_PROGRAM_ID,
            program_type=ProgramType.COMPUTE_BUDGET.value,
            instruction_type=instruction_type,
            data=data
        )


class ProgramHandlerRegistry:
    """Registry for program handlers."""
    
    def __init__(self):
        """Initialize the registry with default handlers."""
        self._handlers: Dict[str, ProgramHandler] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register the default handlers."""
        self.register_handler(SystemProgramHandler())
        self.register_handler(TokenProgramHandler())
        self.register_handler(AssociatedTokenProgramHandler())
        self.register_handler(ComputeBudgetProgramHandler())
    
    def register_handler(self, handler: ProgramHandler) -> None:
        """Register a program handler.
        
        Args:
            handler: The program handler to register
        """
        for program_id in handler.get_program_ids():
            self._handlers[program_id] = handler
    
    def get_handler(self, program_id: str) -> Optional[ProgramHandler]:
        """Get a handler for the specified program ID.
        
        Args:
            program_id: The program ID to get a handler for
            
        Returns:
            The handler for the program, or None if no handler is registered
        """
        return self._handlers.get(program_id)
    
    def parse_instruction(self, instruction: Dict[str, Any]) -> InstructionData:
        """Parse an instruction using the appropriate handler.
        
        Args:
            instruction: The instruction to parse
            
        Returns:
            The parsed instruction data
            
        Raises:
            NotFoundError: If no handler is registered for the program
        """
        program_id = instruction.get("programId")
        if not program_id:
            raise InvalidInputError("Instruction is missing programId")
        
        handler = self.get_handler(program_id)
        if handler:
            return handler.parse_instruction(instruction)
        
        # For unknown programs, return a generic instruction data
        program_type = PROGRAM_ID_MAP.get(program_id, ProgramInfo(ProgramType.CUSTOM, program_id, "Unknown")).program_type.value
        
        return InstructionData(
            program_id=program_id,
            program_type=program_type,
            instruction_type="unknown",
            data=instruction.get("data", {})
        )
    
    def extract_transfers(self, instruction: Dict[str, Any], accounts: List[str]) -> List[Dict[str, Any]]:
        """Extract transfers from an instruction using the appropriate handler.
        
        Args:
            instruction: The instruction to extract transfers from
            accounts: List of account public keys
            
        Returns:
            List of transfer details
        """
        program_id = instruction.get("programId")
        if not program_id:
            return []
        
        handler = self.get_handler(program_id)
        if handler:
            return handler.extract_transfers(instruction, accounts)
        
        return []


# Singleton instance of the program handler registry
_program_handler_registry: Optional[ProgramHandlerRegistry] = None


def get_program_handler_registry() -> ProgramHandlerRegistry:
    """Get the global program handler registry instance.
    
    Returns:
        The program handler registry
    """
    global _program_handler_registry
    if _program_handler_registry is None:
        _program_handler_registry = ProgramHandlerRegistry()
    return _program_handler_registry


def get_program_info(program_id: str) -> ProgramInfo:
    """Get information about a program.
    
    Args:
        program_id: The program ID to get information for
        
    Returns:
        Program information
    """
    if program_id in PROGRAM_ID_MAP:
        return PROGRAM_ID_MAP[program_id]
    
    # For unknown programs, return a custom program info
    return ProgramInfo(
        program_type=ProgramType.CUSTOM,
        program_id=program_id,
        name=f"Unknown Program ({program_id[:8]}...)"
    )


def parse_instruction(instruction: Dict[str, Any]) -> InstructionData:
    """Parse an instruction using the program handler registry.
    
    Args:
        instruction: The instruction to parse
        
    Returns:
        The parsed instruction data
    """
    registry = get_program_handler_registry()
    return registry.parse_instruction(instruction)


def extract_transfers(instruction: Dict[str, Any], accounts: List[str]) -> List[Dict[str, Any]]:
    """Extract transfers from an instruction.
    
    Args:
        instruction: The instruction to extract transfers from
        accounts: List of account public keys
        
    Returns:
        List of transfer details
    """
    registry = get_program_handler_registry()
    return registry.extract_transfers(instruction, accounts) 