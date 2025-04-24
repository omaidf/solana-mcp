"""Semantic analysis for text-based token queries."""

import re
import string
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from collections import defaultdict

from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.token_analyzer import TokenAnalyzer
from solana_mcp.solana_client import SolanaClient, validate_public_key

# Set up logging
logger = get_logger(__name__)

# Intent categories with expanded keywords for better detection
INTENTS = {
    "general_info": [
        "tell me about", "information about", "details on", "analyze", "what is", 
        "show me", "overview of", "stats for", "details for", "basic info",
        "info on", "information on", "tell me more about", "describe", "summary of",
        "explain", "brief on", "data about", "facts about", "profile of", "report on",
        "details regarding", "general details", "token info", "token details", "general analysis"
    ],
    "price": [
        "price of", "how much is", "value of", "worth", "costs", "trading at", 
        "current price", "token price", "market price", "price", "price in usd",
        "how much does it cost", "what's the price", "what is the price", "how much for",
        "dollar value", "usd value", "value in usd", "cost in dollars", "going for",
        "selling for", "market value", "trading value", "exchange rate", "conversion rate",
        "priced at", "current valuation", "valued at", "current rate", "market rate"
    ],
    "holders": [
        "who holds", "holder", "ownership", "distribution", "who owns", 
        "largest holders", "top holders", "holder distribution", "token distribution",
        "token holders", "account holders", "who are the holders", "holder list",
        "ownership spread", "distribution pattern", "ownership distribution",
        "who is holding", "who's holding", "accounts holding", "wallets with",
        "token owners", "owner list", "owner distribution", "holder structure",
        "distribution of holders", "token allocation", "allocation of tokens"
    ],
    "whales": [
        "whale", "big holders", "large holders", "major holders", "whale analysis",
        "whales holding", "biggest holders", ">50k", "large accounts", "rich list",
        "big investors", "major investors", "big players", "big wallets", "deep pockets",
        "highest holders", "wealthiest holders", "heaviest investors", "major stakes",
        "big stakes", "large stakes", "whale wallets", "whale accounts", "whale investors",
        "large investments", "significant holdings", "substantial holdings", "top money",
        "heavy bags", "rich holders", "big bags", "big money", "largest stake"
    ],
    "fresh_wallets": [
        "fresh wallet", "new wallet", "new account", "recent buyer", "fresh account",
        "only holding", "just bought", "first purchase", "only this token", "suspicious wallet",
        "new holder", "recent holder", "virgin wallet", "new investors", "first timers",
        "first time holders", "newly created", "recently created", "just created",
        "fresh buyers", "new owners", "recent adopters", "just joined", "newcomers",
        "rookies", "fresh addresses", "virgin accounts", "one token wallet", "exclusive holders",
        "single token holder", "only token held", "suspicious activity", "suspicious pattern"
    ],
    "supply": [
        "supply", "total supply", "circulation", "circulating supply", "token supply",
        "how many tokens", "total tokens", "max supply", "token count", "circulation amount",
        "total circulation", "issued tokens", "token issuance", "supply cap", "maximum supply",
        "token quantity", "amount in circulation", "total amount", "emission", "token emission",
        "token volume", "coins in circulation", "available tokens", "minted tokens", "token limit",
        "supply limit", "market supply", "supply distribution", "supply schedule", "token metrics"
    ],
    "authority": [
        "authority", "mint authority", "freeze authority", "owner", "can mint more",
        "mintable", "frozen", "contract owner", "token authority", "permission",
        "admin", "admin control", "controlled by", "who controls", "minting rights",
        "token controller", "contract authority", "who has authority", "governance",
        "who can mint", "who can freeze", "administrative rights", "token admin",
        "creator rights", "creator control", "controller", "token issuer", "issuing authority",
        "who's in charge", "who is in charge", "privileged accounts", "privileged operations"
    ],
    "age": [
        "age", "how old", "launch date", "created when", "token age", "inception date",
        "born", "genesis", "started", "creation date", "when was it created", "date created",
        "birthday", "how long", "since when", "established", "founded", "launch",
        "when launched", "release date", "when released", "deployment date", "when deployed",
        "when did it start", "start date", "beginning", "origin", "origination date",
        "mint date", "when minted", "first transaction", "initial transaction", "first appearance"
    ],
    "risk": [
        "risk", "dangerous", "scam", "fraud", "rug pull", "rug", "dump", "honeypot",
        "fake", "scammer", "security risk", "vulnerability", "vulnerable", "exploit",
        "how safe", "safety", "trustworthy", "legit", "legitimate", "reliable",
        "risk assessment", "risk analysis", "risk level", "risk factors", "red flags",
        "warning signs", "suspicious", "trust", "safe to buy", "investment risk",
        "potential risks", "security concerns", "security assessment", "risky", "harmful"
    ],
    "comparison": [
        "compare", "versus", "vs", "difference between", "better than", "worse than",
        "compared to", "against", "similarity", "different from", "similar to",
        "in relation to", "how does it compare", "stack up against", "contrast with",
        "side by side", "next to", "comparison with", "distinction from", "relative to",
        "as opposed to", "in contrast to", "performance against", "measures up to",
        "in comparison with", "comparable to", "match up to", "match against", "how it ranks"
    ]
}

# Stopwords to filter out from queries
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "which", "who", "whom", "this", "that", "these",
    "those", "then", "just", "so", "than", "such", "both", "through", "about",
    "for", "is", "of", "while", "during", "to", "from", "in", "out", "on", "off",
    "again", "further", "then", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "should", "now"
])

class TokenQueryAnalyzer:
    """Analyzer for natural language queries about tokens."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)
        self.token_analyzer = TokenAnalyzer(solana_client)
    
    async def analyze_query(self, query: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a natural language query and return relevant token data.
        
        Args:
            query: The natural language query
            request_id: Optional request ID for tracing
            
        Returns:
            Analysis result based on the query intent
        """
        log_with_context(
            logger,
            "info",
            f"Analyzing query: {query}",
            request_id=request_id,
            query=query
        )
        
        # Normalize and clean the query
        normalized_query = self._normalize_query(query)
        cleaned_tokens = self._clean_query(normalized_query)
        
        # Extract intent
        intents = self._extract_intents(normalized_query)
        primary_intent = intents[0][0] if intents else "general_info"
        confidence = intents[0][1] if intents else 0.0
        
        # Extract token address (if present in the query)
        token_address = self._extract_token_address(query)  # Using original query to preserve case
        
        # Extract threshold value (for whale analysis)
        threshold = self._extract_threshold(normalized_query)
        
        result = {
            "query": query,
            "normalized_query": normalized_query,
            "primary_intent": primary_intent,
            "all_intents": [{"intent": intent, "confidence": conf} for intent, conf in intents],
            "confidence": confidence,
            "extracted_parameters": {
                "token_address": token_address,
                "threshold": threshold
            },
            "data": None,
            "error": None
        }
        
        # If no token address was found, return early
        if not token_address:
            result["error"] = "No token address found in query"
            return result
        
        # Validate the token address
        if not validate_public_key(token_address):
            result["error"] = f"Invalid token address: {token_address}"
            return result
            
        try:
            # First, always get basic token metadata to ensure consistent responses
            token_metadata = await self.token_analyzer.get_token_metadata(token_address, request_id=request_id)
            basic_token_info = {
                "token_mint": token_address,
                "token_name": token_metadata.get("name", "Unknown"),
                "token_symbol": token_metadata.get("symbol", "UNKNOWN")
            }
            
            # Get supply info for all queries as well
            try:
                supply_info = await self.token_analyzer.get_token_supply_and_decimals(token_address, request_id=request_id)
                decimals = supply_info.get("value", {}).get("decimals", 0)
                total_supply = supply_info.get("value", {}).get("uiAmountString", "0")
                basic_token_info["decimals"] = decimals
                basic_token_info["total_supply"] = total_supply
            except Exception as e:
                logger.error(f"Error getting supply data: {str(e)}", exc_info=True)
            
            # Execute the appropriate analysis based on intent
            if primary_intent == "general_info":
                analysis = await self.token_analyzer.analyze_token(token_address, request_id=request_id)
                # Convert dataclass to dict
                data = self._dataclass_to_dict(analysis)
                result["data"] = data
                
            elif primary_intent == "price":
                price_data = await self.token_analyzer.get_token_price(token_address, request_id=request_id)
                # Merge with basic token info
                result["data"] = {**basic_token_info, **price_data}
                
            elif primary_intent == "holders":
                holders_data = await self.token_analyzer.get_token_largest_holders(token_address, request_id=request_id)
                # Merge with basic token info
                result["data"] = {**basic_token_info, **holders_data}
                
            elif primary_intent == "whales":
                # Use the threshold if provided, otherwise use default
                whale_data = await self.token_analyzer.get_whale_holders(
                    token_address, 
                    threshold_usd=threshold or 50000.0, 
                    request_id=request_id
                )
                
                # Ensure total_holders is properly included
                if "total_holders" not in whale_data:
                    # Get total holders data if not already included
                    holders_data = await self.token_analyzer.get_token_largest_holders(token_address, request_id=request_id)
                    whale_data["total_holders"] = holders_data.get("total_holders", whale_data.get("total_holders_analyzed", 0))
                
                # Merge with basic token info, ensuring token_name and token_symbol are preserved
                if "token_name" not in whale_data or whale_data["token_name"] == "Unknown":
                    whale_data["token_name"] = basic_token_info["token_name"]
                if "token_symbol" not in whale_data or whale_data["token_symbol"] == "UNKNOWN":
                    whale_data["token_symbol"] = basic_token_info["token_symbol"]
                
                result["data"] = whale_data
                
            elif primary_intent == "fresh_wallets":
                fresh_wallet_data = await self.token_analyzer.get_fresh_wallets(token_address, request_id=request_id)
                # Merge with basic token info
                if "token_name" not in fresh_wallet_data or fresh_wallet_data["token_name"] == "Unknown":
                    fresh_wallet_data["token_name"] = basic_token_info["token_name"]
                if "token_symbol" not in fresh_wallet_data or fresh_wallet_data["token_symbol"] == "UNKNOWN":
                    fresh_wallet_data["token_symbol"] = basic_token_info["token_symbol"]
                result["data"] = fresh_wallet_data
                
            elif primary_intent == "supply":
                supply_data = await self.token_analyzer.get_token_supply_and_decimals(token_address, request_id=request_id)
                # Merge with basic token info
                result["data"] = {**basic_token_info, **supply_data}
                
            elif primary_intent == "authority":
                authority_data = await self.token_analyzer.get_token_mint_authority(token_address, request_id=request_id)
                # Merge with basic token info
                result["data"] = {**basic_token_info, **authority_data}
                
            elif primary_intent == "age":
                age_data = await self.token_analyzer.get_token_age(token_address, request_id=request_id)
                # Merge with basic token info if not already present
                if "token_name" not in age_data or age_data["token_name"] == "Unknown":
                    age_data["token_name"] = basic_token_info["token_name"]
                if "token_symbol" not in age_data or age_data["token_symbol"] == "UNKNOWN":
                    age_data["token_symbol"] = basic_token_info["token_symbol"]
                result["data"] = age_data
                
            elif primary_intent == "risk":
                # For risk analysis, we do a comprehensive token analysis
                analysis = await self.token_analyzer.analyze_token(token_address, request_id=request_id)
                data = self._dataclass_to_dict(analysis)
                
                # Extract risk-specific data points
                risk_data = {
                    "token_mint": data.get("token_mint"),
                    "token_name": data.get("token_name"),
                    "token_symbol": data.get("token_symbol"),
                    "decimals": data.get("decimals"),
                    "total_supply": data.get("total_supply"),
                    "age_days": data.get("age_days"),
                    "owner_can_mint": data.get("owner_can_mint"),
                    "owner_can_freeze": data.get("owner_can_freeze"),
                    "total_holders": data.get("total_holders", 0),
                    "largest_holder_percentage": data.get("largest_holder_percentage"),
                    "whale_holdings_percentage": data.get("whale_holdings_percentage"),
                    "fresh_wallet_percentage": data.get("fresh_wallet_percentage"),
                    "fresh_wallet_holdings_percentage": data.get("fresh_wallet_holdings_percentage"),
                    "risk_factors": [],
                    "risk_level": "Low"
                }
                
                # Identify risk factors
                if data.get("owner_can_mint"):
                    risk_data["risk_factors"].append("Token can be minted by authority")
                if data.get("owner_can_freeze"):
                    risk_data["risk_factors"].append("Token accounts can be frozen by authority")
                if data.get("largest_holder_percentage", 0) > 20:
                    risk_data["risk_factors"].append(f"High concentration: Largest holder owns {data.get('largest_holder_percentage')}%")
                if data.get("whale_holdings_percentage", 0) > 50:
                    risk_data["risk_factors"].append(f"Whale dominance: Whales hold {data.get('whale_holdings_percentage')}%")
                if data.get("fresh_wallet_percentage", 0) > 30:
                    risk_data["risk_factors"].append(f"High proportion of fresh wallets: {data.get('fresh_wallet_percentage')}%")
                if data.get("age_days", 0) < 30:
                    risk_data["risk_factors"].append(f"New token: Only {data.get('age_days')} days old")
                
                # Determine risk level
                if len(risk_data["risk_factors"]) >= 3:
                    risk_data["risk_level"] = "High"
                elif len(risk_data["risk_factors"]) >= 1:
                    risk_data["risk_level"] = "Medium"
                
                result["data"] = risk_data
                
            elif primary_intent == "comparison":
                # Not implemented yet - would need another token to compare with
                result["error"] = "Comparison functionality not yet implemented"
                
            else:
                # Fallback to general token analysis for unknown intents
                analysis = await self.token_analyzer.analyze_token(token_address, request_id=request_id)
                # Convert dataclass to dict
                data = self._dataclass_to_dict(analysis)
                result["data"] = data
                
            # Check if we need to perform multi-intent analysis
            if len(intents) > 1 and intents[1][1] > 0.5:  # If second intent has high confidence
                secondary_data = {}
                secondary_intent = intents[1][0]
                
                # Execute secondary analysis based on the second intent
                if secondary_intent == "price" and primary_intent != "price":
                    price_data = await self.token_analyzer.get_token_price(token_address, request_id=request_id)
                    secondary_data["price"] = {**basic_token_info, **price_data}
                
                elif secondary_intent == "whales" and primary_intent != "whales":
                    whale_data = await self.token_analyzer.get_whale_holders(
                        token_address, 
                        threshold_usd=threshold or 50000.0, 
                        request_id=request_id
                    )
                    # Merge with basic token info
                    if "token_name" not in whale_data or whale_data["token_name"] == "Unknown":
                        whale_data["token_name"] = basic_token_info["token_name"]
                    if "token_symbol" not in whale_data or whale_data["token_symbol"] == "UNKNOWN":
                        whale_data["token_symbol"] = basic_token_info["token_symbol"]
                    secondary_data["whales"] = whale_data
                
                # Add additional data if any secondary analysis was performed
                if secondary_data:
                    result["additional_data"] = secondary_data
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Query analysis completed for: {query}",
            request_id=request_id,
            primary_intent=primary_intent,
            confidence=confidence
        )
        
        return result
    
    def _normalize_query(self, query: str) -> str:
        """Normalize a query for consistent processing.
        
        Args:
            query: The original query
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Replace contractions
        contractions = {
            "what's": "what is",
            "who's": "who is",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "i'm": "i am",
            "you're": "you are",
            "they're": "they are",
            "we're": "we are",
            "isn't": "is not",
            "wasn't": "was not",
            "aren't": "are not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mightn't": "might not",
            "mustn't": "must not",
            "who'll": "who will",
            "what'll": "what will",
            "how'll": "how will"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _clean_query(self, query: str) -> List[str]:
        """Clean and tokenize a query.
        
        Args:
            query: The normalized query
            
        Returns:
            List of cleaned tokens
        """
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        query_no_punct = query.translate(translator)
        
        # Tokenize
        tokens = query_no_punct.split()
        
        # Remove stopwords
        cleaned_tokens = [token for token in tokens if token not in STOPWORDS]
        
        return cleaned_tokens
    
    def _extract_intents(self, query: str) -> List[Tuple[str, float]]:
        """Extract all possible intents from a query with confidence scores.
        
        Args:
            query: The normalized query
            
        Returns:
            List of (intent, confidence) tuples sorted by confidence
        """
        # Count matches for each intent
        intent_scores = defaultdict(float)
        
        # First pass - exact phrase matching
        for intent, keywords in INTENTS.items():
            for keyword in keywords:
                if keyword in query:
                    # Add score based on keyword length (longer keywords are more specific)
                    # Weight phrases higher than single words
                    if ' ' in keyword:
                        intent_scores[intent] += len(keyword) / 3
                    else:
                        intent_scores[intent] += len(keyword) / 5
        
        # Second pass - word frequency analysis
        word_count = defaultdict(int)
        words = query.split()
        for word in words:
            word_count[word] += 1
        
        # Check if any high-frequency words match our intent keywords
        for intent, keywords in INTENTS.items():
            for keyword in keywords:
                if ' ' not in keyword and keyword in word_count:
                    # Words that appear multiple times get higher scores
                    intent_scores[intent] += word_count[keyword] * 0.5
        
        # Third pass - position-based weighting
        # Words at the beginning of the query might be more important
        for intent, keywords in INTENTS.items():
            for keyword in keywords:
                if ' ' not in keyword and query.startswith(keyword + ' '):
                    intent_scores[intent] += 2.0
        
        # Convert to list of tuples and sort by score descending
        intent_list = [(intent, score) for intent, score in intent_scores.items()]
        intent_list.sort(key=lambda x: x[1], reverse=True)
        
        # If we have no matches, default to general_info
        if not intent_list:
            return [("general_info", 0.5)]
            
        # Normalize confidence scores (0-1 range)
        max_possible_score = 20.0  # Reasonable max score threshold
        normalized_intents = []
        for intent, score in intent_list:
            confidence = min(score / max_possible_score, 1.0)
            # Only include intents with some minimum confidence
            if confidence >= 0.1:
                normalized_intents.append((intent, confidence))
        
        return normalized_intents
    
    def _extract_token_address(self, query: str) -> Optional[str]:
        """Extract token address from a query.
        
        Args:
            query: The original query (not normalized to preserve case)
            
        Returns:
            Token address if found, None otherwise
        """
        # Look for Solana addresses (base58 format)
        address_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        matches = re.findall(address_pattern, query)
        
        if matches:
            # Validate each match and return the first valid one
            for match in matches:
                if validate_public_key(match):
                    return match
        
        return None
    
    def _extract_threshold(self, query: str) -> Optional[float]:
        """Extract threshold value for whale analysis.
        
        Args:
            query: The normalized query
            
        Returns:
            Threshold value if found, None otherwise
        """
        # Look for currency amounts (e.g., $10000, 10k, 10,000, etc.)
        currency_patterns = [
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*k\b',  # $10k specifically with k as separate word boundary
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # $10000, $10,000
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k\b\s*(?:usd|USD)?',  # 10k, 10K USD with k as word boundary
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:usd|USD)'  # 10000 USD
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Take the first match
                value_str = matches[0].replace(',', '')
                try:
                    value = float(value_str)
                    # If pattern contains 'k' or 'K', multiply by 1000
                    if 'k' in pattern.lower():
                        value *= 1000
                    return value
                except ValueError:
                    continue
        
        # Look for numerical thresholds with 'more than', 'over', etc.
        threshold_patterns = [
            r'more than (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'greater than (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'over (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'above (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'exceeding (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'at least (?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'minimum (?:of )?(?:\$\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b\s*(?:usd|USD)?\s*(?:or more|and up|plus)'
        ]
        
        for pattern in threshold_patterns:
            matches = re.findall(pattern, query)
            if matches:
                value_str = matches[0].replace(',', '')
                try:
                    value = float(value_str)
                    # Check if 'k' is in the matched pattern
                    if 'k' in pattern[-10:].lower() and 'k' in query[query.find(matches[0])-5:query.find(matches[0])+len(matches[0])+5].lower():
                        value *= 1000
                    return value
                except ValueError:
                    continue
        
        return None
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert a dataclass instance to a dictionary.
        
        Args:
            obj: The dataclass instance
            
        Returns:
            Dictionary representation of the dataclass
        """
        # Handle None values
        if obj is None:
            return {}
            
        # Check if object has __dataclass_fields__ attribute
        if not hasattr(obj, '__dataclass_fields__'):
            # If this is already a dict, return it
            if isinstance(obj, dict):
                return obj
            # If it's a simple type, return as is
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            # For lists, process each item
            if isinstance(obj, list):
                return [self._dataclass_to_dict(item) for item in obj]
            # For other types, try to convert to string
            return str(obj)
        
        result = {}
        # Process each field in the dataclass
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            # Handle datetime objects
            if hasattr(value, 'isoformat'):
                result[field] = value.isoformat()
            # Handle nested dataclasses
            elif hasattr(value, '__dataclass_fields__'):
                result[field] = self._dataclass_to_dict(value)
            # Handle lists of dataclasses
            elif isinstance(value, list) and value and hasattr(value[0], '__dataclass_fields__'):
                result[field] = [self._dataclass_to_dict(item) for item in value]
            # Handle other types
            else:
                result[field] = value
                
        return result 