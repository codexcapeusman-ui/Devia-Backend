"""
Unified AI Agent Service - Refactored Version
Handles single prompt workflow with intent detection, data extraction, and response formatting
Now with database-backed conversations, per-field attempt tracking, and improved accuracy
"""

import json
import logging
import re
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from bson import ObjectId

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings

from services.semantic_kernel_service import SemanticKernelService
from tools.client_tools import ClientTools
from tools.invoice_tools import InvoiceTools
from tools.quote_tools import QuoteTools
from tools.job_tools import JobTools
from tools.expense_tools import ExpenseTools
from tools.manual_task_tools import ManualTaskTools
from config.settings import Settings
from database import get_conversations_collection, get_settings_collection
from models.chats import (
    ChatMessage, Conversation, ConversationCreate, 
    MessageRole, ConversationState as DBConversationState
)


class Intent(str, Enum):
    """Supported intents for AI agent - Order matters for priority"""
    CHIT_CHAT = "chit_chat"
    USER_SETTINGS = "user_settings"  # Profile, company info, rates, integrations
    GENERAL_INFO = "general_info"  # General questions, calculations, information requests
    MANUAL_TASK = "manual_task"
    CUSTOMER = "customer"
    INVOICE = "invoice"
    QUOTE = "quote"
    EXPENSE = "expense"
    JOB = "job"
    UNKNOWN = "unknown"


class Operation(str, Enum):
    """Supported operations for AI agent"""
    GET = "get"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UNKNOWN = "unknown"


class ConversationState(str, Enum):
    """States of conversation flow"""
    INTENT_DETECTION = "intent_detection"
    DATA_EXTRACTION = "data_extraction"
    DATA_COMPLETION = "data_completion"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"


# Maximum attempts per field before filling with N/A
MAX_FIELD_ATTEMPTS = 2

# Minimum confidence for auto-switching intents
INTENT_SWITCH_CONFIDENCE = 0.7


class UnifiedAgentService:
    """
    Unified service that handles all AI agent interactions through a single endpoint
    Workflow: Prompt -> Intent Detection -> Data Extraction -> Missing Data Check -> Response
    
    Key improvements:
    - Database-backed conversation storage
    - Per-field attempt tracking (max 2 attempts per field)
    - Intent switching at confidence >= 0.7
    - Improved extraction accuracy with focused prompts
    - Context parameter support
    """
    
    def __init__(self, sk_service: SemanticKernelService, settings: Settings = None):
        self.sk_service = sk_service
        self.logger = logging.getLogger(__name__)
        
        # Initialize settings
        if settings is None:
            from config.settings import Settings
            settings = Settings()
        self.settings = settings
        
        # Initialize tools
        self.client_tools = ClientTools(settings)
        self.invoice_tools = InvoiceTools(settings)
        self.quote_tools = QuoteTools(settings)
        self.job_tools = JobTools(settings)
        self.expense_tools = ExpenseTools(settings)
        self.manual_task_tools = ManualTaskTools(settings)
        
        # In-memory cache for active conversations (backed by database)
        self._conversation_cache: Dict[str, Dict] = {}
        
        # Required fields for each intent
        self.required_fields = {
            Intent.MANUAL_TASK: ["title", "start_time", "end_time"],
            Intent.CUSTOMER: ["name", "email", "phone", "address"],
            Intent.INVOICE: ["customer_name", "customer_email", "items", "total_amount", "title"],
            Intent.QUOTE: ["customer_name", "customer_email", "services", "estimated_total"],
            Intent.EXPENSE: ["description", "amount", "date", "category"],
            Intent.JOB: ["title", "customer_name", "scheduled_date", "duration"]
        }
        
        # Field aliases for smart matching
        self.field_aliases = {
            "customer_name": ["customer_name", "client_name", "clientName", "name", "clientname"],
            "customer_email": ["customer_email", "client_email", "clientEmail", "email", "clientemail"],
            "services": ["services", "items", "line_items", "lineItems", "service_items"],
            "items": ["items", "services", "line_items", "lineItems", "invoice_items"],
            "estimated_total": ["estimated_total", "estimatedTotal", "total", "total_amount", "totalAmount"],
            "total_amount": ["total_amount", "totalAmount", "total", "estimated_total", "estimatedTotal"],
            "title": ["title", "name", "project_name", "projectName", "description"],
            "description": ["description", "title", "name"],
            "amount": ["amount", "total", "total_amount", "totalAmount"],
            "date": ["date", "expense_date", "expenseDate", "created_at", "createdAt"],
            "category": ["category", "expense_category", "expenseCategory", "type"],
            "name": ["name", "customer_name", "client_name", "clientName", "title"],
            "email": ["email", "customer_email", "client_email", "clientEmail"],
            "phone": ["phone", "phone_number", "phoneNumber", "telephone"],
            "address": ["address", "full_address", "fullAddress", "street_address", "streetAddress"],
            "start_time": ["start_time", "startTime", "start_date", "startDate", "scheduled_date"],
            "end_time": ["end_time", "endTime", "end_date", "endDate"],
            "scheduled_date": ["scheduled_date", "scheduledDate", "start_date", "startDate", "date"],
            "duration": ["duration", "estimated_duration", "estimatedDuration", "hours"],
        }

    # ==================== DATABASE OPERATIONS ====================
    
    async def _get_or_create_conversation(self, user_id: str, language: str = "en", context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get active conversation from database or create new one"""
        try:
            collection = get_conversations_collection()
            
            # Find active conversation for user
            existing = await collection.find_one({
                "user_id": user_id,
                "is_active": True
            })
            
            if existing:
                # Update cache and return
                conversation = self._db_doc_to_conversation(existing)
                self._conversation_cache[user_id] = conversation
                return conversation
            
            # Create new conversation
            new_conversation = {
                "user_id": user_id,
                "state": ConversationState.INTENT_DETECTION.value,
                "intent": None,
                "operation": None,
                "confidence": 0.0,
                "messages": [],
                "extracted_data": context.get("extracted_data", {}) if context else {},
                "field_attempts": {},
                "language": language,
                "context": context or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            }
            
            result = await collection.insert_one(new_conversation)
            new_conversation["_id"] = str(result.inserted_id)
            
            conversation = self._db_doc_to_conversation(new_conversation)
            self._conversation_cache[user_id] = conversation
            return conversation
            
        except Exception as e:
            self.logger.error(f"Database error getting conversation: {e}")
            # Fallback to in-memory only
            return self._get_fallback_conversation(user_id, language, context)
    
    async def _save_conversation(self, user_id: str, conversation: Dict[str, Any]) -> None:
        """Save conversation state to database"""
        try:
            collection = get_conversations_collection()
            conversation["updated_at"] = datetime.utcnow()
            
            # Convert messages to dict format for storage
            messages_for_db = []
            for msg in conversation.get("messages", []):
                if isinstance(msg, dict):
                    messages_for_db.append(msg)
                else:
                    messages_for_db.append({
                        "role": msg.role if hasattr(msg, 'role') else msg.get("role", "user"),
                        "content": msg.content if hasattr(msg, 'content') else msg.get("content", ""),
                        "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else msg.get("timestamp", datetime.utcnow().isoformat())
                    })
            
            update_data = {
                "state": conversation["state"].value if isinstance(conversation["state"], ConversationState) else conversation["state"],
                "intent": conversation.get("intent").value if isinstance(conversation.get("intent"), Intent) else conversation.get("intent"),
                "operation": conversation.get("operation").value if isinstance(conversation.get("operation"), Operation) else conversation.get("operation"),
                "confidence": conversation.get("confidence", 0.0),
                "messages": messages_for_db,
                "extracted_data": conversation.get("data", {}),
                "field_attempts": conversation.get("field_attempts", {}),
                "updated_at": datetime.utcnow(),
                "is_active": conversation.get("is_active", True)
            }
            
            if conversation.get("_id"):
                await collection.update_one(
                    {"_id": ObjectId(conversation["_id"])},
                    {"$set": update_data}
                )
            else:
                await collection.update_one(
                    {"user_id": user_id, "is_active": True},
                    {"$set": update_data}
                )
            
            # Update cache
            self._conversation_cache[user_id] = conversation
            
        except Exception as e:
            self.logger.error(f"Database error saving conversation: {e}")
    
    async def _close_conversation(self, user_id: str) -> None:
        """Mark conversation as completed/inactive in database"""
        try:
            collection = get_conversations_collection()
            await collection.update_one(
                {"user_id": user_id, "is_active": True},
                {"$set": {"is_active": False, "state": "completed", "updated_at": datetime.utcnow()}}
            )
            
            # Clear cache
            if user_id in self._conversation_cache:
                del self._conversation_cache[user_id]
                
        except Exception as e:
            self.logger.error(f"Database error closing conversation: {e}")
    
    def _db_doc_to_conversation(self, doc: Dict) -> Dict[str, Any]:
        """Convert database document to conversation dict"""
        return {
            "_id": str(doc.get("_id", "")),
            "user_id": doc.get("user_id"),
            "state": ConversationState(doc.get("state", "intent_detection")),
            "intent": Intent(doc.get("intent")) if doc.get("intent") else None,
            "operation": Operation(doc.get("operation")) if doc.get("operation") else None,
            "confidence": doc.get("confidence", 0.0),
            "messages": doc.get("messages", []),
            "data": doc.get("extracted_data", {}),
            "field_attempts": doc.get("field_attempts", {}),
            "language": doc.get("language", "en"),
            "context": doc.get("context", {}),
            "created_at": doc.get("created_at", datetime.utcnow()),
            "updated_at": doc.get("updated_at", datetime.utcnow()),
            "is_active": doc.get("is_active", True)
        }
    
    def _get_fallback_conversation(self, user_id: str, language: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Fallback in-memory conversation when database is unavailable"""
        if user_id not in self._conversation_cache:
            self._conversation_cache[user_id] = {
                "user_id": user_id,
                "state": ConversationState.INTENT_DETECTION,
                "intent": None,
                "operation": None,
                "confidence": 0.0,
                "messages": [],
                "data": context.get("extracted_data", {}) if context else {},
                "field_attempts": {},
                "language": language,
                "context": context or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            }
        return self._conversation_cache[user_id]

    # ==================== MAIN PROCESSING ====================
    
    async def process_agent_request(
        self, 
        prompt: str, 
        user_id: str, 
        language: str = "en",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for unified agent processing
        
        Args:
            prompt: User's natural language prompt
            user_id: Unique identifier for the user
            language: Language preference (en/fr)
            context: Optional context data (client_id, quote_id, etc.)
            
        Returns:
            Unified response with status, data, and next action
        """
        try:
            self.logger.info(f"Processing unified request for user {user_id}: {prompt[:100]}...")
            
            # Get or create conversation state from database
            conversation = await self._get_or_create_conversation(user_id, language, context)
            self.logger.info(f"Conversation state: {conversation['state']}, field_attempts: {conversation.get('field_attempts', {})}")
            
            # Handle quick commands: reset/cancel
            lower_prompt = prompt.strip().lower()
            if any(cmd in lower_prompt for cmd in ["never mind", "cancel", "start over", "reset", "stop"]):
                await self._close_conversation(user_id)
                return {
                    "success": True,
                    "message": "Conversation reset. How can I help you now?",
                    "action": "reset"
                }

            # Add user message to history
            conversation["messages"].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.utcnow().isoformat()
            })

            # STEP 1: Intent Detection or Re-detection
            if conversation["state"] == ConversationState.INTENT_DETECTION:
                intent, operation, confidence = await self._detect_intent(prompt, language)
                conversation["intent"] = intent
                conversation["operation"] = operation
                conversation["confidence"] = confidence
                conversation["data"] = context.get("extracted_data", {}) if context else {}
                conversation["field_attempts"] = {}

                self.logger.info(f"Intent detection result: intent={intent}, operation={operation}, confidence={confidence}")

                # Handle CHIT_CHAT
                if intent == Intent.CHIT_CHAT:
                    chit_chat_response = await self._generate_chit_chat_response(prompt, language)
                    await self._close_conversation(user_id)
                    return chit_chat_response

                # Handle USER_SETTINGS (profile, company info, rates, integrations)
                if intent == Intent.USER_SETTINGS:
                    # Pass original prompt in data for context
                    conversation["data"]["_original_prompt"] = prompt
                    user_settings_response = await self._handle_user_settings_query(user_id, conversation["data"], language)
                    await self._close_conversation(user_id)
                    return user_settings_response

                # Handle GENERAL_INFO (informational questions, calculations, etc.)
                if intent == Intent.GENERAL_INFO:
                    general_info_response = await self._generate_general_info_response(prompt, language)
                    await self._close_conversation(user_id)
                    return general_info_response

                # Handle "get all" queries
                if operation == Operation.GET and self._is_get_all_query(prompt):
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
                elif intent == Intent.UNKNOWN or confidence < 0.1:
                    await self._save_conversation(user_id, conversation)
                    return self._create_clarification_response(conversation, language)
                else:
                    conversation["state"] = ConversationState.DATA_EXTRACTION
            
            else:
                # Check for intent switch during data collection (auto-switch at >= 0.7)
                new_intent, new_operation, new_confidence = await self._detect_intent(prompt, language)
                
                # Handle GET queries immediately
                if new_operation == Operation.GET and self._is_get_all_query(prompt):
                    self.logger.info(f"Detected 'get all' query, switching to {new_intent.value}")
                    conversation["intent"] = new_intent
                    conversation["operation"] = new_operation
                    conversation["confidence"] = new_confidence
                    conversation["data"] = {}
                    conversation["field_attempts"] = {}
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
                
                # Auto-switch intent at confidence >= 0.7
                elif new_intent != Intent.UNKNOWN and new_intent != conversation.get("intent") and new_confidence >= INTENT_SWITCH_CONFIDENCE:
                    self.logger.info(f"Auto-switching intent from {conversation.get('intent')} to {new_intent} (confidence={new_confidence})")
                    conversation["intent"] = new_intent
                    conversation["operation"] = new_operation
                    conversation["confidence"] = new_confidence
                    conversation["data"] = {}
                    conversation["field_attempts"] = {}
                    conversation["state"] = ConversationState.DATA_EXTRACTION

            # STEP 2: Data Extraction
            if conversation["state"] in [ConversationState.DATA_EXTRACTION, ConversationState.DATA_COMPLETION]:
                extracted_data = await self._extract_data(
                    prompt, 
                    conversation["intent"], 
                    conversation.get("operation", Operation.UNKNOWN), 
                    language, 
                    conversation["messages"],
                    conversation.get("data", {})  # Pass existing data for cumulative extraction
                )
                
                # Merge with existing data
                self._merge_conversation_data(conversation["data"], extracted_data)
                conversation["state"] = ConversationState.DATA_COMPLETION

            # STEP 3: Check Missing Data with Per-Field Attempt Tracking
            if conversation["state"] == ConversationState.DATA_COMPLETION:
                missing_fields = self._check_missing_data(
                    conversation["intent"], 
                    conversation.get("operation", Operation.UNKNOWN), 
                    conversation["data"]
                )
                
                if missing_fields:
                    # Filter fields that haven't exceeded max attempts
                    fields_to_ask = []
                    fields_to_fill_na = []
                    
                    for field in missing_fields:
                        current_attempts = conversation.get("field_attempts", {}).get(field, 0)
                        if current_attempts >= MAX_FIELD_ATTEMPTS:
                            fields_to_fill_na.append(field)
                        else:
                            fields_to_ask.append(field)
                            # Increment attempt counter for this field
                            if "field_attempts" not in conversation:
                                conversation["field_attempts"] = {}
                            conversation["field_attempts"][field] = current_attempts + 1
                    
                    # Fill N/A for fields that exceeded attempts
                    for field in fields_to_fill_na:
                        self.logger.info(f"Max attempts reached for field '{field}', filling with N/A")
                        if field in ["total_amount", "estimated_total", "amount"]:
                            conversation["data"][field] = 0.0
                        elif field in ["items", "services"]:
                            conversation["data"][field] = []
                        else:
                            conversation["data"][field] = "N/A"
                    
                    # If there are still fields to ask about
                    if fields_to_ask:
                        await self._save_conversation(user_id, conversation)
                        response = self._create_missing_data_response(conversation, fields_to_ask, language)
                        conversation["messages"].append({
                            "role": "assistant",
                            "content": response["message"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        await self._save_conversation(user_id, conversation)
                        return response
                    else:
                        # All fields either filled or N/A'd
                        conversation["state"] = ConversationState.RESPONSE_GENERATION
                else:
                    conversation["state"] = ConversationState.RESPONSE_GENERATION

            # STEP 4: Generate Final Response
            if conversation["state"] == ConversationState.RESPONSE_GENERATION:
                response = await self._generate_final_response(
                    conversation["intent"], 
                    conversation.get("operation", Operation.UNKNOWN), 
                    conversation["data"], 
                    language, 
                    user_id
                )
                
                # Add response to history and close conversation
                conversation["messages"].append({
                    "role": "assistant",
                    "content": response.get("message", ""),
                    "timestamp": datetime.utcnow().isoformat()
                })
                conversation["state"] = ConversationState.COMPLETED
                await self._save_conversation(user_id, conversation)
                
                if response.get("success", False):
                    await self._close_conversation(user_id)
                
                return response

            # Fallback: try re-detection
            try:
                alt_intent, alt_operation, alt_conf = await self._detect_intent(prompt, language)
                if alt_intent != Intent.UNKNOWN and alt_conf >= 0.2:
                    conversation["intent"] = alt_intent
                    conversation["operation"] = alt_operation
                    conversation["confidence"] = alt_conf
                    conversation["data"] = {}
                    conversation["field_attempts"] = {}
                    conversation["state"] = ConversationState.RESPONSE_GENERATION if alt_operation == Operation.GET else ConversationState.DATA_EXTRACTION
                    await self._save_conversation(user_id, conversation)
                    return await self.process_agent_request(prompt, user_id, language, context)
            except Exception:
                self.logger.debug("Fallback re-detection failed")

            return self._create_error_response("Invalid conversation state", language)
            
        except Exception as e:
            self.logger.error(f"Error processing agent request: {e}")
            return self._create_error_response(str(e), language)

    # ==================== INTENT DETECTION ====================
    
    def _extract_person_name_from_query(self, prompt: str) -> Optional[str]:
        """Extract person name from person-related queries like 'who is Eva Malik'"""
        prompt_lower = prompt.strip().lower()
        prompt_clean = prompt.strip()  # Keep original case for name
        
        # Patterns with named groups to extract the person name
        patterns_with_groups = [
            (r"^(?:who|how)\s+(?:is|are|was)\s+(.+?)[\s!.?]*$", 1),
            (r"^tell\s+me\s+about\s+(.+?)[\s!.?]*$", 1),
            (r"^(?:show|get|find)\s+(?:me\s+)?(?:info|information|details?)\s+(?:about|on|for)\s+(.+?)[\s!.?]*$", 1),
            (r"^(?:what|who)\s+about\s+(.+?)[\s!.?]*$", 1),
            (r"^(?:do\s+(?:you|we)\s+have|is\s+there)\s+(?:a\s+)?(?:client|customer)?\s*(?:named|called)?\s*(.+?)[\s!.?]*$", 1),
            (r"^(?:find|search|look\s*up)\s+(.+?)[\s!.?]*$", 1),
        ]
        
        for pattern, group_idx in patterns_with_groups:
            match = re.search(pattern, prompt_clean, re.IGNORECASE)
            if match:
                name = match.group(group_idx).strip()
                # Clean up common words that shouldn't be part of name
                name = re.sub(r'\b(client|customer|named|called)\b', '', name, flags=re.IGNORECASE).strip()
                if name and len(name) > 1:
                    self.logger.info(f"Extracted person name from query: '{name}'")
                    return name
        return None
    
    async def _detect_intent(self, prompt: str, language: str) -> Tuple[Intent, Operation, float]:
        """Detect user intent from the prompt using AI with improved accuracy"""
        
        # Quick regex check for chit-chat (ONLY simple greetings without names)
        chit_chat_patterns = [
            r"^(hi|hello|hey|bonjour|salut|coucou|good\s*(morning|afternoon|evening)|what'?s?\s*up)[\s!.?]*$",
            r"^(thanks?|thank\s*you|merci|awesome|great|perfect|ok|okay|cool|nice|got\s*it)[\s!.?]*$",
            r"^(bye|goodbye|see\s*you|au\s*revoir|ciao|later)[\s!.?]*$",
            r"^(how\s*are\s*you|how'?s?\s*it\s*going|comment\s*(ça\s*)?va|ça\s*va)[\s!.?]*$",
            r"^(who\s*are\s*you|what\s*can\s*you\s*do|help|aide)[\s!.?]*$",
        ]
        
        prompt_lower = prompt.strip().lower()
        
        # Check for USER SETTINGS / PROFILE queries FIRST
        user_settings_patterns = [
            r"\b(my|our)\s+(company|business|profile|settings?|rates?|pricing|integrations?)\b",
            r"\b(show|get|what|tell)\s+(me\s+)?(my|our|about\s+my)\s+",
            r"\b(standard|hourly|service)\s+rates?\b",
            r"\bmy\s+(plumbing|electrical|carpentry|painting|hvac|roofing)\s+(rates?|prices?)\b",
            r"\b(google\s*calendar|stripe|payment)\s+(integration|status|connected|setup)\b",
            r"\bintegration\s+(status|settings?)\b",
            r"\b(am\s+i|are\s+we)\s+(connected|integrated|synced)\b",
            r"\bwhat\s+is\s+(my|our)\s+(company|business)\s*(name)?\b",
            r"\bmy\s+(email|phone|address|contact)\b",
        ]
        
        for pattern in user_settings_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                self.logger.info(f"Detected user settings query pattern")
                return Intent.USER_SETTINGS, Operation.GET, 0.90
        
        # Check for person name queries - these should check database
        person_name = self._extract_person_name_from_query(prompt)
        if person_name:
            # This looks like a person/entity query - treat as CUSTOMER GET
            self.logger.info(f"Detected person query for '{person_name}', treating as customer lookup")
            return Intent.CUSTOMER, Operation.GET, 0.85
        
        for pattern in chit_chat_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return Intent.CHIT_CHAT, Operation.UNKNOWN, 0.95
        
        # Simplified, focused intent detection prompt
        intent_prompt = f"""Analyze this user prompt and determine intent and operation. Return JSON only.

User prompt: "{prompt}"

IMPORTANT: This is a BUSINESS MANAGEMENT APP for invoices, quotes, clients, jobs, and expenses.
It is NOT a general knowledge platform. All queries should be interpreted in business context.

INTENTS (in priority order):
1. chit_chat - ONLY simple greetings without names (hi, hello, thanks, bye)
2. user_settings - Questions about MY/OUR profile, company, rates, integrations:
   - "What is my company name?" → user_settings
   - "Show me my standard rates" → user_settings
   - "What are my plumbing rates?" → user_settings
   - "Is Google Calendar connected?" → user_settings
   - "Show my integrations" → user_settings
   - "What's my business address?" → user_settings
3. customer - ANY query mentioning a person's name should check if they're a client:
   - "who is [Name]" → Look up client named [Name]
   - "how is [Name]" → Look up client named [Name]  
   - "tell me about [Name]" → Look up client
   - "show me [Name]'s info" → Look up client
4. general_info - ONLY for service/price calculations WITHOUT person names:
   - Price/cost calculations ("how much would X cost?", "calculate cost of...")
   - Estimation requests ("estimate for 60m² apartment")
5. manual_task - Personal tasks with colors (red task, blue reminder, planning)
6. invoice - Billing (create invoice, show invoices) - ONLY when user wants to CREATE a record
7. quote - Estimates for SPECIFIC CLIENTS (create quote FOR client X)
8. expense - Cost tracking (add expense, show expenses)
9. job - Client work appointments (schedule job for client X)

OPERATIONS:
- get: Retrieve/show/list/display data
- create: Add/make/schedule/generate new
- update: Modify/change/edit existing
- delete: Remove/cancel existing
- unknown: For chit-chat and user_settings

CRITICAL RULES:
- Questions about "my rates", "my company", "my integrations" = user_settings
- If asking "how much", "calculate", "estimate" WITHOUT a specific client name = general_info
- If asking to CREATE quote/invoice FOR a specific person/company = quote/invoice
- General cost questions = general_info (NOT quote or invoice)
- Color words (red, blue, green) = manual_task
- "client"/"customer" = customer

Return: {{"intent": "...", "operation": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""
        
        try:
            result = await self.sk_service._execute_agent_request(
                system_prompt="You are an intent classifier. Return JSON only.",
                user_prompt=intent_prompt,
                agent_type="intent_detection"
            )
            
            if result.get("success") and result.get("data"):
                ai_response = result["data"]
                
                # Handle wrapped response
                if isinstance(ai_response, dict) and "response" in ai_response:
                    ai_response = ai_response["response"]
                
                # Parse JSON string
                if isinstance(ai_response, str):
                    ai_response = ai_response.strip()
                    if ai_response.startswith("```"):
                        ai_response = re.sub(r"```(?:json)?\s*", "", ai_response)
                        ai_response = ai_response.replace("```", "").strip()
                    try:
                        ai_response = json.loads(ai_response)
                    except json.JSONDecodeError:
                        pass
                
                if isinstance(ai_response, dict):
                    intent_str = ai_response.get("intent", "unknown")
                    operation_str = ai_response.get("operation", "unknown")
                    confidence = float(ai_response.get("confidence", 0.0))
                    
                    try:
                        intent = Intent(intent_str.lower())
                    except ValueError:
                        intent = Intent.UNKNOWN
                        confidence = 0.0
                    
                    try:
                        operation = Operation(operation_str.lower())
                    except ValueError:
                        operation = Operation.UNKNOWN
                    
                    if confidence > 0.5:
                        return intent, operation, confidence
            
        except Exception as e:
            self.logger.error(f"Intent detection failed: {e}")
        
        # Fallback pattern matching
        return self._fallback_intent_detection(prompt)
    
    def _fallback_intent_detection(self, prompt: str) -> Tuple[Intent, Operation, float]:
        """Fallback pattern-based intent detection"""
        prompt_lower = prompt.lower()
        
        # Determine operation
        operation = Operation.UNKNOWN
        if any(w in prompt_lower for w in ["show", "list", "get", "display", "see", "view", "all my", "my "]):
            operation = Operation.GET
        elif any(w in prompt_lower for w in ["create", "add", "make", "schedule", "generate", "new"]):
            operation = Operation.CREATE
        elif any(w in prompt_lower for w in ["update", "change", "modify", "edit"]):
            operation = Operation.UPDATE
        elif any(w in prompt_lower for w in ["delete", "remove", "cancel"]):
            operation = Operation.DELETE
        
        # Determine intent
        if any(w in prompt_lower for w in ["red", "blue", "green", "yellow", "orange", "purple", "pink"]):
            return Intent.MANUAL_TASK, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.85
        if "task" in prompt_lower and not any(w in prompt_lower for w in ["client", "customer", "company"]):
            return Intent.MANUAL_TASK, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.8
        if any(w in prompt_lower for w in ["client", "customer", "contact"]):
            return Intent.CUSTOMER, operation if operation != Operation.UNKNOWN else Operation.GET, 0.8
        if any(w in prompt_lower for w in ["invoice", "bill", "billing"]):
            return Intent.INVOICE, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.8
        if any(w in prompt_lower for w in ["quote", "estimate", "proposal"]):
            return Intent.QUOTE, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.8
        if any(w in prompt_lower for w in ["expense", "cost", "spending", "receipt"]):
            return Intent.EXPENSE, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.8
        if any(w in prompt_lower for w in ["job", "appointment", "meeting", "schedule"]):
            return Intent.JOB, operation if operation != Operation.UNKNOWN else Operation.CREATE, 0.7
        
        return Intent.UNKNOWN, Operation.UNKNOWN, 0.0

    # ==================== DATA EXTRACTION ====================
    
    async def _extract_data(
        self, 
        prompt: str, 
        intent: Intent, 
        operation: Operation,
        language: str,
        history: List[Dict] = None,
        existing_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract data with improved accuracy using focused prompts and cumulative extraction"""
        
        if self._is_specific_id_query(prompt, intent):
            return self._extract_id_from_prompt(prompt, intent)
        
        if operation == Operation.GET:
            return await self._extract_get_query_params(prompt, intent, language)
        
        # Build context from existing data for cumulative extraction
        existing_context = ""
        if existing_data:
            non_empty = {k: v for k, v in existing_data.items() if self._is_meaningful_value(v)}
            if non_empty:
                existing_context = f"\n\nAlready collected data:\n{json.dumps(non_empty, indent=2, default=str)}\n\nExtract any NEW or UPDATED values from the current prompt."
        
        # Build conversation context
        conversation_context = ""
        if history and len(history) > 1:
            recent_history = history[-6:]  # Last 3 exchanges
            conversation_context = "\n\nRecent conversation:\n"
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                conversation_context += f"{role}: {content}\n"
        
        # Focused extraction prompts per intent
        extraction_prompts = self._get_focused_extraction_prompt(intent)
        
        full_prompt = f"""{extraction_prompts}
{conversation_context}
{existing_context}

Current user input: "{prompt}"

Extract ALL available data. Return ONLY valid JSON, no explanations."""
        
        try:
            # Use appropriate service method
            service_method = {
                Intent.MANUAL_TASK: self.sk_service.process_manual_task_request,
                Intent.CUSTOMER: self.sk_service.process_customer_request,
                Intent.INVOICE: self.sk_service.process_invoice_request,
                Intent.QUOTE: self.sk_service.process_quote_request,
                Intent.EXPENSE: self.sk_service.process_expense_request,
                Intent.JOB: self.sk_service.process_job_request,
            }.get(intent)
            
            if not service_method:
                return {}
            
            result = await service_method(
                prompt=full_prompt,
                context={"task": "data_extraction"},
                language=language,
                history=history
            )
            
            if result.get("success") and result.get("data"):
                ai_response = result["data"]
                
                if isinstance(ai_response, dict) and "response" in ai_response:
                    ai_response = ai_response["response"]
                
                if isinstance(ai_response, str):
                    ai_response = ai_response.strip()
                    if ai_response.startswith("```"):
                        ai_response = re.sub(r"```(?:json)?\s*", "", ai_response)
                        ai_response = ai_response.replace("```", "").strip()
                    try:
                        ai_response = json.loads(ai_response)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse extraction response: {ai_response[:200]}")
                        return {}
                
                if isinstance(ai_response, dict):
                    return ai_response
            
        except Exception as e:
            self.logger.error(f"Data extraction failed for {intent}: {e}")
        
        return {}
    
    def _get_focused_extraction_prompt(self, intent: Intent) -> str:
        """Get focused extraction prompt for each intent type"""
        prompts = {
            Intent.INVOICE: """Extract invoice data. Return JSON with:
- customer_name: Client full name
- customer_email: Email address
- title: Invoice title/subject
- items: Array of {description, quantity, unit_price, total}
- total_amount: Total amount
- vat_rate: VAT rate (default 0.20)
- due_date: Due date (ISO format)
- notes: Any notes""",

            Intent.QUOTE: """Extract quote data. Return JSON with:
- customer_name: Client full name
- customer_email: Email address
- title: Quote title
- project_name: Project name
- services: Array of {description, hours, hourly_rate, total}
- estimated_total: Estimated total
- valid_until: Validity date (ISO format)""",

            Intent.CUSTOMER: """Extract customer data. Return JSON with:
- name: Full name
- email: Email address
- phone: Phone number
- address: Full address
- company: Company name (optional)
- notes: Notes (optional)""",

            Intent.JOB: """Extract job data. Return JSON with:
- title: Job title/description
- customer_name: Customer name
- scheduled_date: Date (ISO format)
- scheduled_time: Time (HH:MM)
- duration: Duration in hours
- location: Location (optional)
- notes: Notes (optional)""",

            Intent.EXPENSE: """Extract expense data. Return JSON with:
- description: Expense description
- amount: Amount spent
- date: Date (ISO format)
- category: Category (Materials, Transport, Equipment, Labor, etc.)
- vendor: Vendor name (optional)""",

            Intent.MANUAL_TASK: """Extract task data. Return JSON with:
- title: Task title/description
- start_time: Start datetime (ISO format)
- end_time: End datetime (ISO format)
- color: Color hex code (#ff0000 for red, #0000ff for blue, etc.)
- location: Location (optional)
- notes: Notes (optional)"""
        }
        return prompts.get(intent, "Extract all relevant data as JSON.")

    # ==================== HELPER METHODS ====================
    
    def _merge_conversation_data(self, existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> None:
        """Merge new data with existing, handling nested structures"""
        nested_section_patterns = [
            "extracted_data", "clientinformation", "client_information", 
            "projectdetails", "project_details", "quotedetails", "quote_details",
            "invoicedetails", "invoice_details", "discountinformation",
            "taxandtotals", "dates", "notes", "signatures"
        ]
        
        def is_nested_section(key: str) -> bool:
            normalized = key.lower().replace(" ", "").replace("_", "")
            return any(normalized == p.replace("_", "") for p in nested_section_patterns)
        
        def flatten_and_merge(data: Dict[str, Any], target: Dict[str, Any]) -> None:
            for key, value in data.items():
                if is_nested_section(key) and isinstance(value, dict):
                    flatten_and_merge(value, target)
                elif isinstance(value, list) and key.lower() in ["items", "services"]:
                    if self._is_meaningful_value(value):
                        target["services"] = value
                        target["items"] = value
                elif self._is_meaningful_value(value):
                    normalized_key = self._normalize_field_key(key)
                    target[normalized_key] = value
        
        flatten_and_merge(new_data, existing_data)
    
    def _normalize_field_key(self, key: str) -> str:
        """Normalize field keys to consistent format"""
        key_mappings = {
            "clientname": "customer_name", "client_name": "customer_name",
            "clientemail": "customer_email", "client_email": "customer_email",
            "estimatedtotal": "estimated_total", "totalamount": "total_amount",
            "total": "total_amount", "vatrate": "vat_rate",
            "projectname": "project_name", "validuntil": "valid_until",
        }
        normalized = key.lower().replace(" ", "_").replace("-", "_")
        return key_mappings.get(normalized, normalized)
    
    def _is_meaningful_value(self, value: Any) -> bool:
        """Check if value is meaningful (not empty/null/placeholder)"""
        if value is None:
            return False
        if isinstance(value, str):
            if not value.strip() or value.strip().lower() in ["", "n/a", "na", "null", "none", "undefined"]:
                return False
        elif isinstance(value, (list, dict)):
            if not value:
                return False
        return True
    
    def _check_missing_data(self, intent: Intent, operation: Operation, data: Dict[str, Any]) -> List[str]:
        """Check which required fields are missing"""
        if operation == Operation.GET:
            if data.get("query_type") == "specific_id" and not data.get("id"):
                return ["id"]
            return []
        
        required = self.required_fields.get(intent, [])
        missing = []
        
        for field in required:
            aliases = self.field_aliases.get(field, [field])
            field_found = any(alias in data and self._is_meaningful_value(data[alias]) for alias in aliases)
            if not field_found:
                missing.append(field)
        
        return missing
    
    def _is_specific_id_query(self, prompt: str, intent: Intent) -> bool:
        """Check if prompt asks for specific item by ID"""
        prompt_lower = prompt.lower()
        id_patterns = ["by id", "with id", "id:", "invoice id", "client id", "quote id", "job id"]
        
        if any(p in prompt_lower for p in id_patterns):
            return True
        
        id_regex = r'\b[a-f0-9]{24}\b|\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b'
        return bool(re.search(id_regex, prompt_lower))
    
    def _extract_id_from_prompt(self, prompt: str, intent: Intent) -> Dict[str, Any]:
        """Extract ID from prompt"""
        id_patterns = [
            r'\b[a-f0-9]{24}\b',
            r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',
            r'id[:\s]+([a-f0-9-]+)',
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return {
                    "id": match.group(1) if match.groups() else match.group(0),
                    "query_type": "specific_id"
                }
        
        return {"query_type": "specific_id", "id": None}
    
    def _is_get_all_query(self, prompt: str) -> bool:
        """Check if prompt is a simple 'get all' query WITHOUT filters"""
        prompt_lower = prompt.lower().strip()
        
        # If the prompt contains filter keywords, it's NOT a simple "get all" query
        filter_keywords = [
            "with status", "status", "where", "filter", "for", "of", "named", "called",
            "from", "since", "until", "before", "after", "between", "draft", "sent", 
            "paid", "overdue", "cancelled", "accepted", "rejected", "expired",
            "scheduled", "completed", "in_progress", "active", "archived", "delinquent"
        ]
        
        for keyword in filter_keywords:
            if keyword in prompt_lower:
                return False  # Has filters, need to extract params
        
        # Simple "get all" patterns without filters
        get_all_keywords = ["all my", "all the", "list all", "show all", "get all", "display all", "view all"]
        
        if any(k in prompt_lower for k in get_all_keywords):
            return True
        
        entities = ["clients", "invoices", "quotes", "jobs", "expenses", "tasks"]
        return any(f"my {e}" in prompt_lower or f"all {e}" in prompt_lower for e in entities)

    async def _extract_get_query_params(self, prompt: str, intent: Intent, language: str) -> Dict[str, Any]:
        """Extract search parameters for GET operations"""
        try:
            # First try to extract person name directly from query (for person queries)
            person_name = self._extract_person_name_from_query(prompt)
            if person_name and intent == Intent.CUSTOMER:
                self.logger.info(f"Using directly extracted person name for search: '{person_name}'")
                return {
                    "extracted_data": {"name": person_name, "search_term": person_name},
                    "confidence": 0.95
                }
            
            extraction_prompt = self._get_extraction_prompt_for_get(intent)
            
            result = await self.sk_service.kernel.invoke_prompt(
                f"Extract search parameters from this query and return as JSON: \"{prompt}\"",
                system_message=f"You are a data extraction assistant. {extraction_prompt}\n\nIMPORTANT: Return ONLY valid JSON with the specified fields. Use null for missing values.",
                settings=OpenAIChatPromptExecutionSettings(
                    max_tokens=500, 
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            )
            
            response_text = str(result).strip()
            self.logger.debug(f"GET params extraction raw response: {response_text[:200]}")
            
            # Handle empty response
            if not response_text:
                self.logger.warning("Empty response from GET params extraction")
                return {"extracted_data": {}, "confidence": 0.5}
            
            # Clean markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Try parsing JSON
            try:
                extracted = json.loads(response_text)
            except json.JSONDecodeError as je:
                self.logger.warning(f"JSON parse error: {je}. Response was: {response_text[:200]}")
                return {"extracted_data": {}, "confidence": 0.5}
            
            cleaned = {k: v for k, v in extracted.items() if v is not None and v != "" and v != []}
            self.logger.info(f"Extracted GET query params: {cleaned}")
            
            return {"extracted_data": cleaned, "confidence": 0.9 if cleaned else 0.5}
            
        except Exception as e:
            self.logger.error(f"Error extracting GET params: {e}")
            return {"extracted_data": {}, "confidence": 0.5}

    def _get_extraction_prompt_for_get(self, intent: Intent) -> str:
        """Get extraction prompt for GET operations"""
        prompts = {
            Intent.INVOICE: """Extract these fields for invoice search (use null for fields not mentioned):
{
  "client_name": "Name of the client/customer" or null,
  "status": "draft" | "sent" | "paid" | "overdue" | "cancelled" or null,
  "invoice_type": "deposit" | "progress" | "final" or null,
  "date_from": "ISO date" or null,
  "date_to": "ISO date" or null,
  "search_term": "general search term" or null
}""",
            Intent.QUOTE: """Extract these fields for quote search (use null for fields not mentioned):
{
  "client_name": "Name of the client/customer" or null,
  "status": "draft" | "sent" | "accepted" | "rejected" | "expired" or null,
  "date_from": "ISO date" or null,
  "date_to": "ISO date" or null,
  "search_term": "general search term" or null
}""",
            Intent.CUSTOMER: """Extract these fields for client/customer search (use null for fields not mentioned):
{
  "name": "Name to search for" or null,
  "email": "Email to search for" or null,
  "phone": "Phone number" or null,
  "company": "Company name" or null,
  "status": "active" | "delinquent" | "archived" or null,
  "search_term": "general search term" or null
}""",
            Intent.EXPENSE: """Extract these fields for expense search (use null for fields not mentioned):
{
  "description": "Description to search" or null,
  "category": "Materials" | "Transport" | "Equipment" | "Labor" | "Insurance" | "General" | "Training" | "Marketing" | "Others" or null,
  "amount_min": number or null,
  "amount_max": number or null,
  "date_from": "ISO date" or null,
  "date_to": "ISO date" or null,
  "search_term": "general search term" or null
}""",
            Intent.JOB: """Extract these fields for job search (use null for fields not mentioned):
{
  "title": "Job title" or null,
  "client_name": "Client name" or null,
  "location": "Location" or null,
  "status": "scheduled" | "in_progress" | "completed" | "cancelled" or null,
  "date_from": "ISO date" or null,
  "date_to": "ISO date" or null,
  "search_term": "general search term" or null
}""",
            Intent.MANUAL_TASK: """Extract these fields for manual task search (use null for fields not mentioned):
{
  "title": "Task title" or null,
  "color": "Color filter" or null,
  "location": "Location" or null,
  "date_from": "ISO date" or null,
  "date_to": "ISO date" or null,
  "search_term": "general search term" or null
}"""
        }
        return prompts.get(intent, """Extract any search parameters as JSON:
{
  "search_term": "general search term" or null
}""")

    def _build_search_params(self, intent: Intent, data: Dict[str, Any]) -> Dict[str, str]:
        """Build search parameters from extracted data"""
        params = {}
        
        for field in ["search_term", "title", "description", "project_name"]:
            if data.get(field):
                params["search"] = str(data[field])
                break
        
        if data.get("client_name"):
            params["client_name"] = str(data["client_name"])
        elif data.get("name") and intent == Intent.CUSTOMER:
            params["search"] = str(data["name"])
        
        for key in ["status", "invoice_type", "category", "color", "date_from", "date_to"]:
            if data.get(key):
                params[key] = str(data[key])
        
        return params

    # ==================== RESPONSE GENERATION ====================
    
    async def _generate_final_response(
        self, intent: Intent, operation: Operation, data: Dict[str, Any], language: str, user_id: str
    ) -> Dict[str, Any]:
        """Generate final response - same structure as before for frontend compatibility"""
        
        # Handle USER_SETTINGS queries
        if intent == Intent.USER_SETTINGS:
            return await self._handle_user_settings_query(user_id, data, language)
        
        # Handle GET operations
        if operation == Operation.GET:
            return await self._handle_get_operation(intent, data, user_id, language)
        
        # Handle CREATE operations (return structured data for frontend)
        return self._generate_create_response(intent, data, user_id)
    
    async def _handle_get_operation(self, intent: Intent, data: Dict[str, Any], user_id: str, language: str = "en") -> Dict[str, Any]:
        """Handle GET operations by calling tool methods"""
        try:
            search_params = self._build_search_params(intent, data)
            self.logger.info(f"GET query with search params: {search_params}")
            
            tool_map = {
                Intent.JOB: (self.job_tools.get_jobs, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "status_filter": search_params.get("status", ""), 
                    "client_name": search_params.get("client_name", ""),
                    "date_from": search_params.get("date_from", ""),
                    "date_to": search_params.get("date_to", "")
                }),
                Intent.CUSTOMER: (self.client_tools.get_clients, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "status_filter": search_params.get("status", "")
                }),
                Intent.EXPENSE: (self.expense_tools.get_expenses, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "category_filter": search_params.get("category", ""),
                    "date_from": search_params.get("date_from", ""),
                    "date_to": search_params.get("date_to", "")
                }),
                Intent.INVOICE: (self.invoice_tools.get_invoices, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "status_filter": search_params.get("status", ""), 
                    "client_name": search_params.get("client_name", ""),
                    "invoice_type": search_params.get("invoice_type", ""),
                    "date_from": search_params.get("date_from", ""),
                    "date_to": search_params.get("date_to", "")
                }),
                Intent.QUOTE: (self.quote_tools.get_quotes, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "status_filter": search_params.get("status", ""), 
                    "client_name": search_params.get("client_name", ""),
                    "date_from": search_params.get("date_from", ""),
                    "date_to": search_params.get("date_to", "")
                }),
                Intent.MANUAL_TASK: (self.manual_task_tools.get_manual_tasks, {
                    "user_id": user_id, 
                    "search": search_params.get("search", ""), 
                    "color_filter": search_params.get("color", ""),
                    "date_from": search_params.get("date_from", ""),
                    "date_to": search_params.get("date_to", "")
                }),
            }
            
            if intent in tool_map:
                method, kwargs = tool_map[intent]
                result = await method(**kwargs)
                
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        result = {"raw": result}
                
                # Add search context to result for better messages
                result["_search_term"] = search_params.get("search", "") or search_params.get("client_name", "")
                
                # Generate human-friendly message from the data
                human_message = await self._generate_human_friendly_message(intent, result, language)
                
                return {
                    "success": True,
                    "message": human_message,
                    "intent": intent.value,
                    "operation": "get",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"success": False, "message": f"Unsupported GET for {intent.value}"}
            
        except Exception as e:
            self.logger.error(f"GET operation failed: {e}")
            return {"success": False, "message": f"Failed to retrieve: {str(e)}"}

    async def _handle_user_settings_query(self, user_id: str, data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle queries about user settings, profile, company info, rates, integrations"""
        try:
            # Fetch user settings from database
            settings_collection = get_settings_collection()
            
            # The user_settings collection uses 'user_id' (snake_case)
            user_settings = await settings_collection.find_one({"user_id": user_id})
            
            self.logger.info(f"User settings lookup for {user_id}: {'found' if user_settings else 'not found'}")
            
            if not user_settings:
                no_settings_msg = {
                    "en": "I couldn't find your profile settings. It seems your account settings haven't been configured yet. Would you like to set them up?",
                    "fr": "Je n'ai pas trouvé vos paramètres de profil. Il semble que les paramètres de votre compte n'ont pas encore été configurés. Voulez-vous les configurer?"
                }
                return {
                    "success": True,
                    "message": no_settings_msg.get(language, no_settings_msg["en"]),
                    "intent": "user_settings",
                    "operation": "get",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Clean up settings for display (remove sensitive data)
            display_settings = self._sanitize_settings_for_display(user_settings)
            
            # Get the original user query from data
            original_prompt = data.get("_original_prompt", "")
            
            # Generate human-friendly response using AI
            message = await self._generate_settings_response(display_settings, original_prompt, language)
            
            return {
                "success": True,
                "message": message,
                "intent": "user_settings",
                "operation": "get",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"User settings query failed: {e}")
            return {"success": False, "message": f"Failed to retrieve settings: {str(e)}"}
    
    def _sanitize_settings_for_display(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from settings before sending to AI"""
        sanitized = {}
        
        # Handle nested 'profile' object
        if "profile" in settings and isinstance(settings["profile"], dict):
            profile = settings["profile"]
            sanitized["profile"] = {}
            for field in ["full_name", "name", "email", "phone", "position", "language", "timezone", "avatar"]:
                if field in profile and profile[field]:
                    sanitized["profile"][field] = profile[field]
        
        # Handle nested 'company' object
        if "company" in settings and isinstance(settings["company"], dict):
            company = settings["company"]
            sanitized["company"] = {}
            
            # Safe company fields
            safe_company_fields = [
                "company_name", "name", "legal_form", "siren", "siret", "vat_number",
                "street_address", "zip_code", "city", "phone", "email", "website",
                "rcs", "ape", "liability_insurance", "default_vat_rate",
                "payment_terms", "iban", "bic"
            ]
            
            for field in safe_company_fields:
                if field in company and company[field]:
                    sanitized["company"][field] = company[field]
            
            # Handle service rates inside company
            for rates_field in ["serviceRates", "service_rates", "rates", "pricing", "hourly_rates", "hourlyRates"]:
                if rates_field in company and company[rates_field]:
                    sanitized["serviceRates"] = company[rates_field]
                    break
        
        # Handle integrations - only show status, not tokens/secrets
        if "integrations" in settings:
            sanitized["integrations"] = {}
            integrations = settings["integrations"]
            
            if isinstance(integrations, dict):
                for integration_name, integration_data in integrations.items():
                    if isinstance(integration_data, dict):
                        # Only include safe fields, exclude tokens and secrets
                        sanitized["integrations"][integration_name] = {
                            "status": integration_data.get("status", "not_configured"),
                            "auto_sync": integration_data.get("auto_sync", False),
                            "sync_direction": integration_data.get("sync_direction"),
                            "last_sync": integration_data.get("last_sync"),
                        }
                        # Add calendar_id for google_calendar (it's useful info)
                        if integration_name == "google_calendar" and integration_data.get("calendar_id"):
                            sanitized["integrations"][integration_name]["calendar_id"] = integration_data.get("calendar_id")
                        # Add error message if present
                        if integration_data.get("error_message"):
                            sanitized["integrations"][integration_name]["error_message"] = integration_data.get("error_message")
            elif isinstance(integrations, list):
                for integration_name in integrations:
                    sanitized["integrations"][integration_name] = {"status": "configured"}
        
        # Handle security settings (only non-sensitive)
        if "security" in settings and isinstance(settings["security"], dict):
            security = settings["security"]
            sanitized["security"] = {
                "two_factor_enabled": security.get("two_factor_enabled", False),
                "session_timeout": security.get("session_timeout", 30)
            }
        
        # Handle service rates at top level (fallback for different schema)
        if "serviceRates" not in sanitized:
            for rates_field in ["serviceRates", "service_rates", "rates", "pricing"]:
                if rates_field in settings and settings[rates_field]:
                    sanitized["serviceRates"] = settings[rates_field]
                    break
            
        return sanitized
    
    async def _generate_settings_response(self, settings: Dict[str, Any], original_prompt: str, language: str) -> str:
        """Generate a human-friendly response about user settings"""
        try:
            system_prompt = f"""You are a helpful business assistant. Answer the user's question about their profile/settings based on the provided data.

RULES:
- Be conversational and helpful
- Answer ONLY the specific question asked
- If asked about rates but no rates are configured, politely say "I don't see any service rates configured in your settings. You can add them in the Settings page."
- If asked about integrations, explain the connection status clearly (connected/not configured)
- If asked about company info, provide the relevant details from the data
- Keep responses concise but complete
- Use currency symbol € for amounts unless specified otherwise in settings
- If a specific piece of information is not available, say it's not configured yet
- {"Respond in French" if language == "fr" else "Respond in English"}

USER SETTINGS DATA:
{json.dumps(settings, indent=2, default=str)}"""

            user_prompt = f"User question: {original_prompt}" if original_prompt else "Summarize my profile and settings."
            
            result = await self.sk_service.kernel.invoke_prompt(
                user_prompt,
                system_message=system_prompt,
                settings=OpenAIChatPromptExecutionSettings(
                    max_tokens=400,
                    temperature=0.7
                )
            )
            
            return str(result).strip()
            
        except Exception as e:
            self.logger.error(f"Error generating settings response: {e}")
            # Fallback: return basic info
            company_name = settings.get("company", {}).get("company_name") or settings.get("companyName") or "Your company"
            return f"Here's your profile: Company: {company_name}. Use the settings page to view all details."
    
    async def _generate_human_friendly_message(self, intent: Intent, data: Dict[str, Any], language: str) -> str:
        """Generate a human-friendly summary message from retrieved data"""
        try:
            # Get the data list based on intent
            entity_key = self._get_entity_key(intent)
            items = data.get(entity_key, [])
            total = data.get("total", len(items))
            search_term = data.get("_search_term", "")
            
            # If no data found
            if total == 0 or not items:
                return self._get_no_data_message(intent, language, search_term)
            
            # Build context for LLM
            summary_context = self._build_summary_context(intent, items, total)
            
            # Generate human-friendly message using LLM
            system_prompt = f"""You are a helpful business assistant. Generate a concise, friendly summary of the retrieved data.

RULES:
- Be conversational and helpful
- Include key statistics (counts, totals, statuses)
- Mention important details like amounts, dates, names
- Keep it concise (2-4 sentences max)
- Use currency symbol € for amounts
- {"Respond in French" if language == "fr" else "Respond in English"}
- Do NOT list every item, just summarize
- Include actionable insights if relevant (e.g., "You have 2 overdue invoices that need attention")"""

            user_prompt = f"Summarize this {intent.value} data for the user:\n{summary_context}"
            
            result = await self.sk_service.kernel.invoke_prompt(
                user_prompt,
                system_message=system_prompt,
                settings=OpenAIChatPromptExecutionSettings(
                    max_tokens=300,
                    temperature=0.7
                )
            )
            
            return str(result).strip()
            
        except Exception as e:
            self.logger.error(f"Error generating human message: {e}")
            # Fallback to basic message
            return self._generate_fallback_message(intent, data, language)
    
    def _get_entity_key(self, intent: Intent) -> str:
        """Get the data key for each intent type"""
        key_map = {
            Intent.INVOICE: "invoices",
            Intent.QUOTE: "quotes",
            Intent.CUSTOMER: "clients",
            Intent.EXPENSE: "expenses",
            Intent.JOB: "jobs",
            Intent.MANUAL_TASK: "tasks"
        }
        return key_map.get(intent, "items")
    
    def _get_no_data_message(self, intent: Intent, language: str, search_term: str = "") -> str:
        """Get message when no data is found"""
        
        # If a specific name was searched, provide a personalized message
        if search_term and intent == Intent.CUSTOMER:
            if language == "fr":
                return f"Je n'ai trouvé aucun client nommé '{search_term}' dans vos dossiers. Voulez-vous ajouter {search_term} comme nouveau client?"
            return f"I couldn't find a client named '{search_term}' in your records. Would you like to add {search_term} as a new client?"
        
        if search_term:
            entity_names = {
                Intent.INVOICE: {"en": "invoices", "fr": "factures"},
                Intent.QUOTE: {"en": "quotes", "fr": "devis"},
                Intent.CUSTOMER: {"en": "clients", "fr": "clients"},
                Intent.EXPENSE: {"en": "expenses", "fr": "dépenses"},
                Intent.JOB: {"en": "jobs", "fr": "travaux"},
                Intent.MANUAL_TASK: {"en": "tasks", "fr": "tâches"}
            }
            entity = entity_names.get(intent, {}).get(language, entity_names.get(intent, {}).get("en", "records"))
            if language == "fr":
                return f"Je n'ai trouvé aucun {entity} correspondant à '{search_term}'. Voulez-vous essayer une autre recherche?"
            return f"I couldn't find any {entity} matching '{search_term}'. Would you like to try a different search?"
        
        messages = {
            Intent.INVOICE: {
                "en": "No invoices found matching your criteria. Would you like to create a new invoice?",
                "fr": "Aucune facture trouvée correspondant à vos critères. Voulez-vous créer une nouvelle facture?"
            },
            Intent.QUOTE: {
                "en": "No quotes found matching your criteria. Would you like to create a new quote?",
                "fr": "Aucun devis trouvé correspondant à vos critères. Voulez-vous créer un nouveau devis?"
            },
            Intent.CUSTOMER: {
                "en": "No clients found matching your criteria. Would you like to add a new client?",
                "fr": "Aucun client trouvé correspondant à vos critères. Voulez-vous ajouter un nouveau client?"
            },
            Intent.EXPENSE: {
                "en": "No expenses found matching your criteria. Would you like to record a new expense?",
                "fr": "Aucune dépense trouvée correspondant à vos critères. Voulez-vous enregistrer une nouvelle dépense?"
            },
            Intent.JOB: {
                "en": "No jobs found matching your criteria. Would you like to schedule a new job?",
                "fr": "Aucun travail trouvé correspondant à vos critères. Voulez-vous planifier un nouveau travail?"
            },
            Intent.MANUAL_TASK: {
                "en": "No tasks found matching your criteria. Would you like to create a new task?",
                "fr": "Aucune tâche trouvée correspondant à vos critères. Voulez-vous créer une nouvelle tâche?"
            }
        }
        return messages.get(intent, {}).get(language, messages.get(intent, {}).get("en", "No data found."))
    
    def _build_summary_context(self, intent: Intent, items: List[Dict], total: int) -> str:
        """Build summary context for LLM based on intent type"""
        
        if intent == Intent.INVOICE:
            # Summarize invoices
            status_counts = {}
            total_amount = 0
            client_names = set()
            
            for inv in items:
                status = inv.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                total_amount += inv.get("total", 0)
                if inv.get("clientName"):
                    client_names.add(inv.get("clientName"))
            
            return f"""Total invoices: {total}
Status breakdown: {json.dumps(status_counts)}
Total amount: €{total_amount:.2f}
Clients involved: {len(client_names)} ({', '.join(list(client_names)[:3])}{'...' if len(client_names) > 3 else ''})"""

        elif intent == Intent.QUOTE:
            status_counts = {}
            total_amount = 0
            client_names = set()
            
            for quote in items:
                status = quote.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                total_amount += quote.get("total", 0)
                if quote.get("clientName"):
                    client_names.add(quote.get("clientName"))
            
            return f"""Total quotes: {total}
Status breakdown: {json.dumps(status_counts)}
Total estimated value: €{total_amount:.2f}
Clients: {', '.join(list(client_names)[:3])}{'...' if len(client_names) > 3 else ''}"""

        elif intent == Intent.CUSTOMER:
            status_counts = {}
            total_balance = 0
            
            for client in items:
                status = client.get("status", "active")
                status_counts[status] = status_counts.get(status, 0) + 1
                total_balance += client.get("balance", 0)
            
            client_names = [c.get("name", "Unknown") for c in items[:5]]
            
            return f"""Total clients: {total}
Status breakdown: {json.dumps(status_counts)}
Total outstanding balance: €{total_balance:.2f}
Sample clients: {', '.join(client_names)}{'...' if total > 5 else ''}"""

        elif intent == Intent.EXPENSE:
            category_totals = {}
            total_amount = 0
            
            for exp in items:
                category = exp.get("category", "Other")
                amount = exp.get("amount", 0)
                category_totals[category] = category_totals.get(category, 0) + amount
                total_amount += amount
            
            return f"""Total expenses: {total}
Total amount: €{total_amount:.2f}
By category: {json.dumps(category_totals)}"""

        elif intent == Intent.JOB:
            status_counts = {}
            client_names = set()
            
            for job in items:
                status = job.get("status", "scheduled")
                status_counts[status] = status_counts.get(status, 0) + 1
                if job.get("clientId"):
                    client_names.add(job.get("title", "Untitled"))
            
            return f"""Total jobs: {total}
Status breakdown: {json.dumps(status_counts)}
Recent jobs: {', '.join([j.get('title', 'Untitled') for j in items[:3]])}"""

        elif intent == Intent.MANUAL_TASK:
            color_counts = {}
            
            for task in items:
                color = task.get("color", "default")
                color_counts[color] = color_counts.get(color, 0) + 1
            
            return f"""Total tasks: {total}
By color/category: {json.dumps(color_counts)}
Recent tasks: {', '.join([t.get('title', 'Untitled') for t in items[:3]])}"""

        else:
            return f"Total items: {total}"
    
    def _generate_fallback_message(self, intent: Intent, data: Dict[str, Any], language: str) -> str:
        """Generate a simple fallback message without LLM"""
        print("Generating fallback message...")
        entity_key = self._get_entity_key(intent)
        items = data.get(entity_key, [])
        total = data.get("total", len(items))
        
        entity_name = intent.value + "s" if intent.value[-1] != 's' else intent.value
        
        if language == "fr":
            return f"J'ai trouvé {total} {entity_name}. Voulez-vous plus de détails?"
        return f"I found {total} {entity_name}. Would you like more details?"
    
    def _generate_create_response(self, intent: Intent, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate CREATE response with structured data for frontend"""
        base_response = {
            "success": True,
            "message": "Operation completed successfully",
            "intent": intent.value,
            "operation": "create",
            "timestamp": datetime.now().isoformat()
        }
        
        if intent == Intent.INVOICE:
            base_response["data"] = {
                "userId": user_id,
                "clientName": data.get("customer_name") or data.get("clientName", ""),
                "clientEmail": data.get("customer_email") or data.get("clientEmail", ""),
                "title": data.get("title", ""),
                "items": data.get("items", []),
                "vatRate": data.get("vat_rate", 20),
                "total_amount": data.get("total_amount", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat()
            }
        elif intent == Intent.QUOTE:
            base_response["data"] = {
                "userId": user_id,
                "clientName": data.get("customer_name", ""),
                "clientEmail": data.get("customer_email", ""),
                "title": data.get("title", ""),
                "projectName": data.get("project_name", ""),
                "items": data.get("services") or data.get("items", []),
                "estimated_total": data.get("estimated_total", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat()
            }
        elif intent == Intent.CUSTOMER:
            base_response["data"] = {
                "name": data.get("name", ""),
                "email": data.get("email", ""),
                "phone": data.get("phone", ""),
                "address": data.get("address", ""),
                "company": data.get("company", ""),
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
        elif intent == Intent.JOB:
            base_response["data"] = {
                "title": data.get("title", ""),
                "customer_name": data.get("customer_name", ""),
                "scheduled_date": data.get("scheduled_date", ""),
                "duration": data.get("duration", 0),
                "status": "scheduled",
                "created_at": datetime.now().isoformat()
            }
        elif intent == Intent.EXPENSE:
            base_response["data"] = {
                "description": data.get("description", ""),
                "amount": data.get("amount", 0),
                "date": data.get("date", ""),
                "category": data.get("category", ""),
                "status": "recorded",
                "created_at": datetime.now().isoformat()
            }
        elif intent == Intent.MANUAL_TASK:
            base_response["data"] = {
                "task": {
                    "id": f"MTK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                    "title": data.get("title", ""),
                    "startTime": data.get("start_time", ""),
                    "endTime": data.get("end_time", ""),
                    "color": data.get("color", "#ff0000"),
                    "createdAt": datetime.now().isoformat()
                }
            }
        
        return base_response

    async def _generate_chit_chat_response(self, prompt: str, language: str) -> Dict[str, Any]:
        """Generate conversational response for chit-chat"""
        import random
        
        prompt_lower = prompt.strip().lower()
        
        responses = {
            "en": {
                "greeting": ["Hello! 👋 How can I help you today?", "Hi there! Ready to help with invoices, quotes, or clients."],
                "thanks": ["You're welcome! 😊", "Happy to help!"],
                "farewell": ["Goodbye! 👋", "See you later!"],
                "help": ["I can help with:\n• Invoices\n• Quotes\n• Clients\n• Jobs\n• Expenses\n\nJust tell me what you need!"],
                "default": ["How can I assist you today?"]
            },
            "fr": {
                "greeting": ["Bonjour ! 👋 Comment puis-je vous aider ?"],
                "thanks": ["De rien ! 😊"],
                "farewell": ["Au revoir ! 👋"],
                "help": ["Je peux vous aider avec factures, devis, clients, travaux et dépenses."],
                "default": ["Comment puis-je vous aider ?"]
            }
        }
        
        lang = responses.get(language, responses["en"])
        
        if any(w in prompt_lower for w in ["hi", "hello", "hey", "bonjour"]):
            category = "greeting"
        elif any(w in prompt_lower for w in ["thanks", "merci"]):
            category = "thanks"
        elif any(w in prompt_lower for w in ["bye", "goodbye"]):
            category = "farewell"
        elif any(w in prompt_lower for w in ["help", "what can"]):
            category = "help"
        else:
            category = "default"
        
        return {
            "success": True,
            "message": random.choice(lang.get(category, lang["default"])),
            "action": "chit_chat",
            "intent": "chit_chat"
        }

    async def _generate_general_info_response(self, prompt: str, language: str) -> Dict[str, Any]:
        """Generate informational response for general queries (calculations, estimates, info requests)"""
        try:
            system_prompt = f"""You are a helpful business assistant for a contractor/tradesperson management app.
            
The user is asking a general informational question - they want information, calculations, or estimates.
They are NOT trying to create a quote, invoice, or any record.

CONTEXT:
- You help with electrical work, plumbing, construction, and general contracting
- You know typical pricing for common services in France/Europe
- Currency is € (Euro)

GUIDELINES:
- Provide helpful, informative answers
- Give realistic price ranges for services
- Include breakdown of costs when estimating
- Mention factors that can affect the price
- Be conversational and helpful
- {"Respond in French" if language == "fr" else "Respond in English"}
- If they want to create an actual quote/invoice, suggest they say "create a quote for [client name]"

DO NOT ask for customer details - this is just an informational query."""

            user_prompt = f"User question: {prompt}"
            
            result = await self.sk_service.kernel.invoke_prompt(
                user_prompt,
                system_message=system_prompt,
                settings=OpenAIChatPromptExecutionSettings(
                    max_tokens=500,
                    temperature=0.7
                )
            )
            
            response_message = str(result).strip()
            
            return {
                "success": True,
                "message": response_message,
                "action": "general_info",
                "intent": "general_info",
                "suggestions": [
                    "Create a quote for this",
                    "Create an invoice",
                    "Show my quotes"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating general info response: {e}")
            fallback = {
                "en": "I can help with pricing estimates! For a complete electrical installation in a 60m² apartment, costs typically range from €3,000 to €6,000 depending on the scope. Would you like me to create a quote for a specific client?",
                "fr": "Je peux vous aider avec les estimations de prix ! Pour une installation électrique complète dans un appartement de 60m², les coûts varient généralement entre 3 000 € et 6 000 € selon l'étendue des travaux. Voulez-vous que je crée un devis pour un client spécifique ?"
            }
            return {
                "success": True,
                "message": fallback.get(language, fallback["en"]),
                "action": "general_info",
                "intent": "general_info"
            }

    def _create_clarification_response(self, conversation: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create response asking for intent clarification"""
        messages = {
            "en": "I'm not sure what you'd like. Would you like to create an invoice, quote, add a customer, schedule a job, or track an expense?",
            "fr": "Je ne suis pas sûr de ce que vous souhaitez. Voulez-vous créer une facture, un devis, ajouter un client, planifier un travail ou enregistrer une dépense?"
        }
        
        return {
            "success": False,
            "message": messages.get(language, messages["en"]),
            "action": "clarify_intent",
            "suggestions": ["Create an invoice", "Generate a quote", "Add customer", "Schedule a job", "Track an expense"]
        }

    def _create_missing_data_response(self, conversation: Dict[str, Any], missing_fields: List[str], language: str) -> Dict[str, Any]:
        """Create response asking for missing data"""
        field_names = {
            "en": {
                "customer_name": "customer name", "customer_email": "customer email",
                "items": "items or services", "total_amount": "total amount",
                "services": "services", "estimated_total": "estimated total",
                "name": "name", "email": "email", "phone": "phone", "address": "address",
                "title": "title", "scheduled_date": "date", "duration": "duration",
                "description": "description", "amount": "amount", "date": "date", "category": "category",
                "start_time": "start time", "end_time": "end time"
            },
            "fr": {
                "customer_name": "nom du client", "customer_email": "email du client",
                "items": "articles", "total_amount": "montant total",
                "name": "nom", "email": "email", "phone": "téléphone", "address": "adresse",
                "title": "titre", "scheduled_date": "date", "duration": "durée"
            }
        }
        
        labels = field_names.get(language, field_names["en"])
        missing_labels = [labels.get(f, f) for f in missing_fields]
        
        messages = {
            "en": f"Please provide: {', '.join(missing_labels)}",
            "fr": f"Veuillez fournir: {', '.join(missing_labels)}"
        }
        
        return {
            "success": False,
            "message": messages.get(language, messages["en"]),
            "action": "provide_missing_data",
            "missing_fields": missing_fields,
            "current_data": conversation.get("data", {})
        }

    def _create_error_response(self, error_message: str, language: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "message": f"Error: {error_message}" if language == "en" else f"Erreur: {error_message}",
            "action": "error",
            "error": error_message
        }

    # ==================== PUBLIC API ====================
    
    def reset_conversation(self, user_id: str) -> None:
        """Reset conversation (sync wrapper for closing)"""
        if user_id in self._conversation_cache:
            del self._conversation_cache[user_id]
        # Note: For full cleanup, call _close_conversation async
    
    async def reset_conversation_async(self, user_id: str) -> None:
        """Reset conversation asynchronously"""
        await self._close_conversation(user_id)
    
    def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """Get current conversation status"""
        if user_id not in self._conversation_cache:
            return {"status": "no_active_conversation"}
        
        conv = self._conversation_cache[user_id]
        return {
            "status": "active",
            "state": conv["state"].value if isinstance(conv["state"], ConversationState) else conv["state"],
            "intent": conv["intent"].value if isinstance(conv.get("intent"), Intent) else conv.get("intent"),
            "confidence": conv.get("confidence", 0.0),
            "has_data": bool(conv.get("data")),
            "field_attempts": conv.get("field_attempts", {}),
            "created_at": conv.get("created_at", "").isoformat() if hasattr(conv.get("created_at", ""), "isoformat") else str(conv.get("created_at", "")),
            "updated_at": conv.get("updated_at", "").isoformat() if hasattr(conv.get("updated_at", ""), "isoformat") else str(conv.get("updated_at", ""))
        }

    async def get_user_conversations(self, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """Get paginated list of user's conversations from database"""
        try:
            collection = get_conversations_collection()
            
            skip = (page - 1) * page_size
            
            cursor = collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(page_size + 1)
            conversations = await cursor.to_list(length=page_size + 1)
            
            has_more = len(conversations) > page_size
            if has_more:
                conversations = conversations[:page_size]
            
            total = await collection.count_documents({"user_id": user_id})
            
            return {
                "conversations": [
                    {
                        "id": str(c["_id"]),
                        "intent": c.get("intent"),
                        "operation": c.get("operation"),
                        "state": c.get("state"),
                        "message_count": len(c.get("messages", [])),
                        "created_at": c.get("created_at", "").isoformat() if hasattr(c.get("created_at", ""), "isoformat") else str(c.get("created_at", "")),
                        "updated_at": c.get("updated_at", "").isoformat() if hasattr(c.get("updated_at", ""), "isoformat") else str(c.get("updated_at", "")),
                        "is_active": c.get("is_active", False)
                    }
                    for c in conversations
                ],
                "total": total,
                "page": page,
                "page_size": page_size,
                "has_more": has_more
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching conversations: {e}")
            return {"conversations": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}

    async def get_conversation_by_id(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific conversation by ID"""
        try:
            collection = get_conversations_collection()
            
            doc = await collection.find_one({
                "_id": ObjectId(conversation_id),
                "user_id": user_id
            })
            
            if not doc:
                return None
            
            return {
                "id": str(doc["_id"]),
                "user_id": doc.get("user_id"),
                "intent": doc.get("intent"),
                "operation": doc.get("operation"),
                "state": doc.get("state"),
                "confidence": doc.get("confidence", 0.0),
                "messages": doc.get("messages", []),
                "extracted_data": doc.get("extracted_data", {}),
                "field_attempts": doc.get("field_attempts", {}),
                "language": doc.get("language", "en"),
                "created_at": doc.get("created_at", "").isoformat() if hasattr(doc.get("created_at", ""), "isoformat") else str(doc.get("created_at", "")),
                "updated_at": doc.get("updated_at", "").isoformat() if hasattr(doc.get("updated_at", ""), "isoformat") else str(doc.get("updated_at", "")),
                "is_active": doc.get("is_active", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching conversation {conversation_id}: {e}")
            return None
