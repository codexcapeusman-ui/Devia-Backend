"""
Unified AI Agent Service for Voice Processing
Handles single prompt workflow with intent detection, data extraction, and response formatting
Enhanced with audio transcription and text-to-speech capabilities
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from services.semantic_kernel_service import SemanticKernelService
from .unified_audio_service import UnifiedAudioService
from tools.client_tools import ClientTools
from tools.invoice_tools import InvoiceTools
from tools.quote_tools import QuoteTools
from tools.job_tools import JobTools
from tools.expense_tools import ExpenseTools
from tools.manual_task_tools import ManualTaskTools
from config.settings import Settings

class Intent(str, Enum):
    """Supported intents for AI agent - Order matters for priority"""
    MANUAL_TASK = "manual_task"  # HIGH PRIORITY - Check first
    CUSTOMER = "customer"
    INVOICE = "invoice"
    QUOTE = "quote"
    EXPENSE = "expense"
    JOB = "job"  # LOWER PRIORITY - Check last
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

class UnifiedAgentService:
    """
    Unified service that handles all AI agent interactions through a single endpoint
    Enhanced with voice processing capabilities for audio input and output
    Workflow: Audio -> Transcription -> Intent Detection -> Data Extraction -> Response -> TTS
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
        
        # Initialize audio service for voice processing (loads API key from settings)
        try:
            self.audio_service = UnifiedAudioService()
            self.audio_enabled = True
            self.logger.info("Voice-enabled Unified Agent Service initialized with audio capabilities")
        except Exception as e:
            self.logger.warning(f"Could not initialize audio service: {e}")
            self.audio_service = None
            self.audio_enabled = False
            self.logger.info("Unified Agent Service initialized without audio capabilities")
        
        # In-memory conversation storage (replace with database in production)
        self.conversations: Dict[str, Dict] = {}
        
        # Required fields for each intent
        self.required_fields = {
            Intent.INVOICE: [
                "customer_name", "customer_email", "items", "total_amount", "title"
            ],
            Intent.QUOTE: [
                "customer_name", "customer_email", "services", "estimated_total"
            ],
            Intent.CUSTOMER: [
                "name", "email", "phone", "address"
            ],
            Intent.JOB: [
                "title", "customer_name", "scheduled_date", "duration"
            ],
            Intent.EXPENSE: [
                "description", "amount", "date", "category"
            ],
            Intent.MANUAL_TASK: [
                "title", "start_time", "end_time"
            ]
        }
    
    async def process_agent_request(
        self, 
        prompt: str, 
        user_id: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Main entry point for unified agent processing
        
        Args:
            prompt: User's natural language prompt
            user_id: Unique identifier for the user
            language: Language preference (en/fr)
            
        Returns:
            Unified response with status, data, and next action
        """
        try:
            self.logger.info(f"Processing unified request for user {user_id}: {prompt[:100]}...")
            # Debug logging for user ID and database connection
            from database import is_connected
            self.logger.info(f"[DEBUG] User ID: {user_id} (type: {type(user_id)})")
            self.logger.info(f"[DEBUG] Database connected: {is_connected()}")
            
            # Get or create conversation state
            conversation = self._get_conversation_state(user_id)
            self.logger.info(f"Conversation state: {conversation['state']}, attempt: {conversation.get('missing_data_attempts', 0)}")
            
            # Quick user commands: reset/cancel/start over
            lower_prompt = prompt.strip().lower()
            if any(cmd in lower_prompt for cmd in ["never mind", "cancel", "start over", "reset", "stop"]):
                # Reset conversation and ask for clarification
                self.reset_conversation(user_id)
                conversation = self._get_conversation_state(user_id)
                return {
                    "success": True,
                    "message": "Conversation reset. How can I help you now?",
                    "action": "reset"
                }

            # Step 1: Intent Detection (if not already detected)
            if conversation["state"] == ConversationState.INTENT_DETECTION:
                intent, operation, confidence = await self._detect_intent(prompt, language)
                conversation["intent"] = intent
                conversation["operation"] = operation
                conversation["confidence"] = confidence
                conversation["data"] = {}
                
                self.logger.info(f"Intent detection result: intent={intent}, operation={operation}, confidence={confidence}")
                
                # Special handling for "get all" queries - skip data extraction entirely
                if operation == Operation.GET and self._is_get_all_query(prompt):
                    self.logger.info(f"Detected 'get all' query for {intent.value}, skipping to response generation")
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
                elif intent == Intent.UNKNOWN or confidence < 0.1:
                    self.logger.warning(f"Intent unclear or low confidence: {intent}, {confidence}")
                    return self._create_clarification_response(conversation, language)
                else:
                    conversation["state"] = ConversationState.DATA_EXTRACTION
            
            else:
                # If we're mid-conversation, check whether the user has changed their intent.
                # Only attempt re-detection if we're not in data extraction/completion states (to avoid
                # misinterpreting missing data inputs as new intents)
                if conversation["state"] not in [ConversationState.DATA_EXTRACTION, ConversationState.DATA_COMPLETION]:
                    try:
                        new_intent, new_operation, new_confidence = await self._detect_intent(prompt, language)
                        
                        # Special handling for "get all" queries - always switch to this flow
                        if new_operation == Operation.GET and self._is_get_all_query(prompt):
                            self.logger.info(f"Detected 'get all' query mid-conversation for {new_intent.value}, switching to direct response")
                            conversation["intent"] = new_intent
                            conversation["operation"] = new_operation
                            conversation["confidence"] = new_confidence
                            conversation["data"] = {}
                            conversation["state"] = ConversationState.RESPONSE_GENERATION
                        
                        # If the new intent/operation is different and confidence is reasonably high, switch flows
                        elif new_intent != conversation.get("intent") and new_confidence >= 0.6:
                            self.logger.info(f"User changed intent mid-flow from {conversation.get('intent')} to {new_intent} (conf={new_confidence})")
                            conversation["intent"] = new_intent
                            conversation["operation"] = new_operation
                            conversation["confidence"] = new_confidence
                            # Reset collected data but keep it optional to be merged later if fields overlap
                            conversation["data"] = {}
                            conversation["missing_data_attempts"] = 0
                            conversation["state"] = ConversationState.DATA_EXTRACTION
                    except Exception:
                        # If intent re-detection fails, continue with existing flow
                        self.logger.debug("Intent re-detection failed while mid-conversation; continuing existing flow")
            
            # Step 2: Data Extraction (initial or additional data)
            if conversation["state"] in [ConversationState.DATA_EXTRACTION, ConversationState.DATA_COMPLETION]:
                extracted_data = await self._extract_data(
                    prompt, conversation["intent"], conversation.get("operation", Operation.UNKNOWN), language
                )
                
                # Merge data intelligently - preserve existing valid data
                self._merge_conversation_data(conversation["data"], extracted_data)
                conversation["state"] = ConversationState.DATA_COMPLETION
            
            # Step 3: Check for Missing Data
            if conversation["state"] == ConversationState.DATA_COMPLETION:
                missing_fields = self._check_missing_data(
                    conversation["intent"], conversation.get("operation", Operation.UNKNOWN), conversation["data"]
                )
                
                if missing_fields:
                    # Check if we've already asked for missing data 2 times
                    if conversation.get("missing_data_attempts", 0) >= 2:
                        self.logger.info(f"Max attempts reached, filling missing fields with N/A: {missing_fields}")
                        # Fill missing fields with "N/A" and proceed
                        for field in missing_fields:
                            if field == "total_amount":
                                conversation["data"][field] = 0.0
                            elif field == "items":
                                conversation["data"][field] = []
                            else:
                                conversation["data"][field] = "N/A"
                        conversation["state"] = ConversationState.RESPONSE_GENERATION
                    else:
                        # Increment attempt counter and ask for missing data
                        conversation["missing_data_attempts"] = conversation.get("missing_data_attempts", 0) + 1
                        return self._create_missing_data_response(
                            conversation, missing_fields, language
                        )
                else:
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
            
            # Step 4: Generate Final Response
            if conversation["state"] == ConversationState.RESPONSE_GENERATION:
                response = await self._generate_final_response(
                    conversation["intent"], conversation.get("operation", Operation.UNKNOWN), conversation["data"], language, user_id
                )
                conversation["state"] = ConversationState.COMPLETED
                
                # Reset conversation after successful response
                if response.get("success", False):
                    self.reset_conversation(user_id)
                
                return response
            
            # Fallback
            return self._create_error_response("Invalid conversation state", language)
            
        except Exception as e:
            self.logger.error(f"Error processing agent request: {e}")
            return self._create_error_response(str(e), language)
    
    async def _detect_intent(self, prompt: str, language: str) -> Tuple[Intent, Operation, float]:
        """
        Detect user intent from the prompt using AI
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        intent_prompt = f"""
        Analyze this user prompt and determine their intent and operation. Respond with JSON only.
        
        User prompt: "{prompt}"
        
        INTENT DETECTION PRIORITY ORDER (Check in this order):
        1. MANUAL_TASK (Highest Priority)
        2. CUSTOMER
        3. INVOICE 
        4. QUOTE
        5. EXPENSE
        6. JOB (Lowest Priority - only if no other intent matches)
        
        OPERATIONS:
        - get: Viewing/retrieving existing data (show, list, get, display, find, see, view, retrieve, "all my")
        - create: Creating new data (create, add, schedule, book, make, generate, new)
        - update: Modifying existing data (update, change, modify, edit, adjust)
        - delete: Removing data (delete, remove, cancel, eliminate)
        
        MANUAL_TASK INDICATORS (Check FIRST - Highest Priority):
        ✅ Color words: red, blue, green, yellow, orange, purple, pink, black, white, gray
        ✅ Task language: "task", "manual task", "planning", "reminder", "internal"
        ✅ Personal/team context: "my task", "remind me", "team meeting", "internal planning"
        ✅ Non-client work: No specific client names mentioned
        ✅ Planning context: "work task", "maintenance task", "planning task"
        ✅ Time-only scheduling: Just times without client context
        
        CUSTOMER INDICATORS:
        ✅ Client management: "client", "customer", "contact", customer data"
        ✅ Customer operations: "add client", "show customers", "client information"
        
        INVOICE INDICATORS:
        ✅ Billing: "invoice", "bill", "payment", "charge", "billing"
        
        QUOTE INDICATORS:
        ✅ Estimates: "quote", "estimate", "proposal", "pricing"
        
        EXPENSE INDICATORS:
        ✅ Costs: "expense", "receipt", "cost", "spending", "financial tracking"
        
        JOB INDICATORS (Check LAST - Only if no manual_task match):
        ✅ Client-specific work: "for [ClientName]", "with [ClientName]", specific company names
        ✅ Billable services: "installation for client", "service appointment", "customer meeting"
        ✅ Professional appointments: "appointment with client", "customer service call"
        
        CRITICAL RULES:
        🔴 IF color mentioned → ALWAYS manual_task (red task, blue work, green reminder)
        🔴 IF "task" + no client name → ALWAYS manual_task  
        🔴 IF "planning" or "reminder" → ALWAYS manual_task
        🔴 IF client name mentioned → Then consider job
        🔴 IF just time/date without client → manual_task
        
        EXAMPLES - MANUAL_TASK (High Priority):
        ❌ "create a red placo work task for tomorrow 9-5" → manual_task, create
        ❌ "add yellow planning task for Monday" → manual_task, create
        ❌ "make blue reminder task" → manual_task, create
        ❌ "schedule green work task" → manual_task, create
        ❌ "create maintenance task for tomorrow" → manual_task, create
        ❌ "add placo work task" → manual_task, create
        ❌ "show my manual tasks" → manual_task, get
        ❌ "list red tasks" → manual_task, get
        
        EXAMPLES - JOB (Low Priority):
        ✅ "schedule website maintenance for ABC Corp" → job, create
        ✅ "book appointment with John Smith" → job, create
        ✅ "create meeting for client XYZ" → job, create
        ✅ "show jobs for ABC Corp" → job, get
        
        EXAMPLES - OTHER:
        ✅ "show my clients" → customer, get
        ✅ "create invoice" → invoice, create
        ✅ "add expense" → expense, create
        ✅ "generate quote" → quote, create
        
        Response format:
        {{
            "intent": "intent_name",
            "operation": "operation_name", 
            "confidence": 0.95,
            "reasoning": "Brief explanation focusing on key indicators found"
        }}
        """
        
        try:
            # Use semantic kernel for intent detection
            result = await self.sk_service.process_invoice_request(
                prompt=intent_prompt,
                context={"task": "intent_detection"},
                language=language
            )
            
            # Parse AI response
            if result.get("success") and result.get("data"):
                ai_response = result["data"]
                self.logger.info(f"Raw AI response: {ai_response}")
                
                # Handle case where AI response is wrapped in "response" key
                if isinstance(ai_response, dict) and "response" in ai_response:
                    ai_response = ai_response["response"]
                    
                # Try to parse as JSON if it's a string
                if isinstance(ai_response, str):
                    # Strip markdown code blocks if present
                    ai_response = ai_response.strip()
                    if ai_response.startswith("```json"):
                        ai_response = ai_response[7:]  # Remove ```json
                    if ai_response.startswith("```"):
                        ai_response = ai_response[3:]   # Remove ```
                    if ai_response.endswith("```"):
                        ai_response = ai_response[:-3]  # Remove ```
                    ai_response = ai_response.strip()
                    
                    try:
                        ai_response = json.loads(ai_response)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse AI response as JSON: {ai_response}")
                        pass
                
                self.logger.info(f"Processed AI response: {ai_response}")
                
                # Extract intent and operation data
                if isinstance(ai_response, dict):
                    intent_str = ai_response.get("intent", "unknown")
                    operation_str = ai_response.get("operation", "unknown")
                    confidence = ai_response.get("confidence", 0.0)
                else:
                    intent_str = "unknown"
                    operation_str = "unknown"
                    confidence = 0.0
                
                # Map string to enum (case-insensitive)
                try:
                    intent = Intent(intent_str.lower())
                except ValueError:
                    intent = Intent.UNKNOWN
                    confidence = 0.0
                
                try:
                    operation = Operation(operation_str.lower())
                except ValueError:
                    operation = Operation.UNKNOWN
                
                return intent, operation, confidence
            
        except Exception as e:
            self.logger.error(f"Intent detection failed: {e}")
        
        # Fallback: Simple pattern matching for common GET operations
        prompt_lower = prompt.lower()
        
        # Check for common GET patterns
        if any(word in prompt_lower for word in ["show", "list", "get", "display", "see", "view", "all my", "my clients", "my invoices", "my jobs", "my expenses", "my quotes"]):
            if any(word in prompt_lower for word in ["client", "customer", "contact"]):
                return Intent.CUSTOMER, Operation.GET, 0.8
            elif any(word in prompt_lower for word in ["invoice", "bill", "billing"]):
                return Intent.INVOICE, Operation.GET, 0.8
            elif any(word in prompt_lower for word in ["job", "appointment", "meeting", "schedule"]):
                return Intent.JOB, Operation.GET, 0.8
            elif any(word in prompt_lower for word in ["expense", "cost", "spending"]):
                return Intent.EXPENSE, Operation.GET, 0.8
            elif any(word in prompt_lower for word in ["quote", "estimate", "proposal"]):
                return Intent.QUOTE, Operation.GET, 0.8
        
        return Intent.UNKNOWN, Operation.UNKNOWN, 0.0
    
    async def _extract_data(
        self, 
        prompt: str, 
        intent: Intent, 
        operation: Operation,
        language: str
    ) -> Dict[str, Any]:
        """
        Extract relevant data fields based on the detected intent
        """
        # Check if this is a specific ID query
        if self._is_specific_id_query(prompt, intent):
            return self._extract_id_from_prompt(prompt, intent)
        
        # For GET operations, we don't need to extract data beyond IDs
        if operation == Operation.GET:
            return {
                "extracted_data": {},
                "confidence": 1.0,
                "missing_fields": []
            }
        
        extraction_prompts = {
            Intent.INVOICE: """
            Extract invoice data from this prompt. Return JSON with these fields:
            - customer_name: Customer/client name
            - customer_email: Email address
            - customer_phone: Phone number (optional)
            - customer_address: Address (optional)
            - items: Array of {description, quantity, unit_price, total}
            - subtotal: Subtotal amount
            - tax_rate: Tax rate (default 0.20)
            - tax_amount: Tax amount
            - total_amount: Final total
            - invoice_date: Date (ISO format, default today)
            - due_date: Due date (ISO format, default +30 days)
            """,
            Intent.QUOTE: """
            Extract comprehensive quote data from this prompt. Return JSON with these fields:
            
            CLIENT INFORMATION:
            - customer_name: Customer/client full name
            - customer_email: Email address
            - customer_phone: Phone number (optional)
            - customer_company_type: "COMPANY" or "INDIVIDUAL" (based on context)
            
            PROJECT DETAILS:
            - title: Quote title/subject
            - project_name: Project or job name (mandatory)
            - project_street_address: Street address component (optional)
            - project_zip_code: ZIP/postal code (optional)
            - project_city: City name (optional)
            
            QUOTE DETAILS:
            - services: Array of {description, estimated_hours, hourly_rate, total, type}
            - subtotal: Subtotal amount before discounts
            
            DISCOUNT INFORMATION:
            - discount: Discount amount or percentage value
            - discount_type: "FIXED" (euro amount) or "PERCENTAGE"
            
            DOWN PAYMENT INFORMATION:
            - down_payment: Down payment amount or percentage value
            - down_payment_type: "FIXED" (euro amount) or "PERCENTAGE"
            
            TAX AND TOTALS:
            - vat_rate: VAT rate (default 0.20 = 20%)
            - estimated_total: Final estimated total after all calculations
            
            DATES:
            - valid_until: Quote validity date (ISO format, default +30 days)
            
            NOTES (categorize appropriately):
            - internal_notes: Internal notes (not visible to client)
            - public_notes: Notes visible on PDF/to client
            
            SIGNATURES (if mentioned):
            - contractor_signature: Contractor signature reference
            - client_signature: Client signature reference
            
            Extract discount type:
            - PERCENTAGE: if "%" symbol present or percentage mentioned
            - FIXED: if euro/currency amount specified
            
            Determine company type from context:
            - INDIVIDUAL: "person", "individual", "freelancer", "self-employed"
            - COMPANY: "company", "business", "corp", "ltd", "organization"
            """,
            Intent.CUSTOMER: """
            Extract customer data from this prompt. Return JSON with these fields:
            - name: Full name
            - email: Email address
            - phone: Phone number
            - address: Full address
            - company: Company name (optional)
            - notes: Additional notes (optional)
            - language_preference: Language preference (en/fr)
            """,
            Intent.JOB: """
            Extract job data from this prompt. Return JSON with these fields:
            - title: Job title/description
            - customer_name: Customer name
            - customer_email: Customer email (optional)
            - scheduled_date: Scheduled date (ISO format)
            - scheduled_time: Scheduled time (HH:MM format)
            - duration: Duration in hours
            - location: Job location (optional)
            - notes: Additional notes (optional)
            """,
            Intent.EXPENSE: """
            Extract expense data from this prompt. Return JSON with these fields:
            - description: Expense description
            - amount: Amount spent
            - date: Expense date (ISO format)
            - category: Expense category
            - vendor: Vendor/supplier name (optional)
            - payment_method: Payment method (optional)
            - receipt_number: Receipt number (optional)
            - vat_rate: VAT rate (default 0.20)
            - vat_amount: VAT amount
            """,
            Intent.MANUAL_TASK: """
            Extract manual task data from this prompt. Return JSON with these fields:
            - title: Task title/description
            - start_time: Start date and time (ISO format)
            - end_time: End date and time (ISO format)
            - color: Color (hex code, optional, default #ff0000)
            - client_id: Associated client ID (optional)
            - assigned_to: Assigned worker/team (optional)
            - location: Task location/address (optional)
            - notes: Task notes/details (optional)
            - is_all_day: Whether this is an all-day task (boolean, optional)
            """
        }
        
        extract_prompt = extraction_prompts.get(intent, "")
        if not extract_prompt:
            return {}
        
        full_prompt = f"{extract_prompt}\n\nUser prompt: \"{prompt}\"\n\nReturn only valid JSON:"
        
        try:
            # Use appropriate tool based on intent
            if intent == Intent.INVOICE:
                result = await self.sk_service.process_invoice_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            elif intent == Intent.QUOTE:
                result = await self.sk_service.process_quote_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            elif intent == Intent.CUSTOMER:
                result = await self.sk_service.process_customer_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            elif intent == Intent.JOB:
                result = await self.sk_service.process_job_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            elif intent == Intent.EXPENSE:
                result = await self.sk_service.process_expense_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            elif intent == Intent.MANUAL_TASK:
                result = await self.sk_service.process_manual_task_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language
                )
            else:
                return {}
            
            if result.get("success") and result.get("data"):
                ai_response = result["data"]
                
                # Handle case where AI response is wrapped in "response" key
                if isinstance(ai_response, dict) and "response" in ai_response:
                    ai_response = ai_response["response"]
                
                # Try to parse as JSON if it's a string
                if isinstance(ai_response, str):
                    # Strip markdown code blocks if present
                    ai_response = ai_response.strip()
                    if ai_response.startswith("```json"):
                        ai_response = ai_response[7:]  # Remove ```json
                    if ai_response.startswith("```"):
                        ai_response = ai_response[3:]   # Remove ```
                    if ai_response.endswith("```"):
                        ai_response = ai_response[:-3]  # Remove ```
                    ai_response = ai_response.strip()
                    
                    try:
                        ai_response = json.loads(ai_response)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse extraction response as JSON: {ai_response}")
                        return {}
                
                # Return the parsed data
                if isinstance(ai_response, dict):
                    return ai_response
                else:
                    return result["data"]
            
        except Exception as e:
            self.logger.error(f"Data extraction failed for {intent}: {e}")
        
        return {}
    
    def _merge_conversation_data(self, existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> None:
        """
        Intelligently merge new extracted data with existing conversation data.
        Only updates fields that have meaningful values in new_data.
        Special handling for 'extracted_data' key - merge its contents to top level.
        """
        for key, value in new_data.items():
            if key == "extracted_data" and isinstance(value, dict):
                # Merge extracted_data contents to top level
                for sub_key, sub_value in value.items():
                    if self._is_meaningful_value(sub_value):
                        existing_data[sub_key] = sub_value
            elif self._is_meaningful_value(value):
                existing_data[key] = value
    
    def _is_meaningful_value(self, value: Any) -> bool:
        """
        Check if a value is meaningful (not empty, None, or placeholder)
        """
        if value is None:
            return False
        if isinstance(value, str):
            # Empty string, whitespace only, or common placeholders
            if not value.strip() or value.strip().lower() in ["", "n/a", "na", "null", "none", "undefined"]:
                return False
        elif isinstance(value, (list, dict)):
            # Empty collections
            if not value:
                return False
        elif isinstance(value, (int, float)):
            # Zero values might be meaningful for some fields, but not for amounts
            if value == 0.0 and isinstance(value, float):
                return False
        
        return True
    
    def _check_missing_data(self, intent: Intent, operation: Operation, data: Dict[str, Any]) -> List[str]:
        """
        Check which required fields are missing for the given intent
        For GET operations, only check for missing data if it's a specific ID query
        """
        # For GET operations, only check for missing data if it's a specific ID query
        if operation == Operation.GET:
            # Check if this is a specific ID query that requires an ID
            if data.get("query_type") == "specific_id" and not data.get("id"):
                return ["id"]
            # For general "get all" queries, no data is missing
            return []
        
        required = self.required_fields.get(intent, [])
        missing = []
        
        for field in required:
            if field not in data or not data[field]:
                missing.append(field)
        
        return missing
    
    async def _generate_final_response(
        self, 
        intent: Intent, 
        operation: Operation,
        data: Dict[str, Any], 
        language: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Generate the final response based on intent, operation and extracted data
        For GET operations, queries the database using appropriate tools
        For CREATE operations, returns responses that match manual endpoint structure
        
        Args:
            intent: The detected intent
            operation: The operation to perform
            data: Extracted data from the prompt
            language: Language preference
            user_id: User ID for security filtering
        """
        
        # Handle GET operations
        if operation == Operation.GET:
            try:
                # Check if this is a specific ID query
                if data.get("query_type") == "specific_id" and data.get("id"):
                    # Handle specific ID queries using tools
                    if intent == Intent.JOB:
                        result = await self.job_tools.get_job_by_id(data["id"], user_id)
                    elif intent == Intent.CUSTOMER:
                        result = await self.client_tools.get_client_by_id(data["id"], user_id)
                    elif intent == Intent.EXPENSE:
                        result = await self.expense_tools.get_expense_by_id(data["id"], user_id)
                    elif intent == Intent.INVOICE:
                        result = await self.invoice_tools.get_invoice_by_id(data["id"], user_id)
                    elif intent == Intent.QUOTE:
                        result = await self.quote_tools.get_quote_by_id(data["id"], user_id)
                    else:
                        return {
                            "success": False,
                            "message": f"Unsupported GET by ID operation for intent: {intent.value}",
                            "data": None
                        }
                    
                    # Parse the JSON result from tools
                    if isinstance(result, str):
                        result = json.loads(result)
                    
                    # Extract the actual data
                    if "client" in result:
                        result = result["client"]
                    elif "invoice" in result:
                        result = result["invoice"]
                    elif "quote" in result:
                        result = result["quote"]
                    elif "job" in result:
                        result = result["job"]
                    elif "expense" in result:
                        result = result["expense"]
                    elif "task" in result:
                        result = result["task"]
                        
                else:
                    # Handle general list queries using tools
                    if intent == Intent.JOB:
                        result = await self.job_tools.get_jobs(user_id=user_id, skip=0, limit=50)
                    elif intent == Intent.CUSTOMER:
                        result = await self.client_tools.get_clients(user_id=user_id, skip=0, limit=50)
                    elif intent == Intent.EXPENSE:
                        result = await self.expense_tools.get_expenses(user_id=user_id, skip=0, limit=50)
                    elif intent == Intent.INVOICE:
                        result = await self.invoice_tools.get_invoices(user_id=user_id, skip=0, limit=50)
                    elif intent == Intent.QUOTE:
                        result = await self.quote_tools.get_quotes(user_id=user_id, skip=0, limit=50)
                    elif intent == Intent.MANUAL_TASK:
                        result = await self.manual_task_tools.get_manual_tasks(user_id=user_id, skip=0, limit=50)
                    else:
                        return {
                            "success": False,
                            "message": f"Unsupported GET operation for intent: {intent.value}",
                            "data": None
                        }
                    
                    # Parse the JSON result from tools
                    if isinstance(result, str):
                        result = json.loads(result)
                    
                    # Extract the list data
                    if "clients" in result:
                        result = result["clients"]
                    elif "invoices" in result:
                        result = result["invoices"]
                    elif "quotes" in result:
                        result = result["quotes"]
                    elif "jobs" in result:
                        result = result["jobs"]
                    elif "expenses" in result:
                        result = result["expenses"]
                    elif "tasks" in result:
                        result = result["tasks"]
                    else:
                        result = []
                        
                return {
                    "success": True,
                    "message": f"Retrieved {intent.value}s successfully",
                    "data": result,
                    "intent": intent.value,
                    "operation": operation.value,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"GET operation failed: {e}")
                return {
                    "success": False,
                    "message": f"Failed to retrieve {intent.value}s: {str(e)}",
                    "data": None
                }
        
        base_response = {
            "success": True,
            "message": "Operation completed successfully",
            "intent": intent.value,
            "operation": operation.value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Intent-specific response formatting (dummy data for now)
        if intent == Intent.INVOICE:
            base_response["data"] = {
                "userId": user_id,
                "clientId": data.get("client_id") or data.get("clientId"),
                "clientName": data.get("customer_name") or data.get("clientName", ""),
                "clientEmail": data.get("customer_email") or data.get("clientEmail", ""),
                "clientCompanyType": data.get("customer_company_type") or data.get("clientCompanyType", "company"),
                "quoteId": data.get("quote_id") or data.get("quoteId", ""),
                "number": data.get("invoice_number") or data.get("number", "INV-001"),
                "title": data.get("title", ""),
                "projectName": data.get("project_name") or data.get("projectName", ""),
                "projectAddress": data.get("project_address") or data.get("projectAddress", ""),
                "projectStreetAddress": data.get("project_street_address") or data.get("projectStreetAddress", ""),
                "projectZipCode": data.get("project_zip_code") or data.get("projectZipCode", ""),
                "projectCity": data.get("project_city") or data.get("projectCity", ""),
                "invoiceType": data.get("invoice_type") or data.get("invoiceType", "final"),
                "items": data.get("items", []),
                "discount": data.get("discount", 0),
                "discountType": data.get("discount_type") or data.get("discountType", "fixed"),
                "downPayment": data.get("down_payment") or data.get("downPayment", 0),
                "downPaymentType": data.get("down_payment_type") or data.get("downPaymentType", "percentage"),
                "vatRate": data.get("vat_rate") or data.get("vatRate", 20),
                "dueDate": data.get("due_date") or data.get("dueDate", datetime.now().isoformat()),
                "eInvoiceStatus": data.get("e_invoice_status") or data.get("eInvoiceStatus", "pending"),
                "notes": data.get("notes", ""),
                "internalNotes": data.get("internal_notes") or data.get("internalNotes", ""),
                "publicNotes": data.get("public_notes") or data.get("publicNotes", ""),
                "contractorSignature": data.get("contractor_signature") or data.get("contractorSignature", ""),
                "clientSignature": data.get("client_signature") or data.get("clientSignature", ""),
                "subtotal": data.get("subtotal", 0),
                "tax_amount": data.get("tax_amount", 0),
                "total_amount": data.get("total_amount", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat()
            }
        
        elif intent == Intent.QUOTE:
            base_response["data"] = {
                "userId": user_id,
                "clientId": data.get("client_id") or data.get("clientId"),
                "clientName": data.get("customer_name") or data.get("clientName", ""),
                "clientEmail": data.get("customer_email") or data.get("clientEmail", ""),
                "clientCompanyType": data.get("customer_company_type") or data.get("clientCompanyType", "company"),
                "number": data.get("quote_number") or data.get("number", "QUO-001"),
                "title": data.get("title", ""),
                "projectName": data.get("project_name") or data.get("projectName", ""),
                "projectStreetAddress": data.get("project_street_address") or data.get("projectStreetAddress", ""),
                "projectZipCode": data.get("project_zip_code") or data.get("projectZipCode", ""),
                "projectCity": data.get("project_city") or data.get("projectCity", ""),
                "items": data.get("items", []),
                "discount": data.get("discount", 0),
                "discountType": data.get("discount_type") or data.get("discountType", "fixed"),
                "downPayment": data.get("down_payment") or data.get("downPayment", 0),
                "downPaymentType": data.get("down_payment_type") or data.get("downPaymentType", "percentage"),
                "vatRate": data.get("vat_rate") or data.get("vatRate", 20),
                "validUntil": data.get("valid_until") or data.get("validUntil") or (datetime.now() + datetime.timedelta(days=30)).isoformat(),
                "internalNotes": data.get("internal_notes") or data.get("internalNotes", ""),
                "publicNotes": data.get("public_notes") or data.get("publicNotes", ""),
                "contractorSignature": data.get("contractor_signature") or data.get("contractorSignature", ""),
                "clientSignature": data.get("client_signature") or data.get("clientSignature", ""),
                "subtotal": data.get("subtotal", 0),
                "estimated_total": data.get("estimated_total", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat()
            }
        
        elif intent == Intent.CUSTOMER:
            base_response["data"] = {
                "customer_id": "CUST-2025-001",
                "name": data.get("name", ""),
                "email": data.get("email", ""),
                "phone": data.get("phone", ""),
                "address": data.get("address", ""),
                "company": data.get("company", ""),
                "notes": data.get("notes", ""),
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
        
        elif intent == Intent.JOB:
            base_response["data"] = {
                "job_id": "JOB-2025-001",
                "title": data.get("title", ""),
                "customer_name": data.get("customer_name", ""),
                "scheduled_date": data.get("scheduled_date", ""),
                "scheduled_time": data.get("scheduled_time", ""),
                "duration": data.get("duration", 0),
                "location": data.get("location", ""),
                "status": "scheduled",
                "created_at": datetime.now().isoformat()
            }
        
        elif intent == Intent.EXPENSE:
            base_response["data"] = {
                "expense_id": "EXP-2025-001",
                "description": data.get("description", ""),
                "amount": data.get("amount", 0),
                "date": data.get("date", ""),
                "category": data.get("category", ""),
                "vendor": data.get("vendor", ""),
                "vat_amount": data.get("vat_amount", 0),
                "status": "recorded",
                "created_at": datetime.now().isoformat()
            }
        
        elif intent == Intent.MANUAL_TASK:
            base_response["data"] = {
                "task": {
                    "id": f"MTK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                    "title": data.get("title", ""),
                    "userId": data.get("userId", ""),
                    "clientId": data.get("clientId"),
                    "startTime": data.get("start_time", ""),
                    "endTime": data.get("end_time", ""),
                    "color": data.get("color", "#ff0000"),
                    "notes": data.get("notes", ""),
                    "assignedTo": data.get("assignedTo"),
                    "location": data.get("location", ""),
                    "isAllDay": data.get("isAllDay", False),
                    "createdAt": datetime.now().isoformat(),
                    "updatedAt": datetime.now().isoformat()
                }
            }
        
        return base_response
    
    def _get_conversation_state(self, user_id: str) -> Dict[str, Any]:
        """
        Get or create conversation state for user
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "state": ConversationState.INTENT_DETECTION,
                "intent": None,
                "confidence": 0.0,
                "data": {},
                "missing_data_attempts": 0,  # Track how many times we asked for missing data
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        
        self.conversations[user_id]["updated_at"] = datetime.now().isoformat()
        return self.conversations[user_id]
    
    def _create_clarification_response(
        self, 
        conversation: Dict[str, Any], 
        language: str
    ) -> Dict[str, Any]:
        """
        Create response asking for clarification of intent
        """
        messages = {
            "en": "I'm not sure what you'd like me to help you with. Could you please clarify if you want to create an invoice, quote, add customer data, schedule a job, or track an expense?",
            "fr": "Je ne suis pas sûr de ce que vous aimeriez que je vous aide. Pourriez-vous préciser si vous souhaitez créer une facture, un devis, ajouter des données client, planifier un travail ou suivre une dépense?"
        }
        
        return {
            "success": False,
            "message": messages.get(language, messages["en"]),
            "action": "clarify_intent",
            "suggestions": [
                "Create an invoice",
                "Generate a quote", 
                "Add customer information",
                "Schedule a job",
                "Track an expense"
            ]
        }
    
    def _create_missing_data_response(
        self, 
        conversation: Dict[str, Any], 
        missing_fields: List[str], 
        language: str
    ) -> Dict[str, Any]:
        """
        Create response asking for missing required data
        """
        field_names = {
            "en": {
                "customer_name": "customer name",
                "customer_email": "customer email",
                "items": "items or services",
                "total_amount": "total amount",
                "services": "services or items",
                "estimated_total": "estimated total",
                "name": "name",
                "email": "email address",
                "phone": "phone number",
                "address": "address",
                "title": "job title",
                "scheduled_date": "scheduled date",
                "duration": "duration",
                "description": "description",
                "amount": "amount",
                "date": "date",
                "category": "category"
            },
            "fr": {
                "customer_name": "nom du client",
                "customer_email": "email du client",
                "items": "articles ou services",
                "total_amount": "montant total",
                "services": "services ou articles",
                "estimated_total": "total estimé",
                "name": "nom",
                "email": "adresse email",
                "phone": "numéro de téléphone",
                "address": "adresse",
                "title": "titre du travail",
                "scheduled_date": "date prévue",
                "duration": "durée",
                "description": "description",
                "amount": "montant",
                "date": "date",
                "category": "catégorie"
            }
        }
        
        field_labels = field_names.get(language, field_names["en"])
        missing_labels = [field_labels.get(field, field) for field in missing_fields]
        
        messages = {
            "en": f"I need some additional information to complete this. Please provide: {', '.join(missing_labels)}",
            "fr": f"J'ai besoin d'informations supplémentaires pour compléter ceci. Veuillez fournir: {', '.join(missing_labels)}"
        }
        
        return {
            "success": False,
            "message": messages.get(language, messages["en"]),
            "action": "provide_missing_data",
            "missing_fields": missing_fields,
            "current_data": conversation["data"]
        }
    
    def _create_error_response(self, error_message: str, language: str) -> Dict[str, Any]:
        """
        Create error response
        """
        messages = {
            "en": f"I encountered an error: {error_message}",
            "fr": f"J'ai rencontré une erreur: {error_message}"
        }
        
        return {
            "success": False,
            "message": messages.get(language, messages["en"]),
            "action": "error",
            "error": error_message
        }
    
    def reset_conversation(self, user_id: str) -> None:
        """
        Reset conversation state for a user
        """
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get current conversation status for a user
        """
        if user_id not in self.conversations:
            return {"status": "no_active_conversation"}
        
        conversation = self.conversations[user_id]
        return {
            "status": "active",
            "state": conversation["state"],
            "intent": conversation["intent"],
            "operation": conversation.get("operation"),
            "confidence": conversation["confidence"],
            "has_data": bool(conversation["data"]),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"]
        }
    
    def generate_human_friendly_response(self, structured_response: Dict[str, Any]) -> str:
        """
        Generate human-friendly response from structured agent response
        Specifically designed for voice/audio output with natural language
        
        Args:
            structured_response: The structured response from process_agent_request
            
        Returns:
            Natural language response suitable for text-to-speech
        """
        try:
            if not structured_response.get("success", False):
                error_msg = structured_response.get("message", "I encountered an error")
                return f"I'm sorry, but {error_msg.lower()}. Please try again with more details."
            
            data = structured_response.get("data", {})
            intent = data.get("intent")
            
            # Handle different response types based on intent
            if intent == "invoice":
                return self._generate_invoice_voice_response(data)
            elif intent == "quote":
                return self._generate_quote_voice_response(data)
            elif intent == "customer":
                return self._generate_customer_voice_response(data)
            elif intent == "job":
                return self._generate_job_voice_response(data)
            elif intent == "expense":
                return self._generate_expense_voice_response(data)
            else:
                # Generic response for unknown intents
                action = structured_response.get("action", "processed your request")
                return f"I've {action} successfully. The information has been saved and is ready for your review."
                
        except Exception as e:
            self.logger.error(f"Error generating human response: {e}")
            return "I've processed your request, but I'm having trouble generating a clear summary. Please check the detailed response for more information."
    
    def _generate_invoice_voice_response(self, data: Dict[str, Any]) -> str:
        """Generate human-friendly response for invoice operations"""
        customer_name = data.get("customer_name", "the customer")
        total_amount = data.get("total_amount", 0)
        currency = data.get("currency", "euros")
        
        # Format amount nicely for voice
        if isinstance(total_amount, (int, float)) and total_amount > 0:
            amount_text = f"{total_amount} {currency}"
        else:
            amount_text = "the specified amount"
        
        items = data.get("items", [])
        if items and len(items) > 0:
            if len(items) == 1:
                item_text = f"for {items[0].get('description', 'the service')}"
            else:
                item_text = f"for {len(items)} different items"
        else:
            item_text = "for the requested services"
        
        return f"Perfect! I've created an invoice for {customer_name} {item_text} with a total of {amount_text}. The invoice has been saved in your system and is ready to be sent to the customer."
    
    def _generate_quote_voice_response(self, data: Dict[str, Any]) -> str:
        """Generate human-friendly response for quote operations"""
        customer_name = data.get("customer_name", "the customer")
        estimated_total = data.get("estimated_total", 0)
        currency = data.get("currency", "euros")
        
        if isinstance(estimated_total, (int, float)) and estimated_total > 0:
            amount_text = f"{estimated_total} {currency}"
        else:
            amount_text = "the estimated amount"
        
        services = data.get("services", [])
        if services and len(services) > 0:
            if len(services) == 1:
                service_text = f"for {services[0].get('description', 'the service')}"
            else:
                service_text = f"for {len(services)} different services"
        else:
            service_text = "for the requested services"
        
        return f"Excellent! I've prepared a quote for {customer_name} {service_text} with an estimated total of {amount_text}. The quote is ready for your review and can be sent to the customer whenever you're ready."
    
    def _generate_customer_voice_response(self, data: Dict[str, Any]) -> str:
        """Generate human-friendly response for customer operations"""
        name = data.get("name", "the new contact")
        email = data.get("email")
        phone = data.get("phone")
        
        contact_details = []
        if email and email != "N/A":
            contact_details.append(f"email {email}")
        if phone and phone != "N/A":
            contact_details.append(f"phone number {phone}")
        
        if contact_details:
            contact_text = " with " + " and ".join(contact_details)
        else:
            contact_text = ""
        
        return f"Great! I've successfully added {name}{contact_text} to your customer database. Their information has been saved and they're now available for future invoices, quotes, and job scheduling."
    
    def _generate_job_voice_response(self, data: Dict[str, Any]) -> str:
        """Generate human-friendly response for job operations"""
        title = data.get("title", "the job")
        customer_name = data.get("customer_name", "the customer")
        scheduled_date = data.get("scheduled_date", "the scheduled time")
        duration = data.get("duration")
        
        duration_text = ""
        if duration and duration != "N/A":
            duration_text = f" for {duration}"
        
        return f"Perfect! I've scheduled {title} for {customer_name} on {scheduled_date}{duration_text}. The job has been added to your calendar and all the relevant details have been saved in your system."
    
    def _generate_expense_voice_response(self, data: Dict[str, Any]) -> str:
        """Generate human-friendly response for expense operations"""
        description = data.get("description", "the expense")
        amount = data.get("amount", 0)
        currency = data.get("currency", "euros")
        category = data.get("category")
        date = data.get("date", "today")
        
        if isinstance(amount, (int, float)) and amount > 0:
            amount_text = f"{amount} {currency}"
        else:
            amount_text = "the specified amount"
        
        category_text = ""
        if category and category != "N/A":
            category_text = f" in the {category} category"
        
        return f"Done! I've recorded the expense for {description} with an amount of {amount_text}{category_text} for {date}. This has been added to your expense tracking system for proper record keeping."
    
    def _is_specific_id_query(self, prompt: str, intent: Intent) -> bool:
        """
        Check if the prompt is asking for a specific item by ID
        """
        prompt_lower = prompt.lower()
        id_patterns = [
            "by id", "with id", "id:", "invoice id", "client id", 
            "quote id", "job id", "expense id", "meeting id"
        ]
        
        # Check for ID patterns
        for pattern in id_patterns:
            if pattern in prompt_lower:
                return True
        
        # Check for specific ID formats (UUID, ObjectId, etc.)
        import re
        id_regex = r'\b[a-f0-9]{24}\b|\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b'
        if re.search(id_regex, prompt_lower):
            return True
            
        return False
    
    def _extract_id_from_prompt(self, prompt: str, intent: Intent) -> Dict[str, Any]:
        """
        Extract ID from prompt for specific item queries
        """
        import re
        
        # Look for various ID formats
        id_patterns = [
            r'\b[a-f0-9]{24}\b',  # MongoDB ObjectId
            r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',  # UUID
            r'id[:\s]+([a-f0-9-]+)',  # "id: xyz" or "id xyz"
            r'#([a-f0-9-]+)',  # "#xyz"
        ]
        
        extracted_id = None
        for pattern in id_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                extracted_id = match.group(1) if match.groups() else match.group(0)
                break
        
        return {
            "extracted_data": {
                "id": extracted_id,
                "query_type": "specific_id"
            },
            "confidence": 0.9 if extracted_id else 0.0,
            "missing_fields": [] if extracted_id else ["id"]
        }
    
    # Audio processing methods
    async def process_audio_request(
        self, 
        audio_bytes: bytes, 
        user_id: str, 
        language: str = "en",
        audio_filename: str = "audio.mp3"
    ) -> Dict[str, Any]:
        """
        Complete audio-to-audio processing workflow
        
        Args:
            audio_bytes: Raw audio data
            user_id: User identifier
            language: Language preference
            audio_filename: Original filename for format detection
            
        Returns:
            Complete processing result with transcription, structured data, and audio response
        """
        if not self.audio_enabled:
            return {
                "success": False,
                "error": "Audio processing not available",
                "message": "Audio services are not initialized"
            }
        
        try:
            self.logger.info(f"Starting audio processing for user {user_id}")
            
            # Step 1: Transcribe audio
            transcription_result = await self.audio_service.transcribe_bytes(
                audio_bytes, 
                filename=audio_filename,
                language=language
            )
            
            if not transcription_result or not transcription_result.get("success"):
                return {
                    "success": False,
                    "error": "Audio transcription failed",
                    "transcription_result": transcription_result
                }
            
            transcribed_text = transcription_result["text"]
            self.logger.info(f"Audio transcribed successfully: {transcribed_text[:100]}...")
            
            # Step 2: Process transcribed text through unified agent
            structured_response = await self.process_agent_request(
                prompt=transcribed_text,
                user_id=user_id,
                language=language
            )
            
            # Step 3: Generate human-friendly response
            human_response = self.generate_human_friendly_response(structured_response)
            
            # Step 4: Convert response to audio
            tts_result = await self.audio_service.synthesize_text(
                text=human_response,
                voice="alloy",  # Default voice, could be configurable
                model="tts-1"
            )
            
            if not tts_result or not tts_result.get("success"):
                # Return without audio if TTS fails
                return {
                    "success": True,
                    "transcribed_text": transcribed_text,
                    "structured_response": structured_response,
                    "human_response": human_response,
                    "audio_url": None,
                    "warning": "Audio response generation failed",
                    "transcription_result": transcription_result,
                    "tts_result": tts_result
                }
            
            # Step 5: Return complete result
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "structured_response": structured_response,
                "human_response": human_response,
                "audio_url": tts_result.get("output_path"),
                "transcription_result": transcription_result,
                "tts_result": tts_result
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return {
                "success": False,
                "error": f"Audio processing error: {str(e)}"
            }
    
    async def transcribe_audio(
        self, 
        audio_bytes: bytes, 
        language: Optional[str] = None,
        filename: str = "audio.mp3"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text only
        
        Args:
            audio_bytes: Raw audio data
            language: Optional language code
            filename: Filename for format detection
            
        Returns:
            Transcription result
        """
        if not self.audio_enabled:
            return {
                "success": False,
                "error": "Audio transcription not available"
            }
        
        return await self.audio_service.transcribe_bytes(
            audio_bytes, 
            filename=filename,
            language=language
        )
    
    async def generate_audio_response(
        self, 
        text: str, 
        voice: str = "alloy", 
        model: str = "tts-1"
    ) -> Dict[str, Any]:
        """
        Generate audio from text only
        
        Args:
            text: Text to convert to audio
            voice: Voice to use
            model: TTS model to use
            
        Returns:
            TTS result
        """
        if not self.audio_enabled:
            return {
                "success": False,
                "error": "Audio generation not available"
            }
        
        return await self.audio_service.synthesize_text(text, voice, model=model)
    
    def get_audio_service_info(self) -> Dict[str, Any]:
        """
        Get information about audio service capabilities
        
        Returns:
            Audio service information
        """
        if not self.audio_enabled:
            return {
                "enabled": False,
                "message": "Audio services not available"
            }
        
        return {
            "enabled": True,
            **self.audio_service.get_service_info()
        }
    
    async def test_audio_services(self) -> Dict[str, bool]:
        """
        Test audio service connections
        
        Returns:
            Test results for audio services
        """
        if not self.audio_enabled:
            return {
                "audio_enabled": False,
                "transcription": False,
                "tts": False
            }
        
        test_results = await self.audio_service.test_all_connections()
        test_results["audio_enabled"] = True
        return test_results
    
    def _is_get_all_query(self, prompt: str) -> bool:
        """
        Check if the prompt is a "get all" type query that should skip data extraction
        
        Examples: "get all my clients", "show all invoices", "list all quotes", "retrieve all jobs"
        """
        prompt_lower = prompt.lower().strip()
        
        # Keywords that indicate "get all" operations
        get_all_keywords = [
            "all my", "all the", "all of my", "every", "list all", "show all", 
            "get all", "retrieve all", "display all", "view all", "see all"
        ]
        
        # Check for get all patterns
        for keyword in get_all_keywords:
            if keyword in prompt_lower:
                return True
        
        # Check for specific patterns like "my clients", "all clients", etc.
        entity_patterns = [
            "clients", "invoices", "quotes", "jobs", "expenses", "customer", "invoice", "quote", "job", "expense"
        ]
        
        for entity in entity_patterns:
            # Patterns like "all clients", "my clients", "clients list", etc.
            if f"all {entity}" in prompt_lower or f"my {entity}" in prompt_lower:
                return True
        
        return False