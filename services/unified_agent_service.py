"""
Unified AI Agent Service
Handles single prompt workflow with intent detection, data extraction, and response formatting
"""

import json
import logging
import re
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings

from services.semantic_kernel_service import SemanticKernelService
from tools.client_tools import ClientTools
from tools.invoice_tools import InvoiceTools
from tools.quote_tools import QuoteTools
from tools.job_tools import JobTools
from tools.expense_tools import ExpenseTools
from tools.manual_task_tools import ManualTaskTools
from config.settings import Settings

class Intent(str, Enum):
    """Supported intents for AI agent - Order matters for priority"""
    CHIT_CHAT = "chit_chat"  # HIGHEST PRIORITY - Handle greetings/casual conversation first
    MANUAL_TASK = "manual_task"  # HIGH PRIORITY - Check first for tasks
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
    Workflow: Prompt -> Intent Detection -> Data Extraction -> Missing Data Check -> Response
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
        
        # In-memory conversation storage (replace with database in production)
        self.conversations: Dict[str, Dict] = {}
        
        # Required fields for each intent - Ordered by priority
        self.required_fields = {
            Intent.MANUAL_TASK: [
                "title", "start_time", "end_time"
            ],
            Intent.CUSTOMER: [
                "name", "email", "phone", "address"
            ],
            Intent.INVOICE: [
                "customer_name", "customer_email", "items", "total_amount", "title"
            ],
            Intent.QUOTE: [
                "customer_name", "customer_email", "services", "estimated_total"
            ],
            Intent.EXPENSE: [
                "description", "amount", "date", "category"
            ],
            Intent.JOB: [
                "title", "customer_name", "scheduled_date", "duration"
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

            # 1. Add User Input to History
            conversation["history"].append({"role": "user", "content": prompt})
            
            # ... (Existing Reset Logic) ...

            # Step 1: Intent Detection (for new conversations)
            if conversation["state"] == ConversationState.INTENT_DETECTION:
                intent, operation, confidence = await self._detect_intent(prompt, language)
                conversation["intent"] = intent
                conversation["operation"] = operation
                conversation["confidence"] = confidence
                conversation["data"] = {}

                self.logger.info(f"Intent detection result: intent={intent}, operation={operation}, confidence={confidence}")

                # Handle CHIT_CHAT intent - respond conversationally and don't proceed with business logic
                if intent == Intent.CHIT_CHAT:
                    self.logger.info("Detected chit-chat intent, responding conversationally")
                    chit_chat_response = await self._generate_chit_chat_response(prompt, language)
                    # Reset conversation for next interaction
                    self.reset_conversation(user_id)
                    return chit_chat_response

                # Special handling for "get all" queries - skip data extraction entirely
                if operation == Operation.GET and self._is_get_all_query(prompt):
                    self.logger.info(f"Detected 'get all' query for {intent.value}, skipping to response generation")
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
                elif intent == Intent.UNKNOWN or confidence < 0.1:
                    self.logger.warning(f"Intent unclear or low confidence: {intent}, {confidence}")
                    return self._create_clarification_response(conversation, language)
                else:
                    # Always proceed to data extraction after intent detection
                    conversation["state"] = ConversationState.DATA_EXTRACTION

            else:
                # If we're mid-conversation, check whether the user has changed their intent/operation.
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
                        else:
                            # If user explicitly asks a GET while mid-flow, allow immediate GET
                            if new_operation == Operation.GET and new_confidence >= 0.4:
                                self.logger.info(f"Switching to GET operation mid-flow (confidence {new_confidence})")
                                conversation["operation"] = Operation.GET
                                conversation["state"] = ConversationState.DATA_EXTRACTION
                    except Exception:
                        # If intent re-detection fails, continue with existing flow
                        self.logger.debug("Intent re-detection failed while mid-conversation; continuing existing flow")
            
            # Step 2: Data Extraction (initial or additional data)
            if conversation["state"] in [ConversationState.DATA_EXTRACTION, ConversationState.DATA_COMPLETION]:
                extracted_data = await self._extract_data(
                    prompt, conversation["intent"], conversation.get("operation", Operation.UNKNOWN), language, conversation["history"]  # Pass history
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
                    if conversation.get("missing_data_attempts", 0) >= 3: # Increased to 3
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
                        
                        # Generate the question
                        response = self._create_missing_data_response(conversation, missing_fields, language)
                        
                        # Add AI Question to History so it remembers it asked!
                        conversation["history"].append({"role": "assistant", "content": response["message"]})
                        return response
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
            
            # Fallback: attempt quick re-detection instead of failing immediately
            # This improves resilience when conversation state becomes inconsistent.
            try:
                alt_intent, alt_operation, alt_conf = await self._detect_intent(prompt, language)
                if alt_intent != Intent.UNKNOWN and alt_conf >= 0.2:
                    # Reset conversation to new intent and restart flow
                    conversation["intent"] = alt_intent
                    conversation["operation"] = alt_operation
                    conversation["confidence"] = alt_conf
                    conversation["data"] = {}
                    conversation["missing_data_attempts"] = 0
                    conversation["state"] = ConversationState.RESPONSE_GENERATION if alt_operation == Operation.GET else ConversationState.DATA_EXTRACTION
                    # Re-run the processing loop recursively (one level) by calling process_agent_request again
                    return await self.process_agent_request(prompt, user_id, language)
            except Exception:
                self.logger.debug("Fallback re-detection attempt failed")

            return self._create_error_response("Invalid conversation state", language)
            
        except Exception as e:
            self.logger.error(f"Error processing agent request: {e}")
            return self._create_error_response(str(e), language)
    
    async def _detect_intent(self, prompt: str, language: str) -> Tuple[Intent, Operation, float]:
        """
        Detect user intent from the prompt using AI
        
        Returns:
            Tuple of (intent, operation, confidence_score)
        """
        # First check for chit-chat patterns (quick check before calling LLM)
        chit_chat_patterns = [
            r"^(hi|hello|hey|bonjour|salut|coucou|good\s*(morning|afternoon|evening)|what'?s?\s*up)[\s!.?]*$",
            r"^(thanks?|thank\s*you|merci|awesome|great|perfect|ok|okay|cool|nice|got\s*it)[\s!.?]*$",
            r"^(bye|goodbye|see\s*you|au\s*revoir|ciao|later)[\s!.?]*$",
            r"^(how\s*are\s*you|how'?s?\s*it\s*going|comment\s*(ça\s*)?va|ça\s*va)[\s!.?]*$",
            r"^(who\s*are\s*you|what\s*can\s*you\s*do|help|aide)[\s!.?]*$",
        ]
        
        prompt_lower = prompt.strip().lower()
        for pattern in chit_chat_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return Intent.CHIT_CHAT, Operation.UNKNOWN, 0.95
        
        intent_prompt = f"""
        Analyze this user prompt and determine their intent and operation. Respond with JSON only.
        
        User prompt: "{prompt}"
        
        INTENT DETECTION PRIORITY ORDER (Check in this order):
        1. CHIT_CHAT (Highest Priority) - Greetings, thanks, casual conversation
        2. MANUAL_TASK 
        3. CUSTOMER
        4. INVOICE 
        5. QUOTE
        6. EXPENSE
        7. JOB (Lowest Priority - only if no other intent matches)
        
        OPERATIONS:
        - get: Viewing/retrieving existing data (show, list, get, display, find, see, view, retrieve, "all my")
        - create: Creating new data (create, add, schedule, book, make, generate, new)
        - update: Modifying existing data (update, change, modify, edit, adjust)
        - delete: Removing data (delete, remove, cancel, eliminate)
        - unknown: For chit-chat or unclear operations
        
        CHIT_CHAT INDICATORS (Check FIRST - Highest Priority):
        ✅ Greetings: "hi", "hello", "hey", "good morning", "bonjour", "salut"
        ✅ Thanks: "thanks", "thank you", "merci", "awesome", "great", "perfect"
        ✅ Farewells: "bye", "goodbye", "see you", "au revoir"
        ✅ Small talk: "how are you", "what's up", "how's it going"
        ✅ Help requests: "help", "what can you do", "who are you"
        ✅ Acknowledgments: "ok", "okay", "got it", "understood", "cool", "nice"
        ✅ No business keywords present
        
        MANUAL_TASK INDICATORS (Check SECOND):
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
        🔴 IF greeting/thanks/farewell/small talk → ALWAYS chit_chat
        🔴 IF color mentioned → ALWAYS manual_task (red task, blue work, green reminder)
        🔴 IF "task" + no client name → ALWAYS manual_task  
        🔴 IF "planning" or "reminder" → ALWAYS manual_task
        🔴 IF client name mentioned → Then consider job
        🔴 IF just time/date without client → manual_task
        
        EXAMPLES - CHIT_CHAT:
        ✅ "Hello!" → chit_chat, unknown
        ✅ "Hi there" → chit_chat, unknown
        ✅ "Thanks!" → chit_chat, unknown
        ✅ "How are you?" → chit_chat, unknown
        ✅ "What can you do?" → chit_chat, unknown
        ✅ "Great, thanks" → chit_chat, unknown
        
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
            result = await self.sk_service._execute_agent_request(
                system_prompt="You are an AI assistant that analyzes user prompts to determine intent and operation. Respond with JSON only.",
                user_prompt=intent_prompt,
                agent_type="intent_detection"
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
                
                # Map strings to enums (case-insensitive)
                try:
                    intent = Intent(intent_str.lower())
                except ValueError:
                    intent = Intent.UNKNOWN
                    confidence = 0.0
                
                try:
                    operation = Operation(operation_str.lower())
                except ValueError:
                    operation = Operation.UNKNOWN
                
                # If AI returns valid result with confidence > 0.6, RETURN IT IMMEDIATELY.
                if confidence > 0.6:
                    return intent, operation, confidence
            
        except Exception as e:
            self.logger.error(f"Intent detection failed: {e}")
        
        # ONLY REACH HERE IF AI FAILED OR CONFIDENCE IS LOW
        
        # Fallback: Simple pattern matching
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
        language: str,
        history: List[Dict] = None # NEW ARGUMENT
    ) -> Dict[str, Any]:
        """
        Extract relevant data fields based on the detected intent
        """
        # Check if this is a specific ID query
        if self._is_specific_id_query(prompt, intent):
            return self._extract_id_from_prompt(prompt, intent)
        
        # For GET operations, extract search/filter parameters
        if operation == Operation.GET:
            return await self._extract_get_query_params(prompt, intent, language)
        extraction_prompts = {
            Intent.INVOICE: """
            Extract comprehensive invoice data from this prompt. Return JSON with these fields:
            
            CLIENT INFORMATION:
            - customer_name: Customer/client full name
            - customer_email: Email address
            - customer_phone: Phone number (optional)
            - customer_address: Full address (optional)
            - customer_company_type: "COMPANY" or "INDIVIDUAL" (based on context)
            
            PROJECT DETAILS:
            - title: Invoice title/subject
            - project_name: Project or job name (optional)
            - project_address: Complete project address (optional)
            - project_street_address: Street address component (optional)
            - project_zip_code: ZIP/postal code (optional)
            - project_city: City name (optional)
            
            INVOICE DETAILS:
            - invoice_type: "FINAL", "INTERIM", "ADVANCE", or "CREDIT" (based on context)
            - items: Array of {description, quantity, unit_price, total, type}
            - subtotal: Subtotal amount before discounts
            
            DISCOUNT INFORMATION:
            - discount: Discount amount or percentage value
            - discount_type: "FIXED" (euro amount) or "PERCENTAGE"
            
            DOWN PAYMENT INFORMATION:
            - down_payment: Down payment amount or percentage value
            - down_payment_type: "FIXED" (euro amount) or "PERCENTAGE"
            
            TAX AND TOTALS:
            - vat_rate: VAT rate (default 0.20 = 20%)
            - vat_amount: VAT amount
            - total_amount: Final total after all calculations
            
            DATES:
            - invoice_date: Date (ISO format, default today)
            - due_date: Due date (ISO format, default +30 days)
            
            NOTES (categorize appropriately):
            - notes: General notes about the invoice
            - internal_notes: Internal notes (not visible to client)
            - public_notes: Notes visible on PDF/to client
            
            SIGNATURES (if mentioned):
            - contractor_signature: Contractor signature reference
            - client_signature: Client signature reference
            
            Extract invoice type from context clues:
            - FINAL: "final invoice", "completion", "balance", "remaining payment"
            - INTERIM: "interim invoice", "progress payment", "milestone", "partial"
            - ADVANCE: "advance payment", "upfront", "deposit invoice", "prepayment"
            - CREDIT: "credit note", "refund", "adjustment", "correction"
            
            Determine discount type:
            - PERCENTAGE: if "%" symbol present or percentage mentioned
            - FIXED: if euro/currency amount specified
            
            Determine company type from context:
            - INDIVIDUAL: "person", "individual", "freelancer", "self-employed"
            - COMPANY: "company", "business", "corp", "ltd", "organization"
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
            # Use appropriate tool based on intent - MANUAL_TASK gets highest priority
            if intent == Intent.MANUAL_TASK:
                result = await self.sk_service.process_manual_task_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.CUSTOMER:
                result = await self.sk_service.process_customer_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.INVOICE:
                result = await self.sk_service.process_invoice_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.QUOTE:
                result = await self.sk_service.process_quote_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.EXPENSE:
                result = await self.sk_service.process_expense_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.JOB:
                result = await self.sk_service.process_job_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
                )
            elif intent == Intent.MANUAL_TASK:
                result = await self.sk_service.process_manual_task_request(
                    prompt=full_prompt,
                    context={"task": "data_extraction"},
                    language=language,
                    history=history # Pass it here
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
        Flattens nested structures (like 'CLIENT INFORMATION', 'QUOTE DETAILS', etc.) to top level.
        """
        # Keys that indicate nested data sections that should be flattened (normalized versions)
        nested_section_patterns = [
            "extracted_data", "clientinformation", "client_information", "customerinformation",
            "projectdetails", "project_details", "quotedetails", "quote_details",
            "invoicedetails", "invoice_details", "discountinformation", "discount_information",
            "downpaymentinformation", "down_payment_information", "taxandtotals", "tax_and_totals",
            "dates", "notes", "expensedetails", "expense_details", "jobdetails", "job_details",
            "taskdetails", "task_details", "signatures"
        ]
        
        def is_nested_section(key: str) -> bool:
            """Check if a key represents a nested section that should be flattened"""
            normalized = key.lower().replace(" ", "").replace("_", "")
            for pattern in nested_section_patterns:
                if normalized == pattern.replace("_", ""):
                    return True
            return False
        
        def flatten_and_merge(data: Dict[str, Any], target: Dict[str, Any]) -> None:
            """Recursively flatten nested dictionaries and merge to target"""
            for key, value in data.items():
                # Check if this is a nested section that should be flattened
                if is_nested_section(key) and isinstance(value, dict):
                    # Recursively flatten nested sections - merge contents to top level
                    flatten_and_merge(value, target)
                elif isinstance(value, list) and key.lower() in ["items", "services"]:
                    # Keep items/services as arrays at top level
                    if self._is_meaningful_value(value):
                        target["services"] = value  # Normalize to 'services' for quotes
                        target["items"] = value     # Also keep as 'items' for invoices
                elif isinstance(value, dict) and not is_nested_section(key):
                    # Non-section dict, keep as is but also flatten if it has known fields
                    target[self._normalize_field_key(key)] = value
                elif self._is_meaningful_value(value):
                    # Normalize key to snake_case for consistency
                    normalized_key = self._normalize_field_key(key)
                    target[normalized_key] = value
        
        flatten_and_merge(new_data, existing_data)
    
    def _normalize_field_key(self, key: str) -> str:
        """Normalize field keys to consistent snake_case format"""
        # Map common variations to standard keys
        key_mappings = {
            "clientname": "customer_name",
            "client_name": "customer_name",
            "customername": "customer_name",
            "clientemail": "customer_email",
            "client_email": "customer_email",
            "customeremail": "customer_email",
            "clientcompanytype": "customer_company_type",
            "client_company_type": "customer_company_type",
            "customercompanytype": "customer_company_type",
            "estimatedtotal": "estimated_total",
            "estimated_total": "estimated_total",
            "totalamount": "total_amount",
            "total_amount": "total_amount",
            "total": "total_amount",
            "subtotal": "subtotal",
            "vatrate": "vat_rate",
            "vat_rate": "vat_rate",
            "validuntil": "valid_until",
            "valid_until": "valid_until",
            "projectname": "project_name",
            "project_name": "project_name",
            "publicnotes": "public_notes",
            "public_notes": "public_notes",
            "internalnotes": "internal_notes",
            "internal_notes": "internal_notes",
            "discounttype": "discount_type",
            "discount_type": "discount_type",
            "downpayment": "down_payment",
            "down_payment": "down_payment",
            "downpaymenttype": "down_payment_type",
            "down_payment_type": "down_payment_type",
            "services": "services",
            "items": "items",
        }
        
        # Normalize to lowercase without spaces
        normalized = key.lower().replace(" ", "_").replace("-", "_")
        
        # Return mapped key or original normalized key
        return key_mappings.get(normalized, normalized)
    
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
    
    async def _extract_get_query_params(self, prompt: str, intent: Intent, language: str) -> Dict[str, Any]:
        """
        Extract search and filter parameters from GET operation prompts.
        Uses LLM to intelligently extract client names, status filters, date ranges, etc.
        """
        try:
            # Define extraction prompts based on intent
            extraction_prompt = self._get_extraction_prompt_for_get(intent)
            
            system_prompt = f"""You are a data extraction assistant. Extract search and filter parameters from the user's query.
            
{extraction_prompt}

IMPORTANT RULES:
- Only extract values explicitly mentioned in the prompt
- For client/customer names, extract the exact name mentioned
- For status filters, map to valid status values
- Return null/empty for fields not mentioned
- Be precise - don't infer or assume values not in the prompt

Return ONLY valid JSON, no explanations."""

            user_prompt = f"Extract search parameters from: \"{prompt}\""
            
            result = await self.sk_service.kernel.invoke_prompt(
                user_prompt,
                system_message=system_prompt,
                settings=OpenAIChatPromptExecutionSettings(
                    max_tokens=500,
                    temperature=0.1
                )
            )
            
            response_text = str(result).strip()
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            extracted_data = json.loads(response_text)
            
            # Clean up null/empty values
            cleaned_data = {k: v for k, v in extracted_data.items() if v is not None and v != "" and v != []}
            
            self.logger.info(f"Extracted GET query params: {cleaned_data}")
            
            return {
                "extracted_data": cleaned_data,
                "confidence": 0.9 if cleaned_data else 0.5,
                "missing_fields": []
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting GET query params: {e}")
            return {
                "extracted_data": {},
                "confidence": 0.5,
                "missing_fields": []
            }
    
    def _get_extraction_prompt_for_get(self, intent: Intent) -> str:
        """Get the extraction prompt for GET operations based on intent"""
        
        prompts = {
            Intent.INVOICE: """Extract these fields for invoice search:
- client_name: Name of the client/customer to filter by (string or null)
- client_email: Email of the client to filter by (string or null)
- status: Invoice status filter (one of: draft, sent, paid, overdue, cancelled, or null)
- invoice_type: Type filter (one of: deposit, progress, final, or null)
- project_name: Project name to search for (string or null)
- title: Invoice title to search for (string or null)
- date_from: Start date for filtering (ISO format or null)
- date_to: End date for filtering (ISO format or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"client_name": ..., "client_email": ..., "status": ..., "invoice_type": ..., "project_name": ..., "title": ..., "date_from": ..., "date_to": ..., "search_term": ...}""",

            Intent.QUOTE: """Extract these fields for quote search:
- client_name: Name of the client/customer to filter by (string or null)
- client_email: Email of the client to filter by (string or null)
- status: Quote status filter (one of: draft, sent, accepted, rejected, expired, or null)
- project_name: Project name to search for (string or null)
- title: Quote title to search for (string or null)
- date_from: Start date for filtering (ISO format or null)
- date_to: End date for filtering (ISO format or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"client_name": ..., "client_email": ..., "status": ..., "project_name": ..., "title": ..., "date_from": ..., "date_to": ..., "search_term": ...}""",

            Intent.CUSTOMER: """Extract these fields for client/customer search:
- name: Name of the client to search for (string or null)
- email: Email to search for (string or null)
- phone: Phone number to search for (string or null)
- company: Company name to search for (string or null)
- status: Client status filter (one of: active, delinquent, archived, or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"name": ..., "email": ..., "phone": ..., "company": ..., "status": ..., "search_term": ...}""",

            Intent.EXPENSE: """Extract these fields for expense search:
- description: Description to search for (string or null)
- category: Category filter (one of: Materials, Transport, Equipment, Labor, Insurance, General, Training, Marketing, Others, or null)
- amount_min: Minimum amount filter (number or null)
- amount_max: Maximum amount filter (number or null)
- date_from: Start date for filtering (ISO format or null)
- date_to: End date for filtering (ISO format or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"description": ..., "category": ..., "amount_min": ..., "amount_max": ..., "date_from": ..., "date_to": ..., "search_term": ...}""",

            Intent.JOB: """Extract these fields for job search:
- title: Job title to search for (string or null)
- client_name: Client name to filter by (string or null)
- location: Location to search for (string or null)
- status: Job status filter (one of: scheduled, in_progress, completed, cancelled, or null)
- date_from: Start date for filtering (ISO format or null)
- date_to: End date for filtering (ISO format or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"title": ..., "client_name": ..., "location": ..., "status": ..., "date_from": ..., "date_to": ..., "search_term": ...}""",

            Intent.MANUAL_TASK: """Extract these fields for manual task search:
- title: Task title to search for (string or null)
- location: Location to search for (string or null)
- color: Color filter (string or null)
- date_from: Start date for filtering (ISO format or null)
- date_to: End date for filtering (ISO format or null)
- search_term: General search term if no specific field identified (string or null)

Return JSON: {"title": ..., "location": ..., "color": ..., "date_from": ..., "date_to": ..., "search_term": ...}"""
        }
        
        return prompts.get(intent, """Extract any search parameters mentioned:
- search_term: General search term (string or null)
Return JSON: {"search_term": ...}""")
    
    def _build_search_params(self, intent: Intent, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Build search parameters dictionary from extracted data for GET operations.
        Maps extracted fields to tool method parameters.
        """
        params = {}
        
        # Common search term - use search_term, title, or other text fields as general search
        search_fields = ["search_term", "title", "description", "project_name"]
        for field in search_fields:
            if data.get(field):
                params["search"] = str(data[field])
                break
        
        # Client name (for invoices, quotes, jobs)
        if data.get("client_name"):
            params["client_name"] = str(data["client_name"])
        elif data.get("name") and intent == Intent.CUSTOMER:
            params["search"] = str(data["name"])
        
        # Status filter
        if data.get("status"):
            params["status"] = str(data["status"]).lower()
        
        # Invoice type (specific to invoices)
        if data.get("invoice_type"):
            params["invoice_type"] = str(data["invoice_type"]).lower()
        
        # Category (for expenses)
        if data.get("category"):
            params["category"] = str(data["category"])
        
        # Color (for manual tasks)
        if data.get("color"):
            params["color"] = str(data["color"])
        
        # Date range
        if data.get("date_from"):
            params["date_from"] = str(data["date_from"])
        if data.get("date_to"):
            params["date_to"] = str(data["date_to"])
        
        # Email and phone for client search
        if data.get("email"):
            if not params.get("search"):
                params["search"] = str(data["email"])
        if data.get("phone"):
            if not params.get("search"):
                params["search"] = str(data["phone"])
        if data.get("company"):
            if not params.get("search"):
                params["search"] = str(data["company"])
        
        # Location (for jobs/tasks)
        if data.get("location"):
            if not params.get("search"):
                params["search"] = str(data["location"])
        
        self.logger.debug(f"Built search params for {intent.value}: {params}")
        return params
    
    def _check_missing_data(self, intent: Intent, operation: Operation, data: Dict[str, Any]) -> List[str]:
        """
        Check which required fields are missing for the given intent
        For GET operations, only check for missing data if it's a specific ID query
        Uses smart field matching to find data under various key names
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
        
        # Field aliases - maps required field names to possible alternative keys
        field_aliases = {
            "customer_name": ["customer_name", "client_name", "clientName", "name", "clientname"],
            "customer_email": ["customer_email", "client_email", "clientEmail", "email", "clientemail"],
            "services": ["services", "items", "line_items", "lineItems", "service_items"],
            "items": ["items", "services", "line_items", "lineItems", "invoice_items"],
            "estimated_total": ["estimated_total", "estimatedTotal", "total", "total_amount", "totalAmount", "subtotal"],
            "total_amount": ["total_amount", "totalAmount", "total", "estimated_total", "estimatedTotal", "subtotal"],
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
        
        for field in required:
            # Get all possible aliases for this field
            aliases = field_aliases.get(field, [field])
            
            # Check if any alias has a meaningful value
            field_found = False
            for alias in aliases:
                if alias in data and self._is_meaningful_value(data[alias]):
                    field_found = True
                    break
            
            if not field_found:
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
        For GET operations, calls semantic kernel service to retrieve real data
        For CREATE operations, returns dummy responses that match manual endpoint structure
        """
        
        # Handle GET operations by directly calling tool functions
        if operation == Operation.GET:
            try:
                # Check if tools are initialized
                if not self.client_tools or not self.job_tools or not self.invoice_tools or not self.quote_tools or not self.expense_tools:
                    return {
                        "success": False,
                        "message": "Tools not initialized",
                        "data": None
                    }
                
                # Check if this is a specific ID query
                if data.get("query_type") == "specific_id" and data.get("id"):
                    # Handle specific ID queries
                    if intent == Intent.JOB:
                        result = await self.job_tools.get_job_by_id(data["id"], user_id=user_id)
                    elif intent == Intent.CUSTOMER:
                        result = await self.client_tools.get_client_by_id(data["id"], user_id=user_id)
                    elif intent == Intent.EXPENSE:
                        result = await self.expense_tools.get_expense_by_id(data["id"], user_id=user_id)
                    elif intent == Intent.INVOICE:
                        result = await self.invoice_tools.get_invoice_by_id(data["id"], user_id=user_id)
                    elif intent == Intent.QUOTE:
                        result = await self.quote_tools.get_quote_by_id(data["id"], user_id=user_id)
                    elif intent == Intent.MANUAL_TASK:
                        result = await self.manual_task_tools.get_manual_task_by_id(data["id"], user_id=user_id)
                    else:
                        return {
                            "success": False,
                            "message": f"Unsupported GET by ID operation for intent: {intent.value}",
                            "data": None
                        }
                else:
                    # Handle general list queries with search/filter parameters
                    # Extract search params from the data dict
                    search_params = self._build_search_params(intent, data)
                    self.logger.info(f"GET query with search params: {search_params}")
                    
                    if intent == Intent.JOB:
                        result = await self.job_tools.get_jobs(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            status_filter=search_params.get("status", ""),
                            client_name=search_params.get("client_name", ""),
                            date_from=search_params.get("date_from", ""),
                            date_to=search_params.get("date_to", "")
                        )
                    elif intent == Intent.CUSTOMER:
                        result = await self.client_tools.get_clients(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            status_filter=search_params.get("status", "")
                        )
                    elif intent == Intent.EXPENSE:
                        result = await self.expense_tools.get_expenses(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            category_filter=search_params.get("category", ""),
                            date_from=search_params.get("date_from", ""),
                            date_to=search_params.get("date_to", "")
                        )
                    elif intent == Intent.INVOICE:
                        result = await self.invoice_tools.get_invoices(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            status_filter=search_params.get("status", ""),
                            client_name=search_params.get("client_name", ""),
                            invoice_type=search_params.get("invoice_type", ""),
                            date_from=search_params.get("date_from", ""),
                            date_to=search_params.get("date_to", "")
                        )
                    elif intent == Intent.QUOTE:
                        result = await self.quote_tools.get_quotes(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            status_filter=search_params.get("status", ""),
                            client_name=search_params.get("client_name", ""),
                            date_from=search_params.get("date_from", ""),
                            date_to=search_params.get("date_to", "")
                        )
                    elif intent == Intent.MANUAL_TASK:
                        result = await self.manual_task_tools.get_manual_tasks(
                            user_id=user_id,
                            search=search_params.get("search", ""),
                            color_filter=search_params.get("color", ""),
                            date_from=search_params.get("date_from", ""),
                            date_to=search_params.get("date_to", "")
                        )
                    else:
                        return {
                            "success": False,
                            "message": f"Unsupported GET operation for intent: {intent.value}",
                            "data": None
                        }
                
                # Parse the JSON result
                if isinstance(result, str):
                    try:
                        result_data = json.loads(result)
                    except json.JSONDecodeError:
                        result_data = {"error": "Failed to parse result", "raw": result}
                else:
                    result_data = result
                
                return {
                    "success": True,
                    "message": f"Retrieved {intent.value}s successfully",
                    "data": result_data,
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
    
    async def _generate_chit_chat_response(self, prompt: str, language: str) -> Dict[str, Any]:
        """
        Generate a friendly conversational response for chit-chat intents.
        This makes the bot feel more human and approachable.
        
        Args:
            prompt: User's conversational message
            language: Language preference (en/fr)
            
        Returns:
            Friendly response dictionary
        """
        prompt_lower = prompt.strip().lower()
        
        # Define response templates based on common patterns
        responses = {
            "en": {
                "greeting": [
                    "Hello! 👋 How can I help you with your business today?",
                    "Hi there! Ready to help you manage invoices, quotes, clients, or expenses. What would you like to do?",
                    "Hey! Good to see you. What can I assist you with today?"
                ],
                "thanks": [
                    "You're welcome! Let me know if you need anything else. 😊",
                    "Happy to help! Is there anything else I can do for you?",
                    "No problem! Feel free to ask if you need more assistance."
                ],
                "farewell": [
                    "Goodbye! Have a great day! 👋",
                    "See you later! Don't hesitate to come back if you need help.",
                    "Take care! I'll be here when you need me."
                ],
                "how_are_you": [
                    "I'm doing great, thanks for asking! Ready to help you with your business tasks. What would you like to do?",
                    "I'm here and ready to assist! How can I help you today?"
                ],
                "help": [
                    "I'm your business assistant! I can help you with:\\n• Creating and managing invoices\\n• Generating quotes\\n• Managing clients/customers\\n• Scheduling jobs and tasks\\n• Tracking expenses\\n\\nJust tell me what you need!",
                    "I can assist you with invoices, quotes, client management, job scheduling, and expense tracking. What would you like to work on?"
                ],
                "default": [
                    "I'm here to help with your business tasks! Would you like to create an invoice, quote, manage clients, schedule work, or track expenses?",
                    "Thanks for chatting! How can I assist you with your business today?"
                ]
            },
            "fr": {
                "greeting": [
                    "Bonjour ! 👋 Comment puis-je vous aider avec votre entreprise aujourd'hui ?",
                    "Salut ! Prêt à vous aider à gérer factures, devis, clients ou dépenses. Que souhaitez-vous faire ?",
                    "Bonjour ! Content de vous voir. En quoi puis-je vous aider ?"
                ],
                "thanks": [
                    "De rien ! N'hésitez pas si vous avez besoin d'autre chose. 😊",
                    "Avec plaisir ! Y a-t-il autre chose que je peux faire pour vous ?",
                    "Pas de problème ! N'hésitez pas à demander si vous avez besoin d'aide."
                ],
                "farewell": [
                    "Au revoir ! Passez une excellente journée ! 👋",
                    "À bientôt ! N'hésitez pas à revenir si vous avez besoin d'aide.",
                    "Prenez soin de vous ! Je serai là quand vous aurez besoin de moi."
                ],
                "how_are_you": [
                    "Je vais très bien, merci de demander ! Prêt à vous aider avec vos tâches professionnelles. Que souhaitez-vous faire ?",
                    "Je suis là et prêt à vous aider ! Comment puis-je vous aider aujourd'hui ?"
                ],
                "help": [
                    "Je suis votre assistant professionnel ! Je peux vous aider avec :\\n• Création et gestion de factures\\n• Génération de devis\\n• Gestion des clients\\n• Planification de travaux et tâches\\n• Suivi des dépenses\\n\\nDites-moi ce dont vous avez besoin !",
                    "Je peux vous aider avec les factures, devis, gestion clients, planification de travaux et suivi des dépenses. Sur quoi souhaitez-vous travailler ?"
                ],
                "default": [
                    "Je suis là pour vous aider avec vos tâches professionnelles ! Souhaitez-vous créer une facture, un devis, gérer des clients, planifier du travail ou suivre des dépenses ?",
                    "Merci de discuter ! Comment puis-je vous aider avec votre entreprise aujourd'hui ?"
                ]
            }
        }
        
        lang_responses = responses.get(language, responses["en"])
        
        # Determine response category
        import random
        if any(word in prompt_lower for word in ["hi", "hello", "hey", "bonjour", "salut", "coucou", "good morning", "good afternoon", "good evening"]):
            category = "greeting"
        elif any(word in prompt_lower for word in ["thanks", "thank you", "merci", "awesome", "great", "perfect", "cool", "nice"]):
            category = "thanks"
        elif any(word in prompt_lower for word in ["bye", "goodbye", "see you", "au revoir", "ciao", "later"]):
            category = "farewell"
        elif any(word in prompt_lower for word in ["how are you", "how's it going", "comment ça va", "ça va", "what's up"]):
            category = "how_are_you"
        elif any(word in prompt_lower for word in ["help", "what can you do", "who are you", "aide", "que peux-tu faire"]):
            category = "help"
        else:
            category = "default"
        
        response_text = random.choice(lang_responses.get(category, lang_responses["default"]))
        
        return {
            "success": True,
            "message": response_text,
            "action": "chit_chat",
            "intent": "chit_chat"
        }
    
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
                "missing_data_attempts": 0,
                "history": [],  # Initialize history list
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
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
            "confidence": conversation["confidence"],
            "has_data": bool(conversation["data"]),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"]
        }
    
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