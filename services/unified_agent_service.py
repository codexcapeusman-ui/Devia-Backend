"""
Unified AI Agent Service
Handles single prompt workflow with intent detection, data extraction, and response formatting
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from services.semantic_kernel_service import SemanticKernelService

class Intent(str, Enum):
    """Supported intents for AI agent"""
    INVOICE = "invoice"
    QUOTE = "quote"
    CUSTOMER = "customer"
    JOB = "job"
    EXPENSE = "expense"
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
    
    def __init__(self, sk_service: SemanticKernelService):
        self.sk_service = sk_service
        self.logger = logging.getLogger(__name__)
        
        # In-memory conversation storage (replace with database in production)
        self.conversations: Dict[str, Dict] = {}
        
        # Required fields for each intent
        self.required_fields = {
            Intent.INVOICE: [
                "customer_name", "customer_email", "items", "total_amount"
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

            # Step 1: Intent Detection (for new conversations)
            if conversation["state"] == ConversationState.INTENT_DETECTION:
                intent, operation, confidence = await self._detect_intent(prompt, language)
                conversation["intent"] = intent
                conversation["operation"] = operation
                conversation["confidence"] = confidence
                conversation["data"] = {}

                self.logger.info(f"Intent detection result: intent={intent}, operation={operation}, confidence={confidence}")

                if intent == Intent.UNKNOWN or confidence < 0.1:
                    self.logger.warning(f"Intent unclear or low confidence: {intent}, {confidence}")
                    return self._create_clarification_response(conversation, language)

                # For GET operations, skip data extraction and go directly to response generation
                if operation == Operation.GET:
                    conversation["state"] = ConversationState.RESPONSE_GENERATION
                else:
                    conversation["state"] = ConversationState.DATA_EXTRACTION

            else:
                # If we're mid-conversation, check whether the user has changed their intent/operation.
                try:
                    new_intent, new_operation, new_confidence = await self._detect_intent(prompt, language)
                    # If the new intent/operation is different and confidence is reasonably high, switch flows
                    if new_intent != conversation.get("intent") and new_confidence >= 0.6:
                        self.logger.info(f"User changed intent mid-flow from {conversation.get('intent')} to {new_intent} (conf={new_confidence})")
                        conversation["intent"] = new_intent
                        conversation["operation"] = new_operation
                        conversation["confidence"] = new_confidence
                        # Reset collected data but keep it optional to be merged later if fields overlap
                        conversation["data"] = {}
                        conversation["missing_data_attempts"] = 0
                        conversation["state"] = ConversationState.RESPONSE_GENERATION if new_operation == Operation.GET else ConversationState.DATA_EXTRACTION
                    else:
                        # If user explicitly asks a GET while mid-flow, allow immediate GET
                        if new_operation == Operation.GET and new_confidence >= 0.4:
                            self.logger.info(f"Switching to GET operation mid-flow (confidence {new_confidence})")
                            conversation["operation"] = Operation.GET
                            conversation["state"] = ConversationState.RESPONSE_GENERATION
                except Exception:
                    # If intent re-detection fails, continue with existing flow
                    self.logger.debug("Intent re-detection failed while mid-conversation; continuing existing flow")
            
            # Step 2: Data Extraction (initial or additional data)
            if conversation["state"] in [ConversationState.DATA_EXTRACTION, ConversationState.DATA_COMPLETION]:
                extracted_data = await self._extract_data(
                    prompt, conversation["intent"], language
                )
                
                # Merge data intelligently - preserve existing valid data
                self._merge_conversation_data(conversation["data"], extracted_data)
                conversation["state"] = ConversationState.DATA_COMPLETION
            
            # Step 3: Check for Missing Data
            if conversation["state"] == ConversationState.DATA_COMPLETION:
                missing_fields = self._check_missing_data(
                    conversation["intent"], conversation["data"]
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
            Tuple of (intent, confidence_score)
        """
        intent_prompt = f"""
        Analyze this user prompt and determine their intent and operation. Respond with JSON only.
        
        User prompt: "{prompt}"
        
        Available intents:
        - invoice: Related to invoices
        - quote: Related to quotes/estimates
        - customer: Related to customer data
        - job: Related to jobs/appointments
        - expense: Related to expenses
        - unknown: Intent unclear or not supported
        
        Available operations:
        - get: Retrieving/viewing data (show, list, get, display)
        - create: Creating new data (create, add, schedule, book)
        - update: Modifying existing data (update, change, modify)
        - delete: Removing data (delete, remove, cancel)
        - unknown: Operation unclear
        
        Response format:
        {{
            "intent": "intent_name",
            "operation": "operation_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation"
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
                
                # Map strings to enums
                try:
                    intent = Intent(intent_str)
                except ValueError:
                    intent = Intent.UNKNOWN
                    confidence = 0.0
                
                try:
                    operation = Operation(operation_str)
                except ValueError:
                    operation = Operation.UNKNOWN
                
                return intent, operation, confidence
            
        except Exception as e:
            self.logger.error(f"Intent detection failed: {e}")
        
        return Intent.UNKNOWN, Operation.UNKNOWN, 0.0
    
    async def _extract_data(
        self, 
        prompt: str, 
        intent: Intent, 
        language: str
    ) -> Dict[str, Any]:
        """
        Extract relevant data fields based on the detected intent
        """
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
            Extract quote data from this prompt. Return JSON with these fields:
            - customer_name: Customer/client name
            - customer_email: Email address
            - customer_phone: Phone number (optional)
            - services: Array of {description, estimated_hours, hourly_rate, total}
            - subtotal: Subtotal amount
            - discount_percent: Discount percentage (optional)
            - discount_amount: Discount amount (optional)
            - estimated_total: Final estimated total
            - valid_until: Quote validity date (ISO format, default +30 days)
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
        """
        for key, value in new_data.items():
            # Only update if the new value is meaningful
            if self._is_meaningful_value(value):
                existing_data[key] = value
            # Keep existing value if new value is not meaningful and we have existing data
            elif key not in existing_data:
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
    
    def _check_missing_data(self, intent: Intent, data: Dict[str, Any]) -> List[str]:
        """
        Check which required fields are missing for the given intent
        """
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
        For GET operations, calls semantic kernel service to retrieve real data
        For CREATE operations, returns dummy responses that match manual endpoint structure
        """
        
        # Handle GET operations by directly calling tool functions
        if operation == Operation.GET:
            try:
                # Check if tools are initialized
                if not self.sk_service.job_tools or not self.sk_service.invoice_tools or not self.sk_service.quote_tools:
                    return {
                        "success": False,
                        "message": "Tools not initialized",
                        "data": None
                    }
                
                # Use tools from semantic kernel service
                # Call appropriate GET method based on intent
                if intent == Intent.JOB:
                    result = await self.sk_service.job_tools.get_jobs()
                elif intent == Intent.CUSTOMER:
                    result = await self.sk_service.job_tools.get_clients()
                elif intent == Intent.EXPENSE:
                    result = await self.sk_service.job_tools.get_expenses()
                elif intent == Intent.INVOICE:
                    result = await self.sk_service.invoice_tools.get_invoices()
                elif intent == Intent.QUOTE:
                    result = await self.sk_service.quote_tools.get_quotes()
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
                "invoice_id": "INV-2025-001",
                "invoice_number": "INV-001",
                "customer": {
                    "name": data.get("customer_name", ""),
                    "email": data.get("customer_email", ""),
                    "phone": data.get("customer_phone", ""),
                    "address": data.get("customer_address", "")
                },
                "items": data.get("items", []),
                "subtotal": data.get("subtotal", 0),
                "tax_amount": data.get("tax_amount", 0),
                "total_amount": data.get("total_amount", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat(),
                "due_date": data.get("due_date", "")
            }
        
        elif intent == Intent.QUOTE:
            base_response["data"] = {
                "quote_id": "QUO-2025-001",
                "quote_number": "QUO-001",
                "customer": {
                    "name": data.get("customer_name", ""),
                    "email": data.get("customer_email", ""),
                    "phone": data.get("customer_phone", "")
                },
                "services": data.get("services", []),
                "subtotal": data.get("subtotal", 0),
                "discount_amount": data.get("discount_amount", 0),
                "estimated_total": data.get("estimated_total", 0),
                "status": "draft",
                "created_at": datetime.now().isoformat(),
                "valid_until": data.get("valid_until", "")
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
            "confidence": conversation["confidence"],
            "has_data": bool(conversation["data"]),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"]
        }