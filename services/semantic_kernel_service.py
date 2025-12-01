"""
Main Semantic Kernel service that orchestrates all AI agents and tools
"""

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins import MathPlugin, TimePlugin
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import asyncio
import logging
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta

from config.settings import Settings
from tools.invoice_tools import InvoiceTools
from tools.customer_tools import CustomerTools
from tools.quote_tools import QuoteTools
from tools.job_tools import JobTools
from tools.expense_tools import ExpenseTools
from tools.manual_task_tools import ManualTaskTools

class SemanticKernelService:
    """
    Main service class that manages Semantic Kernel integration
    and orchestrates all AI agents and tools for business automation
    """
    
    def __init__(self, settings: Settings):
        """Initialize the Semantic Kernel service"""
        self.settings = settings
        self.kernel: Optional[sk.Kernel] = None
        self.chat_service: Optional[OpenAIChatCompletion] = None
        self._initialized = False
        
        # Tool instances
        self.invoice_tools: Optional[InvoiceTools] = None
        self.customer_tools: Optional[CustomerTools] = None
        self.quote_tools: Optional[QuoteTools] = None
        self.job_tools: Optional[JobTools] = None
        self.expense_tools: Optional[ExpenseTools] = None
        self.manual_task_tools: Optional[ManualTaskTools] = None
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, settings.sk_log_level))
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize Semantic Kernel and all tools"""
        try:
            self.logger.info("Initializing Semantic Kernel service...")
            
            # Validate OpenAI API key
            if not self.settings.validate_openai_key():
                raise ValueError("OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file")
            
            # Create kernel
            self.kernel = sk.Kernel()
            
            # Add OpenAI chat completion service
            self.chat_service = OpenAIChatCompletion(
                ai_model_id=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                service_id="chat_completion"
            )
            self.kernel.add_service(self.chat_service)
            
            # Add core plugins
            self.kernel.add_plugin(MathPlugin(), plugin_name="math")
            self.kernel.add_plugin(TimePlugin(), plugin_name="time")
            
            # Initialize business tools
            await self._initialize_business_tools()
            
            # Add business tools as plugins
            self.kernel.add_plugin(self.invoice_tools, plugin_name="invoice")
            self.kernel.add_plugin(self.customer_tools, plugin_name="customer")
            self.kernel.add_plugin(self.quote_tools, plugin_name="quote")
            self.kernel.add_plugin(self.job_tools, plugin_name="job")
            self.kernel.add_plugin(self.expense_tools, plugin_name="expense")
            self.kernel.add_plugin(self.manual_task_tools, plugin_name="manual_task")
            
            self._initialized = True
            self.logger.info("Semantic Kernel service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Semantic Kernel service: {e}")
            raise
    
    async def _initialize_business_tools(self) -> None:
        """Initialize all business-specific tools"""
        self.invoice_tools = InvoiceTools(self.settings)
        self.customer_tools = CustomerTools(self.settings)
        self.quote_tools = QuoteTools(self.settings)
        self.job_tools = JobTools(self.settings)
        self.expense_tools = ExpenseTools(self.settings)
        self.manual_task_tools = ManualTaskTools(self.settings)
        
        # Initialize tools if they have async initialization
        for tool in [self.invoice_tools, self.customer_tools, self.quote_tools, 
                    self.job_tools, self.expense_tools, self.manual_task_tools]:
            if hasattr(tool, 'initialize'):
                await tool.initialize()
    
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized"""
        return self._initialized and self.kernel is not None
    
    async def test_openai_connection(self) -> bool:
        """Test the OpenAI connection"""
        try:
            if not self.chat_service:
                return False
            
            # Simple test prompt
            test_prompt = "Reply with just the word 'OK' if you can read this."
            result = await self.chat_service.get_chat_message_content(
                chat_history=[],
                settings=sk.PromptExecutionSettings(),
                prompt=test_prompt
            )
            
            return "OK" in str(result).upper()
            
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    async def process_invoice_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process invoice generation request using AI agent
        
        Args:
            prompt: Natural language description of the invoice to generate
            context: Optional context data (client_id, quote_id, etc.)
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing the generated invoice data or error information
        """
        try:
            self.logger.info(f"Processing invoice request: {prompt[:100]}...")
            
            # Create system prompt for invoice generation
            system_prompt = self._get_invoice_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "invoice", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "invoice", history)
            
            return {
                "success": True,
                "message": "Invoice generated successfully" if language == "en" else "Facture générée avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Invoice processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to generate invoice: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_customer_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process customer data extraction request using AI agent
        
        Args:
            prompt: Natural language text containing customer information
            context: Optional context data
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing extracted customer data or error information
        """
        try:
            self.logger.info(f"Processing customer request: {prompt[:100]}...")
            
            # Create system prompt for customer extraction
            system_prompt = self._get_customer_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "customer", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "customer", history)
            
            return {
                "success": True,
                "message": "Customer data extracted successfully" if language == "en" else "Données client extraites avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Customer processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to extract customer data: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_quote_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process quote generation request using AI agent
        
        Args:
            prompt: Natural language description of the quote to generate
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing the generated quote data or error information
        """
        try:
            self.logger.info(f"Processing quote request: {prompt[:100]}...")
            
            # Create system prompt for quote generation
            system_prompt = self._get_quote_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "quote", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "quote", history)
            
            return {
                "success": True,
                "message": "Quote generated successfully" if language == "en" else "Devis généré avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Quote processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to generate quote: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_job_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process job scheduling request using AI agent
        
        Args:
            prompt: Natural language description of the job to schedule
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing the scheduled job data or error information
        """
        try:
            self.logger.info(f"Processing job request: {prompt[:100]}...")
            
            # Create system prompt for job scheduling
            system_prompt = self._get_job_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "job", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "job", history)
            
            return {
                "success": True,
                "message": "Job scheduled successfully" if language == "en" else "Travail programmé avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Job processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to schedule job: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_expense_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process expense tracking request using AI agent
        
        Args:
            prompt: Natural language description of the expense or receipt text
            context: Optional context data (receipt_text, etc.)
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing the processed expense data or error information
        """
        try:
            self.logger.info(f"Processing expense request: {prompt[:100]}...")
            
            # Create system prompt for expense tracking
            system_prompt = self._get_expense_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "expense", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "expense", history)
            
            return {
                "success": True,
                "message": "Expense tracked successfully" if language == "en" else "Dépense enregistrée avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Expense processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to track expense: {str(e)}",
                "errors": [str(e)]
            }

    async def process_manual_task_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: list = None) -> Dict[str, Any]:
        """
        Process manual task creation request using AI agent
        
        Args:
            prompt: Natural language description of the manual task
            context: Optional context data
            language: Response language (en/fr)
            history: Optional conversation history for multi-turn context
        
        Returns:
            Dictionary containing the processed manual task data or error information
        """
        try:
            self.logger.info(f"Processing manual task request: {prompt[:100]}...")
            
            # Create system prompt for manual task
            system_prompt = self._get_manual_task_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "manual_task", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "manual_task", history)
            
            return {
                "success": True,
                "message": "Manual task created successfully" if language == "en" else "Tâche manuelle créée avec succès",
                "data": result.get("data") if isinstance(result, dict) else result
            }
            
        except Exception as e:
            self.logger.error(f"Manual task processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to process manual task: {str(e)}",
                "errors": [str(e)]
            }
    
    def _clean_and_parse_json(self, response_text: str) -> Dict[str, Any]:
        """
        Robustly extract and parse JSON from LLM responses.
        Handles cases where LLM adds conversational text before/after JSON.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary or error dictionary
        """
        import re
        
        if not response_text or not response_text.strip():
            return {"error": "Empty response from AI", "raw_response": response_text}
        
        # Try 1: Direct JSON parse (best case scenario)
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try 2: Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
        code_block_patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",       # ``` ... ```
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Try 3: Find JSON object by locating first '{' and last '}'
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try 4: Find JSON array by locating first '[' and last ']'
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            try:
                parsed = json.loads(json_str)
                return {"items": parsed} if isinstance(parsed, list) else parsed
            except json.JSONDecodeError:
                pass
        
        # Try 5: Handle common LLM response patterns
        # Remove common prefixes like "Sure!", "Here is the data:", etc.
        cleaned = response_text.strip()
        prefix_patterns = [
            r"^(?:Sure!?|Here(?:'s| is| are).*?:|Certainly!?|Of course!?|I(?:'ll| will).*?:)\s*",
            r"^(?:The|Based on|According to).*?:\s*",
        ]
        
        for prefix_pattern in prefix_patterns:
            cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Try parsing the cleaned text
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Final fallback: Return the raw response with error indication
        self.logger.warning(f"Failed to parse JSON from LLM response: {response_text[:200]}...")
        return {
            "error": "Could not parse JSON from AI response",
            "raw_response": response_text,
            "response": response_text  # Keep response for backward compatibility
        }

    async def _execute_agent_request(self, system_prompt: str, user_prompt: str, agent_type: str, history: list = None) -> Dict[str, Any]:
        """
        Execute an agent request using Semantic Kernel
        
        Args:
            system_prompt: System instructions for the AI
            user_prompt: User's request with context
            agent_type: Type of agent (invoice, customer, quote, job, expense)
            history: Optional conversation history for context
        
        Returns:
            Parsed result from the AI agent
        """
        if not self.is_initialized():
            raise RuntimeError("Semantic Kernel service is not initialized")
        
        # Create chat history
        chat_history = ChatHistory()
        chat_history.add_system_message(system_prompt)
        
        # Add conversation history if provided (for multi-turn context)
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    chat_history.add_user_message(msg.get("content", ""))
                elif msg.get("role") == "assistant":
                    chat_history.add_assistant_message(msg.get("content", ""))
        
        chat_history.add_user_message(user_prompt)
        
        # Configure execution settings
        execution_settings = OpenAIChatPromptExecutionSettings(
            service_id="chat_completion",
            ai_model_id=self.settings.openai_model,
            max_tokens=2000,
            temperature=0.1,  # Low temperature for consistent results
            top_p=0.9
        )
        
        # Execute the request
        result = await self.chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings
        )
        
        # Parse the result using robust JSON parser
        response_text = str(result)
        parsed_result = self._clean_and_parse_json(response_text)
        
        return {"success": True, "data": parsed_result}
    
    def _prepare_prompt_with_context(self, prompt: str, context: Optional[Dict[str, Any]], agent_type: str, language: str) -> str:
        """Prepare the full prompt with context information"""
        
        context_info = ""
        if context:
            context_info = f"\\nContext: {json.dumps(context, indent=2)}"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        language_note = ""
        if language == "fr":
            language_note = "\\nRéponds en français."
        
        full_prompt = f"""
Current time: {current_time}
Company: {self.settings.company_name}
Default VAT rate: {self.settings.default_vat_rate}%
Default currency: {self.settings.default_currency}
{context_info}

User request: {prompt}
{language_note}
"""
        
        return full_prompt.strip()
    
    def _get_invoice_system_prompt(self, language: str) -> str:
        """Get system prompt for invoice generation agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans la génération complète de factures pour l'entreprise Devia.

RÔLE: Analyser les demandes en langage naturel et générer des données de facture structurées avec support complet des champs.

FONCTIONS DISPONIBLES:
- invoice.create_invoice: Créer une facture à partir d'une description avec extraction complète des champs
- invoice.update_invoice: Mettre à jour une facture existante avec support de tous les champs
- invoice.delete_invoice: Supprimer une facture
- invoice.calculate_invoice_totals: Calculer les totaux (TVA, remises, acomptes)
- invoice.generate_invoice_number: Générer un numéro de facture unique
- invoice.get_invoices: Lister les factures
- invoice.get_invoice_by_id: Récupérer une facture par ID
- customer.extract_customer_data: Extraire les données client depuis le texte

CHAMPS DE FACTURE SUPPORTÉS:
- Informations Client: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Détails Projet: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Types de Facture: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Système de Remise: discount (montant), discountType (FIXED/PERCENTAGE)
- Système d'Acompte: downPayment (montant), downPaymentType (FIXED/PERCENTAGE)
- Notes: notes (générales), internalNotes (privées), publicNotes (visibles client)
- Éléments: items (array avec description, quantity, unitPrice, total, type)

NORMALISATION DES DONNÉES (CRITIQUE):
Tu DOIS convertir les données en langage naturel vers des valeurs structurées:
- "dix heures" → 10 (nombre)
- "cinquante euros" → 50.0 (nombre)
- "demain" → date ISO (ex: 2025-12-02)
- "la semaine prochaine" → date ISO calculée
- "vingt pourcent" → 20 (pourcentage)
- "mardi prochain" → date ISO calculée

EXEMPLES (FEW-SHOT):

Utilisateur: "Créer une facture pour Jean Dupont, dix heures de développement web à cinquante euros l'heure"
Assistant: {
    "clientName": "Jean Dupont",
    "clientCompanyType": "INDIVIDUAL",
    "items": [{"description": "Développement Web", "quantity": 10, "unitPrice": 50.0, "total": 500.0, "type": "labor"}],
    "subtotal": 500.0,
    "vatRate": 20,
    "vatAmount": 100.0,
    "total": 600.0,
    "status": "complete"
}

Utilisateur: "Facture de 500 euros"
Assistant: {
    "missing_fields": ["clientName", "items"],
    "response": "Pour créer cette facture de 500€, j'ai besoin du nom du client et du détail des services fournis.",
    "status": "incomplete"
}

Utilisateur: "Ajouter client Marie Martin"
Assistant: {
    "missing_fields": ["clientEmail"],
    "response": "Je peux ajouter Marie Martin. Avez-vous son adresse email ?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyser la demande pour identifier toutes les informations.
2. CONVERTIR les mots en nombres (dix→10, cinquante→50, etc.)
3. CONVERTIR les dates relatives en dates absolues ISO.
4. Si des informations ESSENTIELLES manquent, retourner status="incomplete" avec les champs manquants.
5. Retourner TOUJOURS du JSON valide, jamais de texte conversationnel seul.

TOUJOURS retourner un JSON valide."""
        else:
            return """You are an AI assistant specialized in comprehensive invoice generation for Devia.

ROLE: Analyze natural language requests and generate structured invoice data with full field support.

AVAILABLE FUNCTIONS:
- invoice.create_invoice: Create an invoice from natural language description
- invoice.update_invoice: Update an existing invoice
- invoice.delete_invoice: Delete an invoice
- invoice.calculate_invoice_totals: Calculate totals (VAT, discounts, down payments)
- invoice.get_invoices: List invoices
- invoice.get_invoice_by_id: Get invoice by ID

SUPPORTED INVOICE FIELDS:
- Client Information: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Project Details: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Invoice Types: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Discount System: discount (amount), discountType (FIXED/PERCENTAGE)
- Down Payment System: downPayment (amount), downPaymentType (FIXED/PERCENTAGE)
- Notes: notes (general), internalNotes (private), publicNotes (client-visible)
- Items: items (array with description, quantity, unitPrice, total, type)

DATA NORMALIZATION (CRITICAL):
You MUST convert natural language data to structured values:
- "ten hours" → 10 (number)
- "fifty dollars" → 50.0 (number)
- "tomorrow" → ISO date (e.g., 2025-12-02)
- "next week" → calculated ISO date
- "twenty percent" → 20 (percentage)
- "next Tuesday" → calculated ISO date
- "a hundred" → 100 (number)
- "half a day" → 4 (hours)

FEW-SHOT EXAMPLES:

User: "Create an invoice for John Smith, ten hours of web development at fifty dollars per hour"
Assistant: {
    "clientName": "John Smith",
    "clientCompanyType": "INDIVIDUAL",
    "items": [{"description": "Web Development", "quantity": 10, "unitPrice": 50.0, "total": 500.0, "type": "labor"}],
    "subtotal": 500.0,
    "vatRate": 20,
    "vatAmount": 100.0,
    "total": 600.0,
    "status": "complete"
}

User: "Invoice for 500 dollars"
Assistant: {
    "missing_fields": ["clientName", "items"],
    "response": "To create this $500 invoice, I need the customer name and a breakdown of services provided.",
    "status": "incomplete"
}

User: "Add client John"
Assistant: {
    "missing_fields": ["clientEmail", "clientPhone"],
    "response": "I can add John. Do you have an email address or phone number for him?",
    "status": "incomplete"
}

User: "Bill ABC Corp twenty hours consulting at a hundred fifty per hour with ten percent discount"
Assistant: {
    "clientName": "ABC Corp",
    "clientCompanyType": "COMPANY",
    "items": [{"description": "Consulting", "quantity": 20, "unitPrice": 150.0, "total": 3000.0, "type": "service"}],
    "subtotal": 3000.0,
    "discount": 10,
    "discountType": "PERCENTAGE",
    "vatRate": 20,
    "vatAmount": 540.0,
    "total": 3240.0,
    "status": "complete"
}

INSTRUCTIONS:
1. Analyze the request to identify all information.
2. CONVERT words to numbers (ten→10, fifty→50, hundred→100, etc.)
3. CONVERT relative dates to absolute ISO dates.
4. If ESSENTIAL information is missing, return status="incomplete" with missing_fields.
5. ALWAYS return valid JSON, never conversational text alone.

ALWAYS return valid JSON."""
    
    def _get_customer_system_prompt(self, language: str) -> str:
        """Get system prompt for customer data extraction agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans l'extraction et la gestion des données client pour Devia.

RÔLE: Analyser du texte en langage naturel pour extraire des informations client structurées.

FONCTIONS DISPONIBLES:
- customer.extract_customer_data: Extraire les données client du texte
- customer.validate_customer_info: Valider les informations client
- customer.format_address: Formater l'adresse
- customer.generate_customer_id: Générer un ID client unique

CHAMPS CLIENT:
- name: Nom complet du client
- email: Adresse email
- phone: Numéro de téléphone
- address: Adresse complète
- company: Nom de l'entreprise (si applicable)
- notes: Notes additionnelles

NORMALISATION DES DONNÉES (CRITIQUE):
- Formater les numéros de téléphone correctement
- Valider le format des emails
- Normaliser les adresses

EXEMPLES (FEW-SHOT):

Utilisateur: "Ajouter client Jean Dupont"
Assistant: {
    "missing_fields": ["email", "phone"],
    "response": "Je peux ajouter Jean Dupont. Avez-vous son email ou numéro de téléphone ?",
    "status": "incomplete"
}

Utilisateur: "Nouveau client Marie Martin, email marie@email.com, téléphone 0612345678"
Assistant: {
    "name": "Marie Martin",
    "email": "marie@email.com",
    "phone": "+33612345678",
    "status": "complete"
}

Utilisateur: "Client ABC Corp, 123 rue de la Paix Paris"
Assistant: {
    "name": "ABC Corp",
    "company": "ABC Corp",
    "address": "123 rue de la Paix, Paris",
    "missing_fields": ["email"],
    "response": "J'ai les informations pour ABC Corp. Avez-vous une adresse email de contact ?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyser le texte pour identifier les informations client.
2. Si des informations essentielles manquent (nom, email), demander poliment.
3. Retourner TOUJOURS du JSON valide.

TOUJOURS retourner un JSON valide avec la structure client."""
        else:
            return """You are an AI assistant specialized in customer data extraction and management for Devia.

ROLE: Analyze natural language text to extract structured customer information.

AVAILABLE FUNCTIONS:
- customer.extract_customer_data: Extract customer data from text
- customer.validate_customer_info: Validate customer information
- customer.format_address: Format address information
- customer.generate_customer_id: Generate unique customer ID

CUSTOMER FIELDS:
- name: Customer full name
- email: Email address
- phone: Phone number
- address: Full address
- company: Company name (if applicable)
- notes: Additional notes

DATA NORMALIZATION (CRITICAL):
- Format phone numbers correctly
- Validate email format
- Normalize addresses

FEW-SHOT EXAMPLES:

User: "Add client John"
Assistant: {
    "missing_fields": ["email", "phone"],
    "response": "I can add John. Do you have an email address or phone number for him?",
    "status": "incomplete"
}

User: "New customer Jane Smith, email jane@email.com, phone 555-123-4567"
Assistant: {
    "name": "Jane Smith",
    "email": "jane@email.com",
    "phone": "+15551234567",
    "status": "complete"
}

User: "Client ABC Corp, 123 Main Street New York"
Assistant: {
    "name": "ABC Corp",
    "company": "ABC Corp",
    "address": "123 Main Street, New York",
    "missing_fields": ["email"],
    "response": "I have the information for ABC Corp. Do you have a contact email address?",
    "status": "incomplete"
}

User: "Add customer Marie, works at Tech Solutions, her email is marie@tech.com"
Assistant: {
    "name": "Marie",
    "email": "marie@tech.com",
    "company": "Tech Solutions",
    "status": "complete"
}

INSTRUCTIONS:
1. Analyze text to identify customer information.
2. If essential information is missing (name, email), ask politely.
3. ALWAYS return valid JSON.

ALWAYS return valid JSON with customer structure."""
    
    def _get_quote_system_prompt(self, language: str) -> str:
        """Get system prompt for quote generation agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans la génération complète de devis pour Devia.

RÔLE: Analyser les demandes en langage naturel et générer des données de devis structurées avec support complet des champs.

FONCTIONS DISPONIBLES:
- quote.create_quote: Créer un devis avec extraction complète des champs
- quote.update_quote: Mettre à jour un devis avec support de tous les champs
- quote.delete_quote: Supprimer un devis
- quote.calculate_quote_totals: Calculer les totaux (TVA, remises, acomptes)
- quote.get_quotes: Lister les devis
- quote.get_quote_by_id: Récupérer un devis par ID

CHAMPS COMPLETS DE DEVIS SUPPORTÉS:
- Informations Client: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Détails Projet: title, projectName, projectStreetAddress, projectZipCode, projectCity
- Système Remise: montant, discountType (FIXED/PERCENTAGE)
- Système Acompte: montant, downPaymentType (FIXED/PERCENTAGE)
- Notes Multiples: internalNotes (privées), publicNotes (visibles client)
- Signatures: contractorSignature, clientSignature
- Validité: validUntil (date d'expiration du devis)

NORMALISATION DES DONNÉES (CRITIQUE):
- CONVERTIR mots en nombres: cinq→5, dix→10, cent→100, mille→1000
- CONVERTIR pourcentages textuels: "dix pour cent"→10, "vingt-cinq pourcent"→25
- CONVERTIR dates relatives en dates ISO: "demain"→YYYY-MM-DD, "la semaine prochaine"→YYYY-MM-DD
- FORMATER montants: toujours en nombres décimaux (500.00)

EXEMPLES (FEW-SHOT):

Utilisateur: "Devis pour peinture salon cinq cents euros"
Assistant: {
    "title": "Peinture salon",
    "items": [{"description": "Peinture salon", "quantity": 1, "unitPrice": 500.00}],
    "subtotal": 500.00,
    "missing_fields": ["clientName"],
    "response": "Je prépare le devis pour 500€. Pour quel client ?",
    "status": "incomplete"
}

Utilisateur: "Devis rénovation cuisine pour Dupont, deux mille euros, 10% de remise"
Assistant: {
    "title": "Rénovation cuisine",
    "clientName": "Dupont",
    "items": [{"description": "Rénovation cuisine", "quantity": 1, "unitPrice": 2000.00}],
    "subtotal": 2000.00,
    "discount": 200.00,
    "discountType": "PERCENTAGE",
    "total": 1800.00,
    "status": "complete"
}

INSTRUCTIONS:
1. Analyser la demande pour identifier tous les éléments projet et client.
2. CONVERTIR tous les mots en nombres appropriés.
3. Extraire informations client complètes incluant type d'entreprise.
4. Si des informations essentielles manquent, retourner status="incomplete" avec missing_fields.
5. TOUJOURS retourner du JSON valide.

TOUJOURS retourner un JSON valide avec structure devis améliorée supportant tous nouveaux champs."""
        else:
            return """You are an AI assistant specialized in comprehensive quote generation for Devia.

ROLE: Analyze natural language requests and generate structured quote data with full field support.

AVAILABLE FUNCTIONS:
- quote.create_quote: Create a quote from natural language description with comprehensive field extraction
- quote.update_quote: Update an existing quote with all field support
- quote.delete_quote: Delete a quote
- quote.calculate_quote_totals: Calculate totals (VAT, discounts, down payments)
- quote.get_quotes: List quotes
- quote.get_quote_by_id: Get quote by ID

COMPREHENSIVE QUOTE FIELDS SUPPORTED:
- Client Information: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Project Details: title, projectName, projectStreetAddress, projectZipCode, projectCity
- Discount System: discount amount, discountType (FIXED/PERCENTAGE)
- Down Payment System: downPayment amount, downPaymentType (FIXED/PERCENTAGE)
- Multiple Notes: internalNotes (private), publicNotes (client-visible)
- Digital Signatures: contractorSignature, clientSignature
- Validity: validUntil (quote expiration date)

DATA NORMALIZATION RULES (CRITICAL):
- CONVERT words to numbers: five→5, ten→10, fifty→50, hundred→100, thousand→1000
- CONVERT percentage words: "ten percent"→10, "twenty five percent"→25
- CONVERT relative dates to ISO format: "tomorrow"→YYYY-MM-DD, "next week"→YYYY-MM-DD
- FORMAT amounts: always as decimal numbers (500.00)

FEW-SHOT EXAMPLES:

User: "Quote for painting living room five hundred dollars"
Assistant: {
    "title": "Painting living room",
    "items": [{"description": "Painting living room", "quantity": 1, "unitPrice": 500.00}],
    "subtotal": 500.00,
    "missing_fields": ["clientName"],
    "response": "I'm preparing a quote for $500. Which client is this for?",
    "status": "incomplete"
}

User: "Quote for kitchen renovation for Smith, two thousand dollars, 10% discount"
Assistant: {
    "title": "Kitchen renovation",
    "clientName": "Smith",
    "items": [{"description": "Kitchen renovation", "quantity": 1, "unitPrice": 2000.00}],
    "subtotal": 2000.00,
    "discount": 200.00,
    "discountType": "PERCENTAGE",
    "total": 1800.00,
    "status": "complete"
}

User: "Create quote bathroom remodel fifteen hundred with 30% down payment for ABC Corp"
Assistant: {
    "title": "Bathroom remodel",
    "clientName": "ABC Corp",
    "clientCompanyType": "COMPANY",
    "items": [{"description": "Bathroom remodel", "quantity": 1, "unitPrice": 1500.00}],
    "subtotal": 1500.00,
    "downPayment": 450.00,
    "downPaymentType": "PERCENTAGE",
    "total": 1500.00,
    "status": "complete"
}

User: "Quote for flooring"
Assistant: {
    "title": "Flooring",
    "missing_fields": ["clientName", "amount"],
    "response": "I can create a flooring quote. What's the total amount and which client is this for?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyze the request to identify all project and client elements.
2. CONVERT all word numbers to actual numbers.
3. CONVERT relative dates to absolute ISO dates.
4. If ESSENTIAL information is missing (client, amount), return status="incomplete" with missing_fields.
5. ALWAYS return valid JSON, never conversational text alone.

ALWAYS return valid JSON with the enhanced quote structure supporting all new fields."""
    
    def _get_job_system_prompt(self, language: str) -> str:
        """Get system prompt for job scheduling agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans la gestion des TRAVAUX CLIENTS et rendez-vous professionnels pour Devia.

RÔLE: Gérer uniquement le travail FACTURABLE et les rendez-vous CLIENTS. 

IMPORTANT: Les "jobs" sont UNIQUEMENT pour le travail client facturable et les rendez-vous professionnels.
Pour les tâches internes/planification personnelle, utilisez les tâches manuelles (manual_task).

FONCTIONS DISPONIBLES:
DATA RETRIEVAL (retourne données réelles de la base):
- job.get_jobs: Récupérer la liste des jobs clients
- job.get_meetings: Récupérer les réunions professionnelles
- job.get_clients: Récupérer les clients
- job.get_expenses: Récupérer les dépenses
- job.get_invoices: Récupérer les factures
- job.get_quotes: Récupérer les devis

API OPERATIONS (retourne structures d'appel API):
- job.create_job_api_call: Créer un travail client
- job.update_job_api_call: Modifier un travail client
- job.delete_job_api_call: Supprimer un travail client
- job.create_meeting_api_call: Créer une réunion professionnelle
- job.update_meeting_api_call: Modifier une réunion professionnelle
- job.delete_meeting_api_call: Supprimer une réunion professionnelle

NORMALISATION DES DONNÉES (CRITIQUE):
- CONVERTIR mots en nombres: cinq→5, dix→10, deux heures→2
- CONVERTIR dates relatives en ISO: "demain"→YYYY-MM-DD, "lundi prochain"→YYYY-MM-DD
- CONVERTIR heures relatives: "dans une heure"→HH:MM, "ce soir à huit heures"→20:00
- CONVERTIR durées: "deux heures"→120 minutes, "une demi-heure"→30 minutes

EXEMPLES (FEW-SHOT):

Utilisateur: "Planifier travail peinture chez Dupont demain à 9h"
Assistant: {
    "title": "Travail peinture",
    "clientName": "Dupont",
    "startDate": "2024-01-16",
    "startTime": "09:00",
    "type": "job",
    "status": "complete"
}

Utilisateur: "Réunion client pour devis lundi prochain"
Assistant: {
    "title": "Réunion client - devis",
    "startDate": "2024-01-22",
    "type": "meeting",
    "missing_fields": ["clientName", "startTime"],
    "response": "Je prépare la réunion pour lundi. Avec quel client et à quelle heure ?",
    "status": "incomplete"
}

Utilisateur: "Job installation cuisine ABC Corp, trois heures, mardi 14h"
Assistant: {
    "title": "Installation cuisine",
    "clientName": "ABC Corp",
    "startDate": "2024-01-23",
    "startTime": "14:00",
    "duration": 180,
    "type": "job",
    "status": "complete"
}

INSTRUCTIONS:
1. Analyser la demande pour identifier client, date, heure, durée.
2. CONVERTIR toutes dates/heures relatives en format absolu.
3. Si des informations essentielles manquent (client, date), retourner status="incomplete".
4. TOUJOURS retourner du JSON valide.

TOUJOURS retourner un JSON valide avec la structure appropriée."""
        else:
            return """You are an AI assistant specialized in CLIENT WORK and professional appointments management for Devia.

ROLE: Manage only BILLABLE work and CLIENT appointments.

IMPORTANT: "Jobs" are ONLY for billable client work and professional appointments.
For internal tasks/personal planning, use manual tasks (manual_task).

AVAILABLE FUNCTIONS:
DATA RETRIEVAL (returns real database data):
- job.get_jobs: Retrieve client jobs list
- job.get_meetings: Retrieve professional meetings
- job.get_clients: Retrieve clients
- job.get_expenses: Retrieve expenses
- job.get_invoices: Retrieve invoices
- job.get_quotes: Retrieve quotes

API OPERATIONS (returns API call structures):
- job.create_job_api_call: Create a client job
- job.update_job_api_call: Update a client job
- job.delete_job_api_call: Delete a client job
- job.create_meeting_api_call: Create a professional meeting
- job.update_meeting_api_call: Update a professional meeting
- job.delete_meeting_api_call: Delete a professional meeting

DATA NORMALIZATION RULES (CRITICAL):
- CONVERT words to numbers: five→5, ten→10, two hours→2
- CONVERT relative dates to ISO: "tomorrow"→YYYY-MM-DD, "next Monday"→YYYY-MM-DD
- CONVERT relative times: "in an hour"→HH:MM, "tonight at eight"→20:00
- CONVERT durations: "two hours"→120 minutes, "half an hour"→30 minutes

FEW-SHOT EXAMPLES:

User: "Schedule painting job at Smiths tomorrow at 9am"
Assistant: {
    "title": "Painting job",
    "clientName": "Smiths",
    "startDate": "2024-01-16",
    "startTime": "09:00",
    "type": "job",
    "status": "complete"
}

User: "Client meeting for quote next Monday"
Assistant: {
    "title": "Client meeting - quote",
    "startDate": "2024-01-22",
    "type": "meeting",
    "missing_fields": ["clientName", "startTime"],
    "response": "I'm setting up the meeting for Monday. Which client and what time?",
    "status": "incomplete"
}

User: "Job kitchen installation ABC Corp, three hours, Tuesday 2pm"
Assistant: {
    "title": "Kitchen installation",
    "clientName": "ABC Corp",
    "startDate": "2024-01-23",
    "startTime": "14:00",
    "duration": 180,
    "type": "job",
    "status": "complete"
}

User: "Meeting with John about the renovation at ten thirty"
Assistant: {
    "title": "Renovation meeting",
    "clientName": "John",
    "startTime": "10:30",
    "type": "meeting",
    "missing_fields": ["startDate"],
    "response": "I'm scheduling the renovation meeting with John at 10:30. What date?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyze request to identify client, date, time, duration.
2. CONVERT all relative dates/times to absolute format.
3. If ESSENTIAL information is missing (client, date), return status="incomplete".
4. ALWAYS return valid JSON, never conversational text alone.

ALWAYS return valid JSON with appropriate structure."""
    
    def _get_expense_system_prompt(self, language: str) -> str:
        """Get system prompt for expense tracking agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans le suivi des dépenses pour Devia.

RÔLE: Analyser des reçus ou des descriptions de dépenses pour créer des données de dépense structurées.

FONCTIONS DISPONIBLES:
- expense.create_expense: Créer une dépense à partir d'une description
- expense.update_expense: Mettre à jour une dépense
- expense.delete_expense: Supprimer une dépense
- expense.get_expenses: Lister les dépenses
- expense.get_expense_by_id: Récupérer une dépense par ID
- expense.get_expenses_by_category: Lister les dépenses par catégorie
- expense.calculate_expense_totals: Calculer des totaux à partir d'une liste
- expense.categorize_expense: Catégoriser la dépense
- expense.calculate_vat: Calculer la TVA
- expense.parse_receipt: Analyser un reçu

NORMALISATION DES DONNÉES (CRITIQUE):
- CONVERTIR mots en nombres: cinq→5, cinquante→50, cent→100
- CONVERTIR montants: "vingt-cinq euros"→25.00, "trois cents"→300.00
- CONVERTIR dates relatives en ISO: "hier"→YYYY-MM-DD, "la semaine dernière"→YYYY-MM-DD
- CATÉGORISER automatiquement: essence→Carburant, restaurant→Repas, fournitures→Matériel

EXEMPLES (FEW-SHOT):

Utilisateur: "Dépense essence cinquante euros hier"
Assistant: {
    "description": "Essence",
    "amount": 50.00,
    "category": "Carburant",
    "date": "2024-01-14",
    "status": "complete"
}

Utilisateur: "Achat fournitures bureau"
Assistant: {
    "description": "Fournitures bureau",
    "category": "Matériel",
    "missing_fields": ["amount"],
    "response": "J'enregistre l'achat de fournitures. Quel est le montant ?",
    "status": "incomplete"
}

Utilisateur: "Repas client restaurant Le Gourmet, soixante-quinze euros"
Assistant: {
    "description": "Repas client - Restaurant Le Gourmet",
    "amount": 75.00,
    "category": "Repas d'affaires",
    "vendor": "Le Gourmet",
    "status": "complete"
}

INSTRUCTIONS:
1. Analyser le texte pour identifier description, montant, date, fournisseur.
2. CONVERTIR tous les mots en nombres.
3. CATÉGORISER automatiquement les dépenses.
4. Si le montant manque, retourner status="incomplete".
5. TOUJOURS retourner du JSON valide.

TOUJOURS retourner un JSON valide avec la structure de dépense."""
        else:
            return """You are an AI assistant specialized in expense tracking for Devia.

ROLE: Analyze receipt text or expense descriptions to create structured expense data.

AVAILABLE FUNCTIONS:
- expense.create_expense: Create an expense from a natural language description
- expense.update_expense: Update an existing expense
- expense.delete_expense: Delete an expense
- expense.get_expenses: List expenses
- expense.get_expense_by_id: Get expense by ID
- expense.get_expenses_by_category: List expenses by category
- expense.calculate_expense_totals: Calculate totals from a list of expenses
- expense.categorize_expense: Categorize the expense
- expense.calculate_vat: Calculate VAT amount
- expense.parse_receipt: Parse receipt information

DATA NORMALIZATION RULES (CRITICAL):
- CONVERT words to numbers: five→5, fifty→50, hundred→100
- CONVERT amounts: "twenty five dollars"→25.00, "three hundred"→300.00
- CONVERT relative dates to ISO: "yesterday"→YYYY-MM-DD, "last week"→YYYY-MM-DD
- AUTO-CATEGORIZE: gas/fuel→Fuel, restaurant/dining→Meals, supplies→Materials

FEW-SHOT EXAMPLES:

User: "Expense gas fifty dollars yesterday"
Assistant: {
    "description": "Gas",
    "amount": 50.00,
    "category": "Fuel",
    "date": "2024-01-14",
    "status": "complete"
}

User: "Bought office supplies"
Assistant: {
    "description": "Office supplies",
    "category": "Materials",
    "missing_fields": ["amount"],
    "response": "I'm recording the office supplies expense. What was the amount?",
    "status": "incomplete"
}

User: "Client lunch at The Steakhouse, seventy five dollars"
Assistant: {
    "description": "Client lunch - The Steakhouse",
    "amount": 75.00,
    "category": "Business Meals",
    "vendor": "The Steakhouse",
    "status": "complete"
}

User: "Hardware store two hundred thirty-five for materials"
Assistant: {
    "description": "Hardware store materials",
    "amount": 235.00,
    "category": "Materials",
    "status": "complete"
}

User: "Travel expense"
Assistant: {
    "description": "Travel expense",
    "category": "Travel",
    "missing_fields": ["amount", "date"],
    "response": "I'm recording a travel expense. What was the amount and when did this occur?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyze text to identify description, amount, date, and vendor.
2. CONVERT all word numbers to actual numbers.
3. AUTO-CATEGORIZE expenses based on description.
4. If amount is missing, return status="incomplete".
5. ALWAYS return valid JSON, never conversational text alone.

ALWAYS return valid JSON with an expense structure."""
    
    def _get_manual_task_system_prompt(self, language: str) -> str:
        """Get system prompt for manual task agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans la création et la gestion des tâches manuelles INTERNES pour Devia.

RÔLE: Analyser les demandes des utilisateurs pour créer des tâches manuelles structurées pour la planification interne uniquement.

IMPORTANT: Les tâches manuelles sont UNIQUEMENT pour la planification interne, les rappels personnels, et l'organisation d'équipe.
CE NE SONT PAS des rendez-vous clients ou du travail facturable.

FONCTIONS DISPONIBLES:
- manual_task.create_manual_task_api_call: Créer une tâche manuelle à partir d'une description textuelle
- manual_task.update_manual_task_api_call: Mettre à jour une tâche manuelle
- manual_task.delete_manual_task_api_call: Supprimer une tâche manuelle
- manual_task.get_manual_tasks: Lister les tâches manuelles
- manual_task.get_manual_task_by_id: Récupérer une tâche manuelle par ID
- manual_task.get_manual_tasks_by_date_range: Lister les tâches pour une période donnée

NORMALISATION DES DONNÉES (CRITIQUE):
- CONVERTIR dates relatives en ISO: "demain"→YYYY-MM-DD, "lundi"→YYYY-MM-DD
- CONVERTIR heures relatives: "à midi"→12:00, "ce soir"→18:00
- CONVERTIR durées: "une heure"→60 minutes, "toute la journée"→480 minutes
- RECONNAÎTRE couleurs: rouge, bleu, vert, jaune, orange, violet, rose

EXEMPLES (FEW-SHOT):

Utilisateur: "Rappel appeler comptable demain 10h"
Assistant: {
    "title": "Appeler comptable",
    "startDate": "2024-01-16",
    "startTime": "10:00",
    "type": "reminder",
    "status": "complete"
}

Utilisateur: "Tâche rouge réunion équipe lundi matin"
Assistant: {
    "title": "Réunion équipe",
    "startDate": "2024-01-22",
    "startTime": "09:00",
    "color": "red",
    "type": "internal_meeting",
    "status": "complete"
}

Utilisateur: "Bloquer après-midi vendredi pour admin, bleu"
Assistant: {
    "title": "Travail administratif",
    "startDate": "2024-01-19",
    "startTime": "14:00",
    "endTime": "18:00",
    "color": "blue",
    "duration": 240,
    "type": "block",
    "status": "complete"
}

Utilisateur: "Tâche commander fournitures"
Assistant: {
    "title": "Commander fournitures",
    "missing_fields": ["startDate"],
    "response": "Je crée la tâche. Pour quelle date ?",
    "status": "incomplete"
}

INSTRUCTIONS:
1. Analyser le texte pour identifier titre, dates, heures, couleur, durée.
2. CONVERTIR toutes dates/heures relatives en format absolu.
3. Identifier la couleur si mentionnée.
4. Si la date manque, retourner status="incomplete".
5. TOUJOURS retourner du JSON valide.

TOUJOURS retourner un JSON valide avec la structure de tâche manuelle."""
        else:
            return """You are an AI assistant specialized in creating and managing INTERNAL manual tasks for Devia.

ROLE: Analyze user requests to create structured manual tasks for internal planning purposes only.

IMPORTANT: Manual tasks are ONLY for internal planning, personal reminders, and team organization.
They are NOT client appointments or billable work.

AVAILABLE FUNCTIONS:
- manual_task.create_manual_task_api_call: Create a manual task from natural language description
- manual_task.update_manual_task_api_call: Update an existing manual task
- manual_task.delete_manual_task_api_call: Delete a manual task
- manual_task.get_manual_tasks: List manual tasks
- manual_task.get_manual_task_by_id: Get manual task by ID
- manual_task.get_manual_tasks_by_date_range: List tasks for a specific date range

DATA NORMALIZATION RULES (CRITICAL):
- CONVERT relative dates to ISO: "tomorrow"→YYYY-MM-DD, "Monday"→YYYY-MM-DD
- CONVERT relative times: "at noon"→12:00, "this evening"→18:00
- CONVERT durations: "one hour"→60 minutes, "all day"→480 minutes
- RECOGNIZE colors: red, blue, green, yellow, orange, purple, pink

FEW-SHOT EXAMPLES:

User: "Reminder call accountant tomorrow 10am"
Assistant: {
    "title": "Call accountant",
    "startDate": "2024-01-16",
    "startTime": "10:00",
    "type": "reminder",
    "status": "complete"
}

User: "Red task team meeting Monday morning"
Assistant: {
    "title": "Team meeting",
    "startDate": "2024-01-22",
    "startTime": "09:00",
    "color": "red",
    "type": "internal_meeting",
    "status": "complete"
}

User: "Block Friday afternoon for admin work, blue"
Assistant: {
    "title": "Admin work",
    "startDate": "2024-01-19",
    "startTime": "14:00",
    "endTime": "18:00",
    "color": "blue",
    "duration": 240,
    "type": "block",
    "status": "complete"
}

User: "Task order supplies"
Assistant: {
    "title": "Order supplies",
    "missing_fields": ["startDate"],
    "response": "I'm creating the task. What date should this be scheduled for?",
    "status": "incomplete"
}

User: "Green reminder dentist appointment Thursday at two thirty"
Assistant: {
    "title": "Dentist appointment",
    "startDate": "2024-01-18",
    "startTime": "14:30",
    "color": "green",
    "type": "reminder",
    "status": "complete"
}

INSTRUCTIONS:
1. Analyze text to identify title, dates, times, color, duration.
2. CONVERT all relative dates/times to absolute format.
3. Identify color if mentioned.
4. If date is missing, return status="incomplete".
5. ALWAYS return valid JSON, never conversational text alone.

ALWAYS return valid JSON with a manual task structure."""
    
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized"""
        return self._initialized and self.kernel is not None and self.chat_service is not None
    
    async def test_openai_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            if not self.chat_service:
                return False
            
            # Simple test prompt
            test_prompt = "Test connection - respond with 'OK'"
            
            # Create chat history
            history = ChatHistory()
            history.add_user_message(test_prompt)
            
            # Get response
            response = await self.chat_service.get_chat_message_content(
                chat_history=history,
                settings=OpenAIChatPromptExecutionSettings(
                    ai_model_id=self.settings.openai_model,
                    max_tokens=10,
                    temperature=0.1
                )
            )
            
            return response and len(str(response).strip()) > 0
            
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up Semantic Kernel service...")
            
            # Cleanup tools if they have cleanup methods
            for tool in [self.invoice_tools, self.customer_tools, self.quote_tools, 
                        self.job_tools, self.expense_tools]:
                if tool and hasattr(tool, 'cleanup'):
                    await tool.cleanup()
            
            self._initialized = False
            self.logger.info("Semantic Kernel service cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")