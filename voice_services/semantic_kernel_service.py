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
    
    async def process_invoice_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process invoice generation request using AI agent
        
        Args:
            prompt: Natural language description of the invoice to generate
            context: Optional context data (client_id, quote_id, etc.)
            language: Response language (en/fr)
            history: Previous conversation history
        
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
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Invoice processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to generate invoice: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_customer_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process customer data extraction request using AI agent
        
        Args:
            prompt: Natural language text containing customer information
            context: Optional context data
            language: Response language (en/fr)
            history: Previous conversation history
        
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
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Customer processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to extract customer data: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_quote_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process quote generation request using AI agent
        
        Args:
            prompt: Natural language description of the quote to generate
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
            history: Previous conversation history
        
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
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Quote processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to generate quote: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_job_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process job scheduling request using AI agent
        
        Args:
            prompt: Natural language description of the job to schedule
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
            history: Previous conversation history
        
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
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Job processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to schedule job: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_expense_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process expense tracking request using AI agent
        
        Args:
            prompt: Natural language description of the expense or receipt text
            context: Optional context data (receipt_text, etc.)
            language: Response language (en/fr)
            history: Previous conversation history
        
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
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Expense processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to track expense: {str(e)}",
                "errors": [str(e)]
            }
    
    async def process_manual_task_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en", history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process manual task creation request using AI agent
        
        Args:
            prompt: Natural language description of the manual task
            context: Optional context data (task details, etc.)
            language: Response language (en/fr)
            history: Previous conversation history
        
        Returns:
            Dictionary containing the processed manual task data or error information
        """
        try:
            self.logger.info(f"Processing manual task request: {prompt[:100]}...")
            
            # Create system prompt for manual task creation
            system_prompt = self._get_manual_task_system_prompt(language)
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt_with_context(prompt, context, "manual_task", language)
            
            # Execute with Semantic Kernel
            result = await self._execute_agent_request(system_prompt, full_prompt, "manual_task", history)
            
            return {
                "success": True,
                "message": "Manual task created successfully" if language == "en" else "Tâche manuelle créée avec succès",
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Manual task processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to create manual task: {str(e)}",
                "errors": [str(e)]
            }
    
    async def _execute_agent_request(self, system_prompt: str, user_prompt: str, agent_type: str, chat_history_list: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute an agent request using Semantic Kernel
        
        Args:
            system_prompt: System instructions for the AI
            user_prompt: User's request with context
            agent_type: Type of agent (invoice, customer, quote, job, expense)
            chat_history_list: Previous conversation history as list of {"role": "user/assistant", "content": "..."}
        
        Returns:
            Parsed result from the AI agent
        """
        if not self.is_initialized():
            raise RuntimeError("Semantic Kernel service is not initialized")
        
        # Create chat history
        chat_history = ChatHistory()
        chat_history.add_system_message(system_prompt)
        
        # Add previous conversation history if provided
        if chat_history_list:
            for msg in chat_history_list:
                if msg["role"] == "user":
                    chat_history.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    chat_history.add_assistant_message(msg["content"])
        
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
        
        # Parse the result
        response_text = str(result)
        
        try:
            # Try to parse as JSON first
            parsed_result = json.loads(response_text)
            return parsed_result
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {"response": response_text}
    
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
            return """Tu es un assistant IA spécialisé dans la génération complète de factures pour Devia.

RÔLE: Analyser les demandes vocales et générer des données de facture structurées avec support complet.

FONCTIONS DISPONIBLES:
- invoice.create_invoice: Créer une facture avec extraction complète des champs
- invoice.update_invoice: Mettre à jour une facture avec support de tous les champs
- invoice.delete_invoice: Supprimer une facture
- invoice.calculate_invoice_totals: Calculer les totaux (TVA, remises, acomptes)
- invoice.generate_invoice_number: Générer un numéro de facture unique
- invoice.get_invoices: Lister les factures
- invoice.get_invoice_by_id: Récupérer une facture par ID
- customer.extract_customer_data: Extraire les données client

CHAMPS COMPLETS DE FACTURE SUPPORTÉS:
- Informations Client: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Détails Projet: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Types Facture: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Système Remise: montant, discountType (FIXED/PERCENTAGE)
- Système Acompte: montant, downPaymentType (FIXED/PERCENTAGE)
- Notes Multiples: notes (générales), internalNotes (privées), publicNotes (client)
- Signatures: contractorSignature, clientSignature

INSTRUCTIONS OPTIMISÉES VOIX:
1. Traiter le texte transcrit vocal pouvant contenir des artefacts de reconnaissance.
2. Extraire informations client et projet depuis input conversationnel.
3. Identifier type facture depuis patterns vocaux ("facture finale", "acompte", "facture intermédiaire").
4. Analyser nombres et pourcentages parlés pour remises et acomptes.
5. Distinguer types de notes depuis contexte conversationnel.
6. Gérer patterns de parole informels en maintenant précision.
7. Lors d'appels get_*, TOUJOURS inclure user_id depuis contexte.
8. Retourner JSON structuré optimisé pour réponse vocale.

TOUJOURS retourner JSON valide avec structure facture améliorée."""
        else:
            return """You are an AI assistant specialized in comprehensive invoice generation for Devia.

ROLE: Analyze natural language requests and generate structured invoice data with full field support.

AVAILABLE FUNCTIONS:
- invoice.create_invoice: Create an invoice from natural language description with comprehensive field extraction
- invoice.update_invoice: Update an existing invoice with all field support
- invoice.delete_invoice: Delete an invoice
- invoice.calculate_invoice_totals: Calculate totals (VAT, discounts, down payments)
- invoice.generate_invoice_number: Generate unique invoice number
- invoice.get_invoices: List invoices
- invoice.get_invoice_by_id: Get invoice by ID
- customer.extract_customer_data: Extract customer data from text

COMPREHENSIVE INVOICE FIELDS SUPPORTED:
- Client Information: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Project Details: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Invoice Types: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Discount System: discount amount, discountType (FIXED/PERCENTAGE)
- Down Payment System: downPayment amount, downPaymentType (FIXED/PERCENTAGE)
- Multiple Notes: notes (general), internalNotes (private), publicNotes (client-visible)
- Digital Signatures: contractorSignature, clientSignature (base64 encoded)
- Enhanced Fields: quoteId (for quote conversion), userId

VOICE-OPTIMIZED INSTRUCTIONS:
1. Process voice-transcribed text that may have speech-to-text artifacts.
2. Extract comprehensive client and project information from conversational input.
3. Identify invoice type from natural speech patterns ("final bill", "down payment", "interim invoice").
4. Parse spoken numbers and percentages for discounts and down payments.
5. Distinguish between different note types from conversational context.
6. Handle informal speech patterns while maintaining data accuracy.
7. When calling get_* functions, ALWAYS include the user_id parameter from context.
8. Return structured data as JSON optimized for voice response formatting.

ALWAYS return valid JSON with the enhanced invoice structure supporting all new fields."""
    
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

INSTRUCTIONS:
1. Analyser le texte pour identifier les informations client
2. Extraire nom, email, téléphone, adresse, entreprise
3. Valider et formater les données
4. Générer un ID unique si nécessaire
5. Retourner les données structurées en JSON

TOUJOURS retourner un JSON valide avec la structure client."""
        else:
            return """You are an AI assistant specialized in customer data extraction and management for Devia.

ROLE: Analyze natural language text to extract structured customer information.

AVAILABLE FUNCTIONS:
- customer.extract_customer_data: Extract customer data from text
- customer.validate_customer_info: Validate customer information
- customer.format_address: Format address information
- customer.generate_customer_id: Generate unique customer ID

INSTRUCTIONS:
1. Analyze text to identify customer information
2. Extract name, email, phone, address, company
3. Validate and format the data
4. Generate unique ID if needed
5. Return structured data as JSON

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

INSTRUCTIONS:
1. Analyser la demande pour identifier tous les éléments projet et client.
2. Extraire informations client complètes incluant type d'entreprise.
3. Identifier détails projet (nom, adresse complète incluant code postal et ville).
4. Analyser informations de remise incluant type (pourcentage vs montant fixe).
5. Extraire détails d'acompte si mentionnés (pourcentage ou montant fixe).
6. Catégoriser notes en internes (privées) ou publiques (visibles client).
7. Calculer totaux en considérant remises, acomptes, et TVA.
8. Lors d'appels get_*, TOUJOURS inclure user_id depuis contexte.
9. Retourner données structurées en JSON, prêtes pour l'API complète.

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

INSTRUCTIONS:
1. Analyze the request to identify all project and client elements.
2. Extract comprehensive client information including company type (individual vs company).
3. Identify project details (name, full address components including ZIP and city).
4. Parse discount information including type (percentage vs fixed amount).
5. Extract down payment details if mentioned (percentage or fixed amount).
6. Categorize notes into internal (private) or public (client-visible).
7. Calculate totals considering discounts, down payments, and VAT.
8. When calling get_* functions, ALWAYS include user_id parameter from context.
9. Return structured data as JSON, ready for the comprehensive API.

ALWAYS return valid JSON with the enhanced quote structure supporting all new fields."""
    
    def _get_job_system_prompt(self, language: str) -> str:
        """Get system prompt for job scheduling agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans la gestion complète de calendrier et d'affaires pour Devia.

RÔLE: Analyser les demandes en langage naturel et gérer jobs, réunions, clients, dépenses, factures et devis.

FONCTIONS DISPONIBLES:
DATA RETRIEVAL (retourne données réelles de la base):
- job.get_jobs: Récupérer la liste des jobs
- job.get_meetings: Récupérer les réunions
- job.get_clients: Récupérer les clients
- job.get_expenses: Récupérer les dépenses
- job.get_invoices: Récupérer les factures
- job.get_quotes: Récupérer les devis

API OPERATIONS (retourne structures d'appel API):
- job.create_job_api_call: Créer un job
- job.update_job_api_call: Modifier un job
- job.delete_job_api_call: Supprimer un job
- job.create_meeting_api_call: Créer une réunion
- job.update_meeting_api_call: Modifier une réunion
- job.delete_meeting_api_call: Supprimer une réunion

LEGACY FUNCTIONS (toujours disponibles):
- job.create_job_from_text: Créer un travail à partir du texte
- job.parse_schedule_info: Analyser les informations de planification
- job.validate_schedule: Valider le planning
- job.suggest_optimal_times: Suggérer des créneaux optimaux
- job.reschedule_job: Replanifier un travail existant
- time.get_current_time: Obtenir l'heure actuelle

INSTRUCTIONS:
1. Pour CONSULTER des données, utiliser les fonctions get_* (retournent données réelles).
2. Pour CRÉER/MODIFIER/SUPPRIMER, utiliser les fonctions *_api_call (retournent structure API).
3. Utiliser les fonctions legacy pour analyse et suggestions.
4. Analyser et convertir les expressions temporelles en dates.
5. Retourner les données exactement comme reçues des fonctions.

TOUJOURS retourner un JSON valide avec la structure appropriée."""
        else:
            return """You are an AI assistant specialized in comprehensive calendar and business management for Devia.

ROLE: Analyze natural language requests and manage jobs, meetings, clients, expenses, invoices, and quotes.

AVAILABLE FUNCTIONS:
DATA RETRIEVAL (returns real database data):
- job.get_jobs: Retrieve jobs list
- job.get_meetings: Retrieve meetings
- job.get_clients: Retrieve clients
- job.get_expenses: Retrieve expenses
- job.get_invoices: Retrieve invoices
- job.get_quotes: Retrieve quotes

API OPERATIONS (returns API call structures):
- job.create_job_api_call: Create a job
- job.update_job_api_call: Update a job
- job.delete_job_api_call: Delete a job
- job.create_meeting_api_call: Create a meeting
- job.update_meeting_api_call: Update a meeting
- job.delete_meeting_api_call: Delete a meeting

LEGACY FUNCTIONS (still available):
- job.create_job_from_text: Create job from text description
- job.parse_schedule_info: Parse scheduling information
- job.validate_schedule: Validate schedule feasibility
- job.suggest_optimal_times: Suggest optimal time slots
- job.reschedule_job: Reschedule an existing job
- time.get_current_time: Get current time

INSTRUCTIONS:
1. For VIEWING data, use get_* functions (return real data).
2. For CREATE/UPDATE/DELETE, use *_api_call functions (return API structures).
3. Use legacy functions for analysis and suggestions.
4. Parse and convert time expressions to dates.
5. Return data exactly as received from functions.

ALWAYS return valid JSON with appropriate structure."""
    
    def _get_expense_system_prompt(self, language: str) -> str:
        """Get system prompt for expense tracking agent"""
        if language == "fr":
            return """Tu es un assistant IA spécialisé dans le suivi des dépenses pour Devia.

RÔLE: Analyser du texte de reçus ou des descriptions de dépenses pour créer des données de dépense structurées.

FONCTIONS DISPONIBLES:
- expense.extract_expense_from_text: Extraire une dépense du texte
- expense.categorize_expense: Catégoriser la dépense
- expense.calculate_vat: Calculer la TVA
- expense.parse_receipt: Analyser un reçu

INSTRUCTIONS:
1. Analyser le texte pour identifier les informations de dépense
2. Extraire description, montant, date, fournisseur
3. Catégoriser automatiquement la dépense
4. Calculer la TVA si applicable
5. Générer un ID de dépense unique
6. Retourner les données structurées en JSON

TOUJOURS retourner un JSON valide avec la structure de dépense."""
        else:
            return """You are an AI assistant specialized in expense tracking for Devia.

ROLE: Analyze receipt text or expense descriptions to create structured expense data.

AVAILABLE FUNCTIONS:
- expense.extract_expense_from_text: Extract expense from text
- expense.categorize_expense: Categorize the expense
- expense.calculate_vat: Calculate VAT amount
- expense.parse_receipt: Parse receipt information

INSTRUCTIONS:
1. Analyze text to identify expense information
2. Extract description, amount, date, vendor
3. Automatically categorize the expense
4. Calculate VAT if applicable
5. Generate unique expense ID
6. Return structured data as JSON

ALWAYS return valid JSON with expense structure."""
    
    def _get_manual_task_system_prompt(self, language: str) -> str:
        """Get system prompt for manual task processing"""
        return f"""You are an AI assistant for manual task management and planning.
        
LANGUAGE: Use {language} (English/French) for responses.

ROLE: Analyze manual task descriptions to create structured internal task data.

AVAILABLE FUNCTIONS:
- manual_task.extract_task_info: Extract task information from text
- manual_task.set_priority: Set task priority level
- manual_task.set_color: Set task color for visual organization
- manual_task.parse_schedule: Parse date and time information
- manual_task.create_task: Create a new manual task

INSTRUCTIONS:
1. Analyze text to identify task information (title, timing, priority)
2. Extract color information if mentioned (red, blue, green, yellow, etc.)
3. Parse date/time information (tomorrow, specific dates, time ranges)
4. Determine task priority (high, medium, low)
5. Extract duration or time blocks (9-5, 2 hours, etc.)
6. Generate unique task ID
7. Return structured task data as JSON

TASK TYPES:
- Internal planning tasks
- Team coordination
- Personal reminders
- Maintenance tasks
- Color-coded organization tasks

ALWAYS return valid JSON with manual task structure."""
    
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