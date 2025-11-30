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
    
    async def process_invoice_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process invoice generation request using AI agent
        
        Args:
            prompt: Natural language description of the invoice to generate
            context: Optional context data (client_id, quote_id, etc.)
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "invoice")
            
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
    
    async def process_customer_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process customer data extraction request using AI agent
        
        Args:
            prompt: Natural language text containing customer information
            context: Optional context data
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "customer")
            
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
    
    async def process_quote_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process quote generation request using AI agent
        
        Args:
            prompt: Natural language description of the quote to generate
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "quote")
            
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
    
    async def process_job_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process job scheduling request using AI agent
        
        Args:
            prompt: Natural language description of the job to schedule
            context: Optional context data (client_id, etc.)
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "job")
            
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
    
    async def process_expense_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process expense tracking request using AI agent
        
        Args:
            prompt: Natural language description of the expense or receipt text
            context: Optional context data (receipt_text, etc.)
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "expense")
            
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

    async def process_manual_task_request(self, prompt: str, context: Optional[Dict[str, Any]] = None, language: str = "en") -> Dict[str, Any]:
        """
        Process manual task creation request using AI agent
        
        Args:
            prompt: Natural language description of the manual task
            context: Optional context data
            language: Response language (en/fr)
        
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
            result = await self._execute_agent_request(system_prompt, full_prompt, "manual_task")
            
            return {
                "success": True,
                "message": "Manual task created successfully" if language == "en" else "Tâche manuelle créée avec succès",
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Manual task processing failed: {e}")
            return {
                "success": False,
                "message": f"Failed to process manual task: {str(e)}",
                "errors": [str(e)]
            }
    
    async def _execute_agent_request(self, system_prompt: str, user_prompt: str, agent_type: str) -> Dict[str, Any]:
        """
        Execute an agent request using Semantic Kernel
        
        Args:
            system_prompt: System instructions for the AI
            user_prompt: User's request with context
            agent_type: Type of agent (invoice, customer, quote, job, expense)
        
        Returns:
            Parsed result from the AI agent
        """
        if not self.is_initialized():
            raise RuntimeError("Semantic Kernel service is not initialized")
        
        # Create chat history
        chat_history = ChatHistory()
        chat_history.add_system_message(system_prompt)
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

NOUVEAUX CHAMPS COMPLETS DE FACTURE SUPPORTÉS:
- Informations Client: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Détails Projet: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Types de Facture: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Système de Remise: montant remise, discountType (FIXED/PERCENTAGE)
- Système d'Acompte: montant acompte, downPaymentType (FIXED/PERCENTAGE)
- Notes Multiples: notes (générales), internalNotes (privées), publicNotes (visibles client)
- Signatures Numériques: contractorSignature, clientSignature (encodées base64)
- Champs Améliorés: quoteId (pour conversion devis), userId

INSTRUCTIONS:
1. Analyser la demande pour identifier toutes les informations de facturation et projet.
2. Extraire les données client complètes incluant le type d'entreprise (individuel vs société).
3. Identifier les détails projet (nom, composants d'adresse complète incluant code postal et ville).
4. Déterminer le type de facture depuis le contexte (finale, acompte, avance, avoir).
5. Analyser les informations de remise incluant le type (pourcentage vs montant fixe).
6. Extraire les détails d'acompte si mentionnés (pourcentage ou montant fixe).
7. Catégoriser les notes en internes (privées), publiques (visibles client), ou générales.
8. Calculer les totaux en considérant remises, acomptes, et TVA.
9. Lors d'appels get_*, TOUJOURS inclure le paramètre user_id depuis le contexte.
10. Retourner les données structurées en JSON, prêtes pour l'API complète.

TOUJOURS retourner un JSON valide avec la structure de facture améliorée supportant tous les nouveaux champs."""
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

NEW COMPREHENSIVE INVOICE FIELDS SUPPORTED:
- Client Information: clientName, clientEmail, clientCompanyType (COMPANY/INDIVIDUAL)
- Project Details: title, projectName, projectAddress, projectStreetAddress, projectZipCode, projectCity
- Invoice Types: invoiceType (FINAL/INTERIM/ADVANCE/CREDIT)
- Discount System: discount amount, discountType (FIXED/PERCENTAGE)
- Down Payment System: downPayment amount, downPaymentType (FIXED/PERCENTAGE)
- Multiple Notes: notes (general), internalNotes (private), publicNotes (client-visible)
- Digital Signatures: contractorSignature, clientSignature (base64 encoded)
- Enhanced Fields: quoteId (for quote conversion), userId

INSTRUCTIONS:
1. Analyze the request to identify all billing and project information.
2. Extract comprehensive client data including company type (individual vs company).
3. Identify project details (name, full address components including ZIP and city).
4. Determine invoice type from context (final, interim, advance payment, credit).
5. Parse discount information including type (percentage vs fixed amount).
6. Extract down payment details if mentioned (percentage or fixed amount).
7. Categorize notes into internal (private), public (client-visible), or general.
8. Calculate totals considering discounts, down payments, and VAT.
9. When calling get_* functions, ALWAYS include the user_id parameter from context.
10. Return structured data as JSON, ready for the comprehensive API.

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
- customer.extract_customer_preferences: Extraire préférences et notes du client

INSTRUCTIONS:
1. Analyser le texte pour identifier les informations client.
2. Extraire nom, email, téléphone, adresse, entreprise.
3. Valider et formater les données.
4. Générer un ID unique si nécessaire.
5. Ajouter les préférences client si disponibles.
6. Retourner les données structurées en JSON.

TOUJOURS retourner un JSON valide avec la structure client."""
        else:
            return """You are an AI assistant specialized in customer data extraction and management for Devia.

ROLE: Analyze natural language text to extract structured customer information.

AVAILABLE FUNCTIONS:
- customer.extract_customer_data: Extract customer data from text
- customer.validate_customer_info: Validate customer information
- customer.format_address: Format address information
- customer.generate_customer_id: Generate unique customer ID
- customer.extract_customer_preferences: Extract customer preferences and notes

INSTRUCTIONS:
1. Analyze text to identify customer information.
2. Extract name, email, phone, address, company.
3. Validate and format the data.
4. Generate unique ID if needed.
5. Include customer preferences when available.
6. Return structured data as JSON.

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

LEGACY FUNCTIONS (toujours disponibles):
- job.create_job_from_text: Créer un travail à partir du texte
- job.parse_schedule_info: Analyser les informations de planification
- job.validate_schedule: Valider le planning
- job.suggest_optimal_times: Suggérer des créneaux optimaux
- job.reschedule_job: Replanifier un travail existant

CARACTÉRISTIQUES DES JOBS CLIENTS:
- Toujours associés à un client spécifique
- Travail facturable et professionnel
- Rendez-vous et services clients
- Pas de codage couleur (contrairement aux tâches manuelles)

INSTRUCTIONS:
1. Pour CONSULTER des données, utiliser les fonctions get_* (retournent données réelles).
2. Pour CRÉER/MODIFIER/SUPPRIMER, utiliser les fonctions *_api_call (retournent structure API).
3. Utiliser les fonctions legacy pour analyse et suggestions.
4. Lors de l'appel des fonctions get_*, TOUJOURS inclure le paramètre user_id du contexte pour filtrer les données par utilisateur.
5. Retourner les données exactement comme reçues des fonctions.

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

LEGACY FUNCTIONS (still available):
- job.create_job_from_text: Create a job from natural language description
- job.parse_schedule_info: Parse scheduling information
- job.validate_schedule: Validate schedule feasibility
- job.suggest_optimal_times: Suggest optimal time slots
- job.reschedule_job: Reschedule an existing job

CLIENT JOB CHARACTERISTICS:
- Always associated with a specific client
- Billable and professional work
- Client appointments and services
- No color coding (unlike manual tasks)

INSTRUCTIONS:
1. For VIEWING data, use get_* functions (return real data).
2. For CREATE/UPDATE/DELETE, use *_api_call functions (return API structures).
3. Use legacy functions for analysis and suggestions.
4. When calling get_* functions, ALWAYS include the user_id parameter from the context to filter data by user.
5. Return data exactly as received from functions.

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

INSTRUCTIONS:
1. Analyser le texte pour identifier description, montant, date, fournisseur (ou utiliser expense.parse_receipt pour un reçu).
2. Catégoriser la dépense via expense.categorize_expense si nécessaire.
3. Calculer la TVA si applicable avec expense.calculate_vat.
4. Retourner les données structurées en JSON, prêtes pour l'API.

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

INSTRUCTIONS:
1. Analyze the text to identify description, amount, date, and vendor (or use expense.parse_receipt for receipts).
2. Categorize the expense with expense.categorize_expense when helpful.
3. Calculate VAT if applicable using expense.calculate_vat.
4. Return structured data as JSON, ready for the API.

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

CARACTÉRISTIQUES DES TÂCHES MANUELLES:
- Codage couleur pour l'organisation visuelle (rouge, bleu, vert, jaune, etc.)
- Planification interne et rappels personnels
- Coordination d'équipe et réunions internes
- Pas de facturation client (contrairement aux "jobs")

INSTRUCTIONS:
1. Analyser le texte pour identifier titre, dates de début/fin, couleur, lieu, notes.
2. Extraire les détails comme couleur (ex: rouge, bleu), durée, lieu du travail.
3. Retourner les données structurées en JSON pour l'API des tâches manuelles.
4. Les tâches manuelles sont destinées à la planification interne, pas à la facturation cliente.

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

MANUAL TASK CHARACTERISTICS:
- Color coding for visual organization (red, blue, green, yellow, etc.)
- Internal planning and personal reminders
- Team coordination and internal meetings
- No client billing (unlike "jobs")

INSTRUCTIONS:
1. Analyze the text to identify title, start/end times, color, location, and notes.
2. Extract details like color (e.g., red, blue), duration, work location.
3. Return structured data as JSON, ready for the manual task API.
4. Manual tasks are for internal planning only, not for client billing.

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