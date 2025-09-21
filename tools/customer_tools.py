"""
Customer management tools for Semantic Kernel
These tools handle customer data extraction, validation, and management
"""

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import List, Dict, Any, Optional
import json
import re
import uuid
from datetime import datetime
import email_validator
from email_validator import validate_email, EmailNotValidError

from config.settings import Settings
from models import Client, ClientStatus

class CustomerTools:
    """
    Semantic Kernel tools for customer data management
    Provides AI-powered customer information extraction and validation
    """
    
    def __init__(self, settings: Settings):
        """Initialize customer tools with application settings"""
        self.settings = settings
    
    @kernel_function(
        description="Extract customer data from natural language text",
        name="extract_customer_data"
    )
    def extract_customer_data(self, text: str) -> str:
        """
        Extract structured customer information from natural language text
        
        Args:
            text: Natural language text containing customer information
            
        Returns:
            JSON string containing extracted customer data
            
        Example:
            Input: "John Doe from ABC Company, email john@abc.com, phone 555-123-4567, located at 123 Main St, Paris"
            Output: JSON with structured customer data
        """
        try:
            customer_data = {
                "id": str(uuid.uuid4()),
                "name": "",
                "email": "",
                "phone": "",
                "address": "",
                "company": "",
                "balance": 0.0,
                "status": "active",
                "notes": "",
                "created_at": datetime.now().isoformat()
            }
            
            # Extract name
            customer_data["name"] = self._extract_name(text)
            
            # Extract email
            customer_data["email"] = self._extract_email(text)
            
            # Extract phone
            customer_data["phone"] = self._extract_phone(text)
            
            # Extract address
            customer_data["address"] = self._extract_address(text)
            
            # Extract company
            customer_data["company"] = self._extract_company(text)
            
            # Extract notes
            customer_data["notes"] = self._extract_notes(text)
            
            # Validate and clean the data
            customer_data = self._validate_and_clean_customer_data(customer_data)
            
            return json.dumps(customer_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to extract customer data: {str(e)}"})
    
    @kernel_function(
        description="Validate customer information and check for completeness",
        name="validate_customer_info"
    )
    def validate_customer_info(self, customer_json: str) -> str:
        """
        Validate customer information and return validation results
        
        Args:
            customer_json: JSON string containing customer data
            
        Returns:
            JSON string containing validation results and suggestions
        """
        try:
            customer_data = json.loads(customer_json)
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": [],
                "completeness_score": 0
            }
            
            # Validate required fields
            if not customer_data.get("name", "").strip():
                validation_results["errors"].append("Customer name is required")
                validation_results["is_valid"] = False
            
            # Validate email format
            email = customer_data.get("email", "").strip()
            if email:
                try:
                    validate_email(email)
                except EmailNotValidError:
                    validation_results["errors"].append("Invalid email format")
                    validation_results["is_valid"] = False
            else:
                validation_results["warnings"].append("Email address is missing")
            
            # Validate phone format
            phone = customer_data.get("phone", "").strip()
            if phone:
                if not self._is_valid_phone(phone):
                    validation_results["warnings"].append("Phone number format appears invalid")
            else:
                validation_results["warnings"].append("Phone number is missing")
            
            # Check address completeness
            address = customer_data.get("address", "").strip()
            if not address:
                validation_results["warnings"].append("Address is missing")
            elif not self._is_complete_address(address):
                validation_results["suggestions"].append("Address appears incomplete - consider adding city/postal code")
            
            # Calculate completeness score
            fields = ["name", "email", "phone", "address", "company"]
            filled_fields = sum(1 for field in fields if customer_data.get(field, "").strip())
            validation_results["completeness_score"] = int((filled_fields / len(fields)) * 100)
            
            # Add suggestions based on completeness
            if validation_results["completeness_score"] < 60:
                validation_results["suggestions"].append("Consider gathering more customer information for better service")
            
            return json.dumps(validation_results, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to validate customer info: {str(e)}"})
    
    @kernel_function(
        description="Format and standardize address information",
        name="format_address"
    )
    def format_address(self, address_text: str, country: str = "France") -> str:
        """
        Format and standardize address information
        
        Args:
            address_text: Raw address text
            country: Country for address formatting (default: France)
            
        Returns:
            JSON string containing formatted address components
        """
        try:
            formatted_address = {
                "street": "",
                "city": "",
                "postal_code": "",
                "country": country,
                "formatted": ""
            }
            
            # Clean input
            address_text = address_text.strip()
            
            # Extract postal code (French format: 5 digits)
            postal_pattern = r'\\b(\\d{5})\\b'
            postal_match = re.search(postal_pattern, address_text)
            if postal_match:
                formatted_address["postal_code"] = postal_match.group(1)
                address_text = address_text.replace(postal_match.group(0), "").strip()
            
            # Extract common French city names and patterns
            # This is a simplified approach - in production, use a proper address API
            city_patterns = [
                r'\\b(Paris)\\b',
                r'\\b(Lyon)\\b',
                r'\\b(Marseille)\\b',
                r'\\b(Toulouse)\\b',
                r'\\b(Nice)\\b',
                r'\\b(Nantes)\\b',
                r'\\b(Strasbourg)\\b',
                r'\\b(Montpellier)\\b',
                r'\\b(Bordeaux)\\b',
                r'\\b(Lille)\\b',
                r'\\b([A-Z][a-z]+(?:-[A-Z][a-z]+)*)\\b'  # General city pattern
            ]
            
            for pattern in city_patterns:
                city_match = re.search(pattern, address_text, re.IGNORECASE)
                if city_match:
                    formatted_address["city"] = city_match.group(1).title()
                    address_text = address_text.replace(city_match.group(0), "").strip()
                    break
            
            # Remaining text is likely the street address
            # Clean up common separators
            street = re.sub(r'[,;]+', ',', address_text).strip(' ,;')
            formatted_address["street"] = street
            
            # Create formatted address
            parts = [
                formatted_address["street"],
                formatted_address["postal_code"],
                formatted_address["city"],
                formatted_address["country"]
            ]
            formatted_address["formatted"] = ", ".join(part for part in parts if part)
            
            return json.dumps(formatted_address, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to format address: {str(e)}"})
    
    @kernel_function(
        description="Generate a unique customer ID based on name and email",
        name="generate_customer_id"
    )
    def generate_customer_id(self, name: str, email: str = "") -> str:
        """
        Generate a unique customer ID
        
        Args:
            name: Customer name
            email: Customer email (optional)
            
        Returns:
            Unique customer ID string
        """
        try:
            # Create a base ID from name
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
            
            if email:
                email_part = email.split('@')[0].lower()
                clean_email = re.sub(r'[^a-zA-Z0-9]', '', email_part)
                base_id = f"{clean_name}_{clean_email}"
            else:
                base_id = clean_name
            
            # Add timestamp suffix for uniqueness
            timestamp_suffix = int(datetime.now().timestamp()) % 10000
            
            customer_id = f"CUST_{base_id}_{timestamp_suffix:04d}".upper()
            
            # Ensure reasonable length
            if len(customer_id) > 50:
                customer_id = customer_id[:50]
            
            return customer_id
            
        except Exception as e:
            return f"CUST_{uuid.uuid4().hex[:8].upper()}"
    
    @kernel_function(
        description="Extract and categorize customer preferences and notes from text",
        name="extract_customer_preferences"
    )
    def extract_customer_preferences(self, text: str) -> str:
        """
        Extract customer preferences, requirements, and notes from text
        
        Args:
            text: Text containing customer information and preferences
            
        Returns:
            JSON string containing categorized preferences and notes
        """
        try:
            preferences = {
                "communication": {
                    "preferred_method": "",
                    "language": "fr",  # Default to French
                    "best_time": ""
                },
                "business": {
                    "industry": "",
                    "size": "",
                    "urgency": "normal"
                },
                "service": {
                    "previous_client": False,
                    "referral_source": "",
                    "special_requirements": []
                },
                "billing": {
                    "payment_terms": "30",  # Default 30 days
                    "billing_contact": "",
                    "vat_number": ""
                },
                "notes": ""
            }
            
            # Extract communication preferences
            if any(word in text.lower() for word in ["email", "e-mail", "mail"]):
                preferences["communication"]["preferred_method"] = "email"
            elif any(word in text.lower() for word in ["phone", "call", "telephone"]):
                preferences["communication"]["preferred_method"] = "phone"
            
            # Extract language preference
            if any(word in text.lower() for word in ["english", "anglais"]):
                preferences["communication"]["language"] = "en"
            
            # Extract time preferences
            time_patterns = [
                r"(?:morning|matin)",
                r"(?:afternoon|après-midi)",
                r"(?:evening|soir)",
                r"(?:weekday|semaine)",
                r"(?:weekend|week-end)"
            ]
            
            for pattern in time_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    preferences["communication"]["best_time"] = re.search(pattern, text, re.IGNORECASE).group(0)
                    break
            
            # Extract business information
            industry_keywords = {
                "technology": ["tech", "software", "IT", "digital", "web"],
                "healthcare": ["health", "medical", "clinic", "hospital"],
                "finance": ["bank", "finance", "insurance", "accounting"],
                "retail": ["retail", "shop", "store", "commerce"],
                "construction": ["construction", "building", "renovation"],
                "consulting": ["consulting", "conseil", "advisory"]
            }
            
            for industry, keywords in industry_keywords.items():
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    preferences["business"]["industry"] = industry
                    break
            
            # Extract urgency
            if any(word in text.lower() for word in ["urgent", "asap", "immediately", "rush"]):
                preferences["business"]["urgency"] = "high"
            elif any(word in text.lower() for word in ["flexible", "no rush", "when possible"]):
                preferences["business"]["urgency"] = "low"
            
            # Extract special requirements
            special_req_patterns = [
                r"(?:require|need|must have)\\s+([^,\\.;]+)",
                r"(?:important|critical|essential)\\s*:?\\s*([^,\\.;]+)",
                r"(?:special|specific)\\s+(?:requirement|need)\\s*:?\\s*([^,\\.;]+)"
            ]
            
            for pattern in special_req_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    requirement = match.group(1).strip()
                    if requirement and len(requirement) > 5:
                        preferences["service"]["special_requirements"].append(requirement)
            
            # Extract payment/billing information
            if "vat" in text.lower() or "tva" in text.lower():
                vat_pattern = r"(?:vat|tva)\\s*:?\\s*([A-Z0-9]+)"
                vat_match = re.search(vat_pattern, text, re.IGNORECASE)
                if vat_match:
                    preferences["billing"]["vat_number"] = vat_match.group(1)
            
            # Extract payment terms
            payment_patterns = [
                r"(?:payment|paiement)\\s+(?:in|dans)\\s+(\\d+)\\s*(?:days|jours)",
                r"(\\d+)\\s*(?:days|jours)\\s+(?:payment|paiement)"
            ]
            
            for pattern in payment_patterns:
                payment_match = re.search(pattern, text, re.IGNORECASE)
                if payment_match:
                    preferences["billing"]["payment_terms"] = payment_match.group(1)
                    break
            
            # Extract referral information
            referral_patterns = [
                r"(?:referred by|recommended by|suggested by)\\s+([^,\\.;]+)",
                r"(?:heard about|found)\\s+(?:us|you)\\s+(?:through|via)\\s+([^,\\.;]+)"
            ]
            
            for pattern in referral_patterns:
                referral_match = re.search(pattern, text, re.IGNORECASE)
                if referral_match:
                    preferences["service"]["referral_source"] = referral_match.group(1).strip()
                    break
            
            # Check if previous client
            if any(phrase in text.lower() for phrase in ["previous client", "worked before", "past project"]):
                preferences["service"]["previous_client"] = True
            
            # Extract general notes
            preferences["notes"] = self._extract_notes(text)
            
            return json.dumps(preferences, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to extract preferences: {str(e)}"})
    
    def _extract_name(self, text: str) -> str:
        """Extract person name from text"""
        # Patterns for names
        name_patterns = [
            r"(?:name|nom)\\s*:?\\s*([A-Z][a-z]+\\s+[A-Z][a-z]+)",  # "Name: John Doe"
            r"(?:je suis|i am|my name is)\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)",  # "I am John Doe"
            r"^([A-Z][a-z]+\\s+[A-Z][a-z]+)",  # Name at start of text
            r"([A-Z][a-z]+\\s+[A-Z][a-z]+)(?:\\s+from|\\s+at|\\s+with)",  # "John Doe from ABC"
            r"(?:monsieur|madame|mr|mrs|ms)\\.?\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)"  # "Mr. John Doe"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return ""
    
    def _extract_email(self, text: str) -> str:
        """Extract email address from text"""
        email_pattern = r'\\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})\\b'
        match = re.search(email_pattern, text)
        return match.group(1).lower() if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        # French phone number patterns
        phone_patterns = [
            r'(?:phone|tel|telephone|mobile)\\s*:?\\s*([+\\d\\s\\-\\.\\(\\)]{10,})',
            r'\\b(\\+33\\s*[1-9](?:[\\s\\-\\.]*\\d){8})\\b',  # French international
            r'\\b(0[1-9](?:[\\s\\-\\.]*\\d){8})\\b',  # French national
            r'\\b([\\d\\s\\-\\.\\(\\)]{10,})\\b'  # General pattern
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phone = match.group(1).strip()
                # Clean up phone number
                phone = re.sub(r'[^\\d\\+]', ' ', phone)
                phone = re.sub(r'\\s+', ' ', phone).strip()
                return phone
        
        return ""
    
    def _extract_address(self, text: str) -> str:
        """Extract address from text"""
        address_patterns = [
            r'(?:address|adresse)\\s*:?\\s*([^,\\.;]+(?:street|st|avenue|ave|road|rd|drive|dr|rue|boulevard|bd)[^,\\.;]*)',
            r'(?:at|à|located at|situé)\\s+([^,\\.;]*\\d+[^,\\.;]*(?:street|st|avenue|ave|road|rd|drive|dr|rue|boulevard|bd)[^,\\.;]*)',
            r'(\\d+[^,\\.;]*(?:street|st|avenue|ave|road|rd|drive|dr|rue|boulevard|bd)[^,\\.;]*)',
            r'(?:address|adresse)\\s*:?\\s*([^,\\.;]+,\\s*[^,\\.;]+)'  # General address with comma
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                # Clean up address
                address = re.sub(r'\\s+', ' ', address)
                return address
        
        return ""
    
    def _extract_company(self, text: str) -> str:
        """Extract company name from text"""
        company_patterns = [
            r'(?:company|entreprise|societe|firm)\\s*:?\\s*([^,\\.;]+)',
            r'(?:from|de|at|chez)\\s+([^,\\.;]+(?:company|corp|inc|ltd|llc|sarl|sas|sa))',
            r'([^,\\.;]+(?:company|corp|inc|ltd|llc|sarl|sas|sa))',
            r'(?:work at|working at|employed by)\\s+([^,\\.;]+)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                # Clean up company name
                company = re.sub(r'\\s+', ' ', company)
                return company.title()
        
        return ""
    
    def _extract_notes(self, text: str) -> str:
        """Extract notes or additional information from text"""
        # Look for note indicators
        note_patterns = [
            r'(?:note|notes|comment|commentaire|additional|supplémentaire)\\s*:?\\s*([^,\\.;]+)',
            r'(?:important|special|particulier)\\s*:?\\s*([^,\\.;]+)'
        ]
        
        for pattern in note_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no specific notes found, return part of the original text as context
        if len(text) > 200:
            return text[:200] + "..."
        
        return text
    
    def _validate_and_clean_customer_data(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean customer data"""
        # Clean whitespace
        for key, value in customer_data.items():
            if isinstance(value, str):
                customer_data[key] = value.strip()
        
        # Validate email format
        email = customer_data.get("email", "")
        if email:
            try:
                valid_email = validate_email(email)
                customer_data["email"] = valid_email.email
            except EmailNotValidError:
                customer_data["email"] = ""
        
        # Clean phone number
        phone = customer_data.get("phone", "")
        if phone:
            # Remove extra characters and normalize
            clean_phone = re.sub(r'[^\\d\\+\\s\\-\\(\\)]', '', phone)
            customer_data["phone"] = clean_phone.strip()
        
        return customer_data
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Check if phone number appears valid"""
        # Remove all non-digit characters except +
        digits_only = re.sub(r'[^\\d]', '', phone)
        
        # Check length (should be between 8 and 15 digits)
        if len(digits_only) < 8 or len(digits_only) > 15:
            return False
        
        # Check for French patterns
        if phone.startswith('+33') and len(digits_only) == 11:
            return True
        elif phone.startswith('0') and len(digits_only) == 10:
            return True
        elif len(digits_only) >= 10:
            return True
        
        return False
    
    def _is_complete_address(self, address: str) -> bool:
        """Check if address appears complete"""
        # Look for indicators of a complete address
        has_number = bool(re.search(r'\\d+', address))
        has_street = bool(re.search(r'(?:street|st|avenue|ave|road|rd|rue|boulevard|bd)', address, re.IGNORECASE))
        has_city = len(address.split(',')) > 1 or bool(re.search(r'\\b[A-Z][a-z]+\\b', address))
        
        return has_number and (has_street or has_city)