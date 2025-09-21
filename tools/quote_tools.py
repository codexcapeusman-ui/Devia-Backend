"""
Quote management tools for Semantic Kernel
These tools handle quote generation, modification, and conversion
"""

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import List, Dict, Any, Optional
import json
import re
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from config.settings import Settings
from models import Quote, QuoteItem, Client, QuoteStatus, ItemType

class QuoteTools:
    """
    Semantic Kernel tools for quote generation and management
    Provides AI-powered quote creation from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize quote tools with application settings"""
        self.settings = settings
        self.default_vat_rate = settings.default_vat_rate
        self.company_name = settings.company_name
        self.currency = settings.default_currency
    
    @kernel_function(
        description="Generate a complete quote from natural language text description",
        name="generate_quote_from_text"
    )
    def generate_quote_from_text(self, description: str, client_id: Optional[str] = None, validity_days: int = 30) -> str:
        """
        Generate a complete quote from text description
        
        Args:
            description: Natural language description of the quote
            client_id: Optional client ID if known
            validity_days: Number of days the quote remains valid (default: 30)
            
        Returns:
            JSON string containing the generated quote data
            
        Example:
            Input: "Quote for website redesign including 3 pages, logo design, and 6 months hosting for ABC Company"
            Output: JSON with complete quote structure
        """
        try:
            # Extract quote items from description
            items = self._extract_quote_items_from_description(description)
            
            # Extract client information if not provided
            if not client_id:
                client_data = self._extract_client_from_description(description)
            else:
                # In a real implementation, you would fetch client data from database
                client_data = {"id": client_id, "name": "Client", "email": "", "phone": "", "address": ""}
            
            # Calculate totals
            subtotal = sum(item["total"] for item in items)
            discount = self._extract_discount_from_description(description)
            vat_amount = (subtotal - discount) * (self.default_vat_rate / 100)
            total = subtotal - discount + vat_amount
            
            # Generate quote number
            quote_number = self._generate_quote_number()
            
            # Set validity period
            valid_until = datetime.now() + timedelta(days=validity_days)
            
            # Create quote structure
            quote_data = {
                "id": str(uuid.uuid4()),
                "client_id": client_data.get("id", str(uuid.uuid4())),
                "client": client_data,
                "number": quote_number,
                "items": items,
                "subtotal": round(subtotal, 2),
                "discount": round(discount, 2),
                "vat_rate": self.default_vat_rate,
                "vat_amount": round(vat_amount, 2),
                "total": round(total, 2),
                "status": "draft",
                "created_at": datetime.now().isoformat(),
                "valid_until": valid_until.isoformat(),
                "notes": self._extract_notes_from_description(description)
            }
            
            return json.dumps(quote_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to generate quote: {str(e)}"})
    
    @kernel_function(
        description="Extract quote items (services, products, labor) from text description",
        name="extract_quote_items"
    )
    def extract_quote_items(self, description: str) -> str:
        """
        Extract quote items from natural language description
        
        Args:
            description: Text containing item descriptions, quantities, and prices
            
        Returns:
            JSON string containing list of extracted quote items
            
        Example:
            Input: "3 page website design €2000, logo creation €500, 6 months hosting €300"
            Output: JSON array with quote item objects
        """
        try:
            items = self._extract_quote_items_from_description(description)
            return json.dumps(items, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to extract quote items: {str(e)}"})
    
    @kernel_function(
        description="Calculate quote totals with VAT, discounts, and final amount",
        name="calculate_quote_totals"
    )
    def calculate_quote_totals(self, items_json: str, discount: float = 0.0, vat_rate: Optional[float] = None) -> str:
        """
        Calculate totals for quote items with VAT and discounts
        
        Args:
            items_json: JSON string containing array of items with quantities and prices
            discount: Discount amount to apply
            vat_rate: VAT rate to use (defaults to company default)
            
        Returns:
            JSON string containing calculated totals
        """
        try:
            items = json.loads(items_json)
            if not isinstance(items, list):
                raise ValueError("Items must be an array")
            
            # Calculate subtotal
            subtotal = sum(item.get("total", 0) for item in items)
            
            # Apply discount
            discounted_subtotal = subtotal - discount
            
            # Calculate VAT
            vat_rate = vat_rate or self.default_vat_rate
            vat_amount = discounted_subtotal * (vat_rate / 100)
            
            # Calculate total
            total = discounted_subtotal + vat_amount
            
            result = {
                "subtotal": round(subtotal, 2),
                "discount": round(discount, 2),
                "discounted_subtotal": round(discounted_subtotal, 2),
                "vat_rate": vat_rate,
                "vat_amount": round(vat_amount, 2),
                "total": round(total, 2),
                "currency": self.currency
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to calculate quote totals: {str(e)}"})
    
    @kernel_function(
        description="Set quote validity period based on project type and complexity",
        name="set_validity_period"
    )
    def set_validity_period(self, quote_json: str, project_type: str = "", complexity: str = "medium") -> str:
        """
        Set appropriate validity period for a quote based on project characteristics
        
        Args:
            quote_json: JSON string containing quote data
            project_type: Type of project (web, software, consulting, etc.)
            complexity: Project complexity (simple, medium, complex)
            
        Returns:
            JSON string containing updated quote with validity period
        """
        try:
            quote_data = json.loads(quote_json)
            
            # Determine validity period based on project characteristics
            validity_days = 30  # Default
            
            # Adjust based on project type
            project_type_days = {
                "web": 45,
                "website": 45,
                "software": 60,
                "app": 60,
                "consulting": 30,
                "design": 30,
                "development": 60,
                "maintenance": 90,
                "hosting": 15,
                "simple": 15
            }
            
            for ptype, days in project_type_days.items():
                if ptype.lower() in project_type.lower():
                    validity_days = days
                    break
            
            # Adjust based on complexity
            complexity_multipliers = {
                "simple": 0.7,
                "medium": 1.0,
                "complex": 1.5,
                "enterprise": 2.0
            }
            
            multiplier = complexity_multipliers.get(complexity.lower(), 1.0)
            validity_days = int(validity_days * multiplier)
            
            # Set minimum and maximum bounds
            validity_days = max(14, min(validity_days, 120))
            
            # Update quote data
            quote_data["valid_until"] = (datetime.now() + timedelta(days=validity_days)).isoformat()
            quote_data["validity_days"] = validity_days
            
            # Add note about validity
            existing_notes = quote_data.get("notes", "")
            validity_note = f"This quote is valid for {validity_days} days from the issue date."
            
            if existing_notes:
                quote_data["notes"] = f"{existing_notes}\\n\\n{validity_note}"
            else:
                quote_data["notes"] = validity_note
            
            return json.dumps(quote_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to set validity period: {str(e)}"})
    
    @kernel_function(
        description="Apply discount to quote based on client type, project size, or special conditions",
        name="apply_quote_discount"
    )
    def apply_quote_discount(self, quote_json: str, discount_type: str = "amount", discount_value: float = 0.0, reason: str = "") -> str:
        """
        Apply discount to a quote
        
        Args:
            quote_json: JSON string containing quote data
            discount_type: Type of discount ("amount" or "percentage")
            discount_value: Discount value (amount in currency or percentage)
            reason: Reason for the discount
            
        Returns:
            JSON string containing updated quote with discount applied
        """
        try:
            quote_data = json.loads(quote_json)
            
            subtotal = quote_data.get("subtotal", 0)
            current_discount = quote_data.get("discount", 0)
            
            # Calculate discount amount
            if discount_type.lower() == "percentage":
                discount_amount = subtotal * (discount_value / 100)
            else:
                discount_amount = discount_value
            
            # Apply discount
            new_discount = current_discount + discount_amount
            
            # Ensure discount doesn't exceed subtotal
            new_discount = min(new_discount, subtotal * 0.5)  # Max 50% discount
            
            # Recalculate totals
            discounted_subtotal = subtotal - new_discount
            vat_rate = quote_data.get("vat_rate", self.default_vat_rate)
            vat_amount = discounted_subtotal * (vat_rate / 100)
            total = discounted_subtotal + vat_amount
            
            # Update quote data
            quote_data["discount"] = round(new_discount, 2)
            quote_data["vat_amount"] = round(vat_amount, 2)
            quote_data["total"] = round(total, 2)
            
            # Add discount note
            discount_note = f"Discount applied: {discount_value}"
            if discount_type.lower() == "percentage":
                discount_note += "%"
            else:
                discount_note += f" {self.currency}"
            
            if reason:
                discount_note += f" ({reason})"
            
            existing_notes = quote_data.get("notes", "")
            if existing_notes:
                quote_data["notes"] = f"{existing_notes}\\n\\n{discount_note}"
            else:
                quote_data["notes"] = discount_note
            
            return json.dumps(quote_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to apply discount: {str(e)}"})
    
    @kernel_function(
        description="Generate quote variations with different options or packages",
        name="generate_quote_variations"
    )
    def generate_quote_variations(self, base_quote_json: str, variation_types: str = "basic,premium") -> str:
        """
        Generate quote variations (basic, premium, enterprise packages)
        
        Args:
            base_quote_json: JSON string containing base quote data
            variation_types: Comma-separated list of variation types
            
        Returns:
            JSON string containing array of quote variations
        """
        try:
            base_quote = json.loads(base_quote_json)
            variations = []
            
            variation_list = [v.strip().lower() for v in variation_types.split(",")]
            
            for variation_type in variation_list:
                variation_quote = base_quote.copy()
                variation_quote["id"] = str(uuid.uuid4())
                variation_quote["number"] = self._generate_quote_number(suffix=variation_type.upper())
                
                # Modify items and pricing based on variation type
                if variation_type == "basic":
                    # Basic package - remove premium features, reduce quantities
                    variation_quote["items"] = self._create_basic_variation(base_quote["items"])
                    variation_quote["notes"] = f"Basic package - {base_quote.get('notes', '')}"
                    
                elif variation_type == "premium":
                    # Premium package - add extra features, increase quantities
                    variation_quote["items"] = self._create_premium_variation(base_quote["items"])
                    variation_quote["notes"] = f"Premium package with additional features - {base_quote.get('notes', '')}"
                    
                elif variation_type == "enterprise":
                    # Enterprise package - comprehensive solution
                    variation_quote["items"] = self._create_enterprise_variation(base_quote["items"])
                    variation_quote["notes"] = f"Enterprise package with full service - {base_quote.get('notes', '')}"
                
                # Recalculate totals
                self._recalculate_quote_totals(variation_quote)
                
                variations.append(variation_quote)
            
            return json.dumps(variations, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to generate quote variations: {str(e)}"})
    
    def _extract_quote_items_from_description(self, description: str) -> List[Dict[str, Any]]:
        """
        Internal method to extract quote items from description using regex patterns
        """
        items = []
        
        # Common patterns for quote items
        patterns = [
            # Pattern: "website design €2000" or "logo creation €500"
            r'([^,\\.;]+?)\\s*[€$£](\\d+(?:\\.\\d+)?)',
            # Pattern: "3 pages at €200 each" or "5 hours at €50/hour"
            r'(\\d+)\\s+([^@]+?)\\s*(?:at|@)\\s*[€$£]?(\\d+(?:\\.\\d+)?)(?:/[a-z]+|\\s*each)?',
            # Pattern: "design services for €1500"
            r'([^,\\.;]+?)\\s*for\\s*[€$£](\\d+(?:\\.\\d+)?)',
            # Pattern: "hosting (6 months) €300"
            r'([^,\\.;]+?)\\s*\\([^)]+\\)\\s*[€$£](\\d+(?:\\.\\d+)?)',
            # Pattern: "maintenance: €100/month"
            r'([^,\\.;:]+):\\s*[€$£](\\d+(?:\\.\\d+)?)(?:/month|/year)?'
        ]
        
        item_id = 1
        
        for pattern in patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) == 2:
                        # Simple item with description and price
                        description_part = match.group(1).strip()
                        price = float(match.group(2))
                        quantity = 1.0
                        
                        # Check if description contains quantity information
                        qty_match = re.search(r'(\\d+)\\s*(page|hour|item|piece)', description_part, re.IGNORECASE)
                        if qty_match:
                            quantity = float(qty_match.group(1))
                            unit_price = price / quantity
                        else:
                            unit_price = price
                    
                    elif len(match.groups()) == 3:
                        # Item with quantity, description, and price
                        quantity = float(match.group(1))
                        description_part = match.group(2).strip()
                        unit_price = float(match.group(3))
                        price = quantity * unit_price
                    
                    else:
                        continue
                    
                    # Clean up description
                    description_part = description_part.strip(' -.,;:')
                    if not description_part:
                        description_part = "Service"
                    
                    # Determine item type based on description
                    item_type = self._determine_item_type(description_part)
                    
                    item = {
                        "id": str(item_id),
                        "description": description_part.title(),
                        "quantity": quantity,
                        "unit_price": round(unit_price, 2),
                        "total": round(price, 2),
                        "type": item_type
                    }
                    
                    items.append(item)
                    item_id += 1
                    
                except (ValueError, IndexError):
                    continue
        
        # If no items found, try to create a general item
        if not items:
            # Look for total amount
            total_pattern = r'(?:total|amount|price)[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)'
            total_match = re.search(total_pattern, description, re.IGNORECASE)
            if total_match:
                total_amount = float(total_match.group(1))
                items.append({
                    "id": "1",
                    "description": "Service",
                    "quantity": 1.0,
                    "unit_price": total_amount,
                    "total": total_amount,
                    "type": "service"
                })
            else:
                # Create a placeholder item
                items.append({
                    "id": "1",
                    "description": "Service to be defined",
                    "quantity": 1.0,
                    "unit_price": 0.0,
                    "total": 0.0,
                    "type": "service"
                })
        
        return items
    
    def _determine_item_type(self, description: str) -> str:
        """Determine item type based on description"""
        description_lower = description.lower()
        
        # Material indicators
        material_keywords = ["material", "equipment", "hardware", "software license", "tool", "product"]
        if any(keyword in description_lower for keyword in material_keywords):
            return "material"
        
        # Labor indicators
        labor_keywords = ["hour", "development", "programming", "coding", "installation", "setup", "work"]
        if any(keyword in description_lower for keyword in labor_keywords):
            return "labor"
        
        # Default to service
        return "service"
    
    def _extract_client_from_description(self, description: str) -> Dict[str, Any]:
        """Extract client information from description"""
        client_data = {
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
        
        # Extract client/company name
        client_patterns = [
            r'(?:for|to|client)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+(?:company|corp|inc|ltd)',
            r'(?:company|corp)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in client_patterns:
            match = re.search(pattern, description)
            if match:
                name = match.group(1).strip()
                if "company" in description.lower() or "corp" in description.lower():
                    client_data["company"] = name
                    client_data["name"] = name
                else:
                    client_data["name"] = name
                break
        
        return client_data
    
    def _extract_discount_from_description(self, description: str) -> float:
        """Extract discount amount from description"""
        discount_patterns = [
            r'discount[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)',
            r'(?:less|minus)[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)',
            r'[€$£]?(\\d+(?:\\.\\d+)?)\\s*(?:discount|off|reduction)',
            r'(\\d+(?:\\.\\d+)?)%\\s*(?:discount|off|reduction)'
        ]
        
        for pattern in discount_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # If it looks like a percentage and is reasonable
                if "%" in pattern and value <= 50:
                    return 0  # Return 0 for now, percentage will be handled separately
                return value
        
        return 0.0
    
    def _extract_notes_from_description(self, description: str) -> str:
        """Extract notes or additional information from description"""
        note_patterns = [
            r'(?:note|notes|comment|special)[:\\s]*([^,\\.;]+)',
            r'(?:includes?|include)[:\\s]*([^,\\.;]+)',
            r'(?:additional|extra)[:\\s]*([^,\\.;]+)'
        ]
        
        notes = []
        for pattern in note_patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                note = match.group(1).strip()
                if note and len(note) > 3:
                    notes.append(note)
        
        return "; ".join(notes) if notes else ""
    
    def _generate_quote_number(self, suffix: str = "") -> str:
        """Generate a unique quote number"""
        current_year = datetime.now().year
        timestamp_suffix = int(datetime.now().timestamp()) % 10000
        
        base_number = f"QUO-{current_year}-{timestamp_suffix:04d}"
        
        if suffix:
            base_number += f"-{suffix}"
        
        return base_number
    
    def _create_basic_variation(self, base_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create basic package variation"""
        basic_items = []
        
        for item in base_items:
            basic_item = item.copy()
            basic_item["id"] = str(uuid.uuid4())
            
            # Reduce quantity or price for basic package
            if "premium" in item["description"].lower() or "advanced" in item["description"].lower():
                # Skip premium features
                continue
            
            # Reduce quantities by 20%
            if basic_item["quantity"] > 1:
                basic_item["quantity"] = max(1, basic_item["quantity"] * 0.8)
                basic_item["total"] = basic_item["quantity"] * basic_item["unit_price"]
            
            # Add "Basic" prefix to description
            if not basic_item["description"].startswith("Basic"):
                basic_item["description"] = f"Basic {basic_item['description']}"
            
            basic_items.append(basic_item)
        
        return basic_items
    
    def _create_premium_variation(self, base_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create premium package variation"""
        premium_items = []
        
        for item in base_items:
            premium_item = item.copy()
            premium_item["id"] = str(uuid.uuid4())
            
            # Increase quantity or add premium features
            if premium_item["quantity"] > 1:
                premium_item["quantity"] = premium_item["quantity"] * 1.3
            else:
                premium_item["unit_price"] = premium_item["unit_price"] * 1.2
            
            premium_item["total"] = premium_item["quantity"] * premium_item["unit_price"]
            
            # Add "Premium" prefix to description
            if not premium_item["description"].startswith("Premium"):
                premium_item["description"] = f"Premium {premium_item['description']}"
            
            premium_items.append(premium_item)
        
        # Add additional premium services
        premium_items.append({
            "id": str(uuid.uuid4()),
            "description": "Premium Support (3 months)",
            "quantity": 1.0,
            "unit_price": 300.0,
            "total": 300.0,
            "type": "service"
        })
        
        return premium_items
    
    def _create_enterprise_variation(self, base_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create enterprise package variation"""
        enterprise_items = []
        
        for item in base_items:
            enterprise_item = item.copy()
            enterprise_item["id"] = str(uuid.uuid4())
            
            # Significantly increase scope for enterprise
            if enterprise_item["quantity"] > 1:
                enterprise_item["quantity"] = enterprise_item["quantity"] * 1.5
            else:
                enterprise_item["unit_price"] = enterprise_item["unit_price"] * 1.4
            
            enterprise_item["total"] = enterprise_item["quantity"] * enterprise_item["unit_price"]
            
            # Add "Enterprise" prefix to description
            if not enterprise_item["description"].startswith("Enterprise"):
                enterprise_item["description"] = f"Enterprise {enterprise_item['description']}"
            
            enterprise_items.append(enterprise_item)
        
        # Add additional enterprise services
        enterprise_items.extend([
            {
                "id": str(uuid.uuid4()),
                "description": "Enterprise Support (12 months)",
                "quantity": 1.0,
                "unit_price": 1200.0,
                "total": 1200.0,
                "type": "service"
            },
            {
                "id": str(uuid.uuid4()),
                "description": "Training & Documentation",
                "quantity": 1.0,
                "unit_price": 800.0,
                "total": 800.0,
                "type": "service"
            },
            {
                "id": str(uuid.uuid4()),
                "description": "Priority Technical Support",
                "quantity": 1.0,
                "unit_price": 500.0,
                "total": 500.0,
                "type": "service"
            }
        ])
        
        return enterprise_items
    
    def _recalculate_quote_totals(self, quote_data: Dict[str, Any]) -> None:
        """Recalculate quote totals after modifications"""
        subtotal = sum(item["total"] for item in quote_data["items"])
        discount = quote_data.get("discount", 0)
        vat_rate = quote_data.get("vat_rate", self.default_vat_rate)
        
        discounted_subtotal = subtotal - discount
        vat_amount = discounted_subtotal * (vat_rate / 100)
        total = discounted_subtotal + vat_amount
        
        quote_data["subtotal"] = round(subtotal, 2)
        quote_data["vat_amount"] = round(vat_amount, 2)
        quote_data["total"] = round(total, 2)