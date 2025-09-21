"""
Invoice generation tools for Semantic Kernel
These tools handle invoice creation, item extraction, and calculations
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
from models import Invoice, QuoteItem, Client, InvoiceStatus

class InvoiceTools:
    """
    Semantic Kernel tools for invoice generation and management
    Provides AI-powered invoice creation from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize invoice tools with application settings"""
        self.settings = settings
        self.default_vat_rate = settings.default_vat_rate
        self.company_name = settings.company_name
        self.currency = settings.default_currency
    
    @kernel_function(
        description="Generate a complete invoice from natural language text description",
        name="generate_invoice_from_text"
    )
    def generate_invoice_from_text(self, description: str, client_id: Optional[str] = None) -> str:
        """
        Generate a complete invoice from text description
        
        Args:
            description: Natural language description of the invoice
            client_id: Optional client ID if known
            
        Returns:
            JSON string containing the generated invoice data
            
        Example:
            Input: "Create invoice for John Doe at 123 Main St for website development 40 hours at €50/hour"
            Output: JSON with complete invoice structure
        """
        try:
            # Extract invoice items from description
            items = self._extract_items_from_description(description)
            
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
            
            # Generate invoice number
            invoice_number = self._generate_invoice_number()
            
            # Create invoice structure
            invoice_data = {
                "id": str(uuid.uuid4()),
                "client_id": client_data.get("id", str(uuid.uuid4())),
                "client": client_data,
                "number": invoice_number,
                "items": items,
                "subtotal": round(subtotal, 2),
                "discount": round(discount, 2),
                "vat_rate": self.default_vat_rate,
                "vat_amount": round(vat_amount, 2),
                "total": round(total, 2),
                "status": "draft",
                "created_at": datetime.now().isoformat(),
                "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "notes": self._extract_notes_from_description(description)
            }
            
            return json.dumps(invoice_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to generate invoice: {str(e)}"})
    
    @kernel_function(
        description="Extract billable items (services, products, labor) from text description",
        name="extract_items_from_description"
    )
    def extract_items_from_description(self, description: str) -> str:
        """
        Extract billable items from natural language description
        
        Args:
            description: Text containing item descriptions, quantities, and prices
            
        Returns:
            JSON string containing list of extracted items
            
        Example:
            Input: "Website development 40 hours at €50/hour, hosting setup €200, domain registration €15"
            Output: JSON array with item objects
        """
        try:
            items = self._extract_items_from_description(description)
            return json.dumps(items, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to extract items: {str(e)}"})
    
    @kernel_function(
        description="Calculate invoice totals with VAT, discounts, and final amount",
        name="calculate_totals"
    )
    def calculate_totals(self, items_json: str, discount: float = 0.0, vat_rate: Optional[float] = None) -> str:
        """
        Calculate totals for invoice items with VAT and discounts
        
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
            return json.dumps({"error": f"Failed to calculate totals: {str(e)}"})
    
    @kernel_function(
        description="Generate a unique invoice number following company format",
        name="generate_invoice_number"
    )
    def generate_invoice_number(self, prefix: str = "INV") -> str:
        """
        Generate a unique invoice number
        
        Args:
            prefix: Prefix for the invoice number (default: INV)
            
        Returns:
            Unique invoice number string
        """
        try:
            # Generate format: PREFIX-YYYY-NNNN
            current_year = datetime.now().year
            
            # In a real implementation, you would get the next sequential number from database
            # For now, use timestamp-based approach
            timestamp_suffix = int(datetime.now().timestamp()) % 10000
            
            invoice_number = f"{prefix}-{current_year}-{timestamp_suffix:04d}"
            
            return invoice_number
            
        except Exception as e:
            return f"INV-{datetime.now().year}-0001"
    
    @kernel_function(
        description="Convert a quote to an invoice with updated status and due date",
        name="convert_quote_to_invoice"
    )
    def convert_quote_to_invoice(self, quote_json: str) -> str:
        """
        Convert an existing quote to an invoice
        
        Args:
            quote_json: JSON string containing quote data
            
        Returns:
            JSON string containing converted invoice data
        """
        try:
            quote_data = json.loads(quote_json)
            
            # Convert quote to invoice
            invoice_data = {
                **quote_data,
                "id": str(uuid.uuid4()),  # New ID for invoice
                "number": self._generate_invoice_number(),  # New invoice number
                "status": "draft",  # Invoice status instead of quote status
                "created_at": datetime.now().isoformat(),
                "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "quote_id": quote_data.get("id"),  # Reference to original quote
            }
            
            # Remove quote-specific fields
            if "valid_until" in invoice_data:
                del invoice_data["valid_until"]
            
            return json.dumps(invoice_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to convert quote to invoice: {str(e)}"})
    
    def _extract_items_from_description(self, description: str) -> List[Dict[str, Any]]:
        """
        Internal method to extract items from description using regex patterns
        """
        items = []
        
        # Common patterns for items with quantities and prices
        patterns = [
            # Pattern: "40 hours at €50/hour" or "40h at €50/h"
            r'(\\d+(?:\\.\\d+)?)\\s*(?:hours?|hrs?|h)\\s*(?:at|@)\\s*[€$£]?(\\d+(?:\\.\\d+)?)(?:/hour|/hr|/h)?',
            # Pattern: "website development €2000" or "hosting €200"
            r'([^,\\.;]+?)\\s*[€$£](\\d+(?:\\.\\d+)?)',
            # Pattern: "3 x consulting sessions at €150 each"
            r'(\\d+)\\s*x?\\s*([^@]+?)\\s*(?:at|@)\\s*[€$£]?(\\d+(?:\\.\\d+)?)(?:\\s*each)?',
            # Pattern: "domain registration for €15"
            r'([^,\\.;]+?)\\s*for\\s*[€$£](\\d+(?:\\.\\d+)?)'
        ]
        
        item_id = 1
        
        for pattern in patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) == 2:
                        # Simple item with description and price
                        if match.group(1).replace('.', '').replace(',', '').isdigit():
                            # First group is quantity, need to find description
                            quantity = float(match.group(1))
                            unit_price = float(match.group(2))
                            description_part = "Service"  # Default description
                        else:
                            # First group is description
                            description_part = match.group(1).strip()
                            quantity = 1.0
                            unit_price = float(match.group(2))
                    
                    elif len(match.groups()) == 3:
                        # Item with quantity, description, and price
                        if match.group(1).replace('.', '').replace(',', '').isdigit():
                            quantity = float(match.group(1))
                            description_part = match.group(2).strip()
                            unit_price = float(match.group(3))
                        else:
                            # Hour-based pattern
                            quantity = float(match.group(1))
                            unit_price = float(match.group(2))
                            description_part = "Hourly service"
                    
                    else:
                        continue
                    
                    # Clean up description
                    description_part = description_part.strip(' -.,;:')
                    if not description_part:
                        description_part = "Service"
                    
                    # Calculate total
                    total = quantity * unit_price
                    
                    # Determine item type based on description
                    item_type = "service"  # Default
                    if any(word in description_part.lower() for word in ["material", "product", "equipment", "hardware"]):
                        item_type = "material"
                    elif any(word in description_part.lower() for word in ["hour", "labor", "work", "development"]):
                        item_type = "labor"
                    
                    item = {
                        "id": str(item_id),
                        "description": description_part.title(),
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total": round(total, 2),
                        "type": item_type
                    }
                    
                    items.append(item)
                    item_id += 1
                    
                except (ValueError, IndexError):
                    continue
        
        # If no items found, try to extract a simple total amount
        if not items:
            total_pattern = r'total[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)'
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
        
        return items
    
    def _extract_client_from_description(self, description: str) -> Dict[str, Any]:
        """
        Extract client information from description
        """
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
        
        # Extract name patterns
        name_patterns = [
            r'(?:for|to|client)\\s+([A-Z][a-z]+\\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\\s+[A-Z][a-z]+)(?:\\s+at|\\s+from)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description)
            if match:
                client_data["name"] = match.group(1).strip()
                break
        
        # Extract email
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, description)
        if email_match:
            client_data["email"] = email_match.group(1)
        
        # Extract phone
        phone_pattern = r'(?:phone|tel|mobile)[:\\s]*([+\\d\\s\\-\\(\\)]+)'
        phone_match = re.search(phone_pattern, description, re.IGNORECASE)
        if phone_match:
            client_data["phone"] = phone_match.group(1).strip()
        
        # Extract address
        address_patterns = [
            r'(?:at|address)[:\\s]*([^,\\.;]+(?:street|st|avenue|ave|road|rd|drive|dr)[^,\\.;]*)',
            r'(\\d+\\s+[^,\\.;]+(?:street|st|avenue|ave|road|rd|drive|dr)[^,\\.;]*)'
        ]
        
        for pattern in address_patterns:
            address_match = re.search(pattern, description, re.IGNORECASE)
            if address_match:
                client_data["address"] = address_match.group(1).strip()
                break
        
        # Extract company
        company_patterns = [
            r'(?:company|corp|inc|ltd|llc)[:\\s]*([^,\\.;]+)',
            r'([^,\\.;]+(?:company|corp|inc|ltd|llc))'
        ]
        
        for pattern in company_patterns:
            company_match = re.search(pattern, description, re.IGNORECASE)
            if company_match:
                client_data["company"] = company_match.group(1).strip()
                break
        
        return client_data
    
    def _extract_discount_from_description(self, description: str) -> float:
        """
        Extract discount amount from description
        """
        # Pattern for discount amounts
        discount_patterns = [
            r'discount[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)',
            r'(?:less|minus)[:\\s]*[€$£]?(\\d+(?:\\.\\d+)?)',
            r'[€$£]?(\\d+(?:\\.\\d+)?)\\s*(?:discount|off)'
        ]
        
        for pattern in discount_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_notes_from_description(self, description: str) -> str:
        """
        Extract notes or additional information from description
        """
        # Look for note indicators
        note_patterns = [
            r'(?:note|notes|comment|comments)[:\\s]*([^,\\.;]+)',
            r'(?:additional|extra|special)[:\\s]*([^,\\.;]+)'
        ]
        
        for pattern in note_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _generate_invoice_number(self) -> str:
        """
        Generate a unique invoice number
        """
        current_year = datetime.now().year
        timestamp_suffix = int(datetime.now().timestamp()) % 10000
        return f"INV-{current_year}-{timestamp_suffix:04d}"