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
from models.invoices import Invoice, InvoiceItem, InvoiceStatus, EInvoiceStatus, ItemType

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
        description="Create a new invoice from natural language description",
        name="create_invoice"
    )
    def create_invoice(self, description: str) -> str:
        """
        Create a new invoice from text description
        
        Args:
            description: Natural language description of the invoice
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Extract invoice items from description
            items = self._extract_items_from_description(description)
            
            # Extract client information
            client_data = self._extract_client_from_description(description)
            
            # Calculate totals
            subtotal = sum(item["total"] for item in items)
            discount = self._extract_discount_from_description(description)
            vat_rate = self._extract_vat_rate_from_description(description) or self.default_vat_rate
            vat_amount = (subtotal - discount) * (vat_rate / 100)
            total = subtotal - discount + vat_amount
            
            # Generate invoice number
            invoice_number = self._generate_invoice_number()
            
            # Extract due date
            due_date = self._extract_due_date_from_description(description)
            
            # Create response matching API format
            response = {
                "action": "create_invoice",
                "endpoint": "/api/invoices/",
                "method": "POST",
                "data": {
                    "clientId": client_data.get("id", str(uuid.uuid4())),
                    "number": invoice_number,
                    "items": [
                        {
                            "id": item["id"],
                            "description": item["description"],
                            "quantity": item["quantity"],
                            "unitPrice": item["unit_price"],
                            "total": item["total"],
                            "type": item["type"]
                        } for item in items
                    ],
                    "discount": round(discount, 2),
                    "vatRate": vat_rate,
                    "dueDate": due_date.isoformat() if due_date else (datetime.now() + timedelta(days=30)).isoformat(),
                    "notes": self._extract_notes_from_description(description)
                },
                "preview": {
                    "invoice": {
                        "id": str(uuid.uuid4()),
                        "clientId": client_data.get("id", str(uuid.uuid4())),
                        "number": invoice_number,
                        "items": [
                            {
                                "id": item["id"],
                                "description": item["description"],
                                "quantity": item["quantity"],
                                "unitPrice": item["unit_price"],
                                "total": item["total"],
                                "type": item["type"]
                            } for item in items
                        ],
                        "subtotal": round(subtotal, 2),
                        "discount": round(discount, 2),
                        "vatRate": vat_rate,
                        "vatAmount": round(vat_amount, 2),
                        "total": round(total, 2),
                        "status": "draft",
                        "dueDate": due_date.isoformat() if due_date else (datetime.now() + timedelta(days=30)).isoformat(),
                        "eInvoiceStatus": None,
                        "notes": self._extract_notes_from_description(description),
                        "createdAt": datetime.now().isoformat(),
                        "updatedAt": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create invoice: {str(e)}"})
    
    @kernel_function(
        description="Update an existing invoice",
        name="update_invoice"
    )
    def update_invoice(self, invoice_id: str, description: str) -> str:
        """
        Update an existing invoice based on description
        
        Args:
            invoice_id: ID of the invoice to update
            description: Natural language description of changes
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Parse what needs to be updated from description
            update_data = {}
            
            # Check for status changes
            status_keywords = {
                "draft": ["draft"],
                "sent": ["send", "sent", "email", "mail"],
                "paid": ["paid", "payment", "received"],
                "overdue": ["overdue", "late"],
                "cancelled": ["cancel", "cancelled", "void"]
            }
            
            for status, keywords in status_keywords.items():
                if any(keyword in description.lower() for keyword in keywords):
                    update_data["status"] = status
                    break
            
            # Check for new items
            items = self._extract_items_from_description(description)
            if items:
                update_data["items"] = [
                    {
                        "id": item["id"],
                        "description": item["description"],
                        "quantity": item["quantity"],
                        "unitPrice": item["unit_price"],
                        "total": item["total"],
                        "type": item["type"]
                    } for item in items
                ]
            
            # Check for discount changes
            discount = self._extract_discount_from_description(description)
            if discount > 0:
                update_data["discount"] = discount
            
            # Check for VAT rate changes
            vat_rate = self._extract_vat_rate_from_description(description)
            if vat_rate:
                update_data["vatRate"] = vat_rate
            
            # Check for due date changes
            due_date = self._extract_due_date_from_description(description)
            if due_date:
                update_data["dueDate"] = due_date.isoformat()
            
            # Check for notes
            notes = self._extract_notes_from_description(description)
            if notes:
                update_data["notes"] = notes
            
            # Check for invoice number changes
            number = self._extract_invoice_number_from_description(description)
            if number:
                update_data["number"] = number
            
            # Calculate preview totals if items changed
            preview_totals = {}
            if "items" in update_data:
                subtotal = sum(item["total"] for item in items)
                discount_amount = update_data.get("discount", 0)
                vat_rate_value = update_data.get("vatRate", self.default_vat_rate)
                vat_amount = (subtotal - discount_amount) * (vat_rate_value / 100)
                total = subtotal - discount_amount + vat_amount
                
                preview_totals = {
                    "subtotal": round(subtotal, 2),
                    "discount": round(discount_amount, 2),
                    "vatRate": vat_rate_value,
                    "vatAmount": round(vat_amount, 2),
                    "total": round(total, 2)
                }
            
            response = {
                "action": "update_invoice",
                "endpoint": f"/api/invoices/{invoice_id}",
                "method": "PUT",
                "data": update_data,
                "preview": {
                    "invoice": {
                        "id": invoice_id,
                        **update_data,
                        **preview_totals,
                        "updatedAt": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to update invoice: {str(e)}"})
    
    @kernel_function(
        description="Delete an invoice by ID",
        name="delete_invoice"
    )
    def delete_invoice(self, invoice_id: str, description: str = "") -> str:
        """
        Delete an invoice by ID
        
        Args:
            invoice_id: ID of the invoice to delete
            description: Optional reason for deletion
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            response = {
                "action": "delete_invoice",
                "endpoint": f"/api/invoices/{invoice_id}",
                "method": "DELETE",
                "data": {},
                "preview": {
                    "message": "Invoice will be permanently deleted",
                    "invoice_id": invoice_id,
                    "reason": description if description else "User requested deletion"
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to prepare invoice deletion: {str(e)}"})
    
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
        description="Get all invoices with optional filtering and search",
        name="get_invoices"
    )
    async def get_invoices(self, search: str = "", status_filter: str = "", client_id: str = "", user_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> str:
        """
        Retrieve a list of invoices with optional filtering
        
        Args:
            search: Optional search text to filter by invoice number or client ID
            status_filter: Filter by status: draft, sent, paid, overdue, cancelled
            client_id: Filter by client ID
            user_id: Filter by user ID (required for security)
            skip: Number of invoices to skip
            limit: Maximum number of invoices to return
            
        Returns:
            JSON string containing the list of invoices
        """
        try:
            from database import get_invoices_collection
            from bson import ObjectId
            
            invoices_collection = get_invoices_collection()
            query_dict = {}

            # Add search filter
            if search:
                import re
                regex = re.compile(re.escape(search), re.IGNORECASE)
                query_dict["$or"] = [
                    {"number": {"$regex": regex}},
                    {"clientId": {"$regex": regex}}
                ]

            # Add status filter
            if status_filter:
                valid_statuses = ["draft", "sent", "paid", "overdue", "cancelled"]
                if status_filter not in valid_statuses:
                    return json.dumps({"error": f"Invalid status filter: {status_filter}"})
                query_dict["status"] = status_filter

            # Add client ID filter
            if client_id:
                query_dict["clientId"] = client_id

            # Add user ID filter
            if user_id:
                query_dict["userId"] = user_id

            # Get total count
            total = await invoices_collection.count_documents(query_dict)

            # Get invoices with pagination
            invoices_cursor = invoices_collection.find(query_dict).skip(skip).limit(limit).sort("createdAt", -1)
            invoices = []
            async for invoice_doc in invoices_cursor:
                # Convert to response format
                invoice_response = {
                    "id": str(invoice_doc["_id"]),
                    "clientId": invoice_doc.get("clientId", ""),
                    "number": invoice_doc.get("number", ""),
                    "items": invoice_doc.get("items", []),
                    "subtotal": invoice_doc.get("subtotal", 0.0),
                    "discount": invoice_doc.get("discount", 0.0),
                    "vatRate": invoice_doc.get("vatRate", 20.0),
                    "vatAmount": invoice_doc.get("vatAmount", 0.0),
                    "total": invoice_doc.get("total", 0.0),
                    "status": invoice_doc.get("status", "draft"),
                    "dueDate": invoice_doc.get("dueDate", "").isoformat() if isinstance(invoice_doc.get("dueDate"), datetime) else invoice_doc.get("dueDate", ""),
                    "eInvoiceStatus": invoice_doc.get("eInvoiceStatus"),
                    "notes": invoice_doc.get("notes"),
                    "createdAt": invoice_doc.get("createdAt", "").isoformat() if isinstance(invoice_doc.get("createdAt"), datetime) else invoice_doc.get("createdAt", ""),
                    "updatedAt": invoice_doc.get("updatedAt", "").isoformat() if isinstance(invoice_doc.get("updatedAt"), datetime) else invoice_doc.get("updatedAt", "")
                }
                invoices.append(invoice_response)

            response = {
                "invoices": invoices,
                "total": total
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get invoices: {str(e)}"})
    @kernel_function(
        description="Get a specific invoice by ID",
        name="get_invoice_by_id"
    )
    async def get_invoice_by_id(self, invoice_id: str, user_id: Optional[str] = None) -> str:
        """
        Retrieve a specific invoice by ID
        
        Args:
            invoice_id: Invoice ID to retrieve
            user_id: Filter by user ID (required for security)
            
        Returns:
            JSON string containing the invoice details
        """
        try:
            from database import get_invoices_collection
            from bson import ObjectId
            
            invoices_collection = get_invoices_collection()

            try:
                query = {"_id": ObjectId(invoice_id)}
                if user_id:
                    query["userId"] = user_id
                invoice_doc = await invoices_collection.find_one(query)
            except:
                return json.dumps({"error": "Invalid invoice ID format"})

            if not invoice_doc:
                return json.dumps({"error": "Invoice not found"})

            # Convert to response format
            invoice_response = {
                "id": str(invoice_doc["_id"]),
                "clientId": invoice_doc.get("clientId", ""),
                "number": invoice_doc.get("number", ""),
                "items": invoice_doc.get("items", []),
                "subtotal": invoice_doc.get("subtotal", 0.0),
                "discount": invoice_doc.get("discount", 0.0),
                "vatRate": invoice_doc.get("vatRate", 20.0),
                "vatAmount": invoice_doc.get("vatAmount", 0.0),
                "total": invoice_doc.get("total", 0.0),
                "status": invoice_doc.get("status", "draft"),
                "dueDate": invoice_doc.get("dueDate", "").isoformat() if isinstance(invoice_doc.get("dueDate"), datetime) else invoice_doc.get("dueDate", ""),
                "eInvoiceStatus": invoice_doc.get("eInvoiceStatus"),
                "notes": invoice_doc.get("notes"),
                "createdAt": invoice_doc.get("createdAt", "").isoformat() if isinstance(invoice_doc.get("createdAt"), datetime) else invoice_doc.get("createdAt", ""),
                "updatedAt": invoice_doc.get("updatedAt", "").isoformat() if isinstance(invoice_doc.get("updatedAt"), datetime) else invoice_doc.get("updatedAt", "")
            }

            response = {
                "invoice": invoice_response
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get invoice: {str(e)}"})
    def convert_quote_to_invoice(self, quote_id: str, description: str = "") -> str:
        """
        Convert an existing quote to an invoice
        
        Args:
            quote_id: ID of the quote to convert
            description: Optional additional details for the conversion
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Generate new invoice number
            invoice_number = self._generate_invoice_number()
            
            # Extract any additional information from description
            due_date = self._extract_due_date_from_description(description) if description else None
            notes = self._extract_notes_from_description(description) if description else ""
            
            response = {
                "action": "convert_quote_to_invoice",
                "endpoint": "/api/invoices/",
                "method": "POST",
                "data": {
                    "quote_id": quote_id,
                    "number": invoice_number,
                    "dueDate": due_date.isoformat() if due_date else (datetime.now() + timedelta(days=30)).isoformat(),
                    "notes": notes,
                    "status": "draft"
                },
                "preview": {
                    "message": f"Quote {quote_id} will be converted to invoice {invoice_number}",
                    "new_invoice_number": invoice_number,
                    "due_date": due_date.isoformat() if due_date else (datetime.now() + timedelta(days=30)).isoformat(),
                    "status": "draft"
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to convert quote to invoice: {str(e)}"})

    @kernel_function(
        description="Calculate invoice totals with VAT, discounts, and final amount",
        name="calculate_invoice_totals"
    )
    def calculate_invoice_totals(self, items_json: str, discount: float = 0.0, vat_rate: float = 20.0) -> str:
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
            
            # Calculate subtotal from item totals
            subtotal = sum(float(item.get("total", 0)) for item in items)
            
            # Apply discount
            subtotal_after_discount = subtotal - discount
            
            # Calculate VAT
            vat_amount = subtotal_after_discount * (vat_rate / 100)
            
            # Calculate total
            total = subtotal_after_discount + vat_amount
            
            result = {
                "subtotal": round(subtotal, 2),
                "vatAmount": round(vat_amount, 2),
                "total": round(total, 2),
                "discount": round(discount, 2),
                "vatRate": vat_rate,
                "currency": self.currency
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to calculate totals: {str(e)}"})
    
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
    
    def _extract_vat_rate_from_description(self, description: str) -> Optional[float]:
        """
        Extract VAT rate from description
        """
        # Pattern for VAT rates
        vat_patterns = [
            r'(?:vat|tax)[:\s]*(\d+(?:\.\d+)?)%?',
            r'(\d+(?:\.\d+)?)%?\s*(?:vat|tax)',
            r'tva[:\s]*(\d+(?:\.\d+)?)%?'  # French VAT
        ]
        
        for pattern in vat_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                rate = float(match.group(1))
                return rate if rate <= 100 else rate / 100  # Handle percentage formats
        
        return None

    def _extract_due_date_from_description(self, description: str) -> Optional[datetime]:
        """
        Extract due date from description
        """
        # Pattern for due dates
        date_patterns = [
            r'(?:due|pay by|payment due)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:due|pay by|payment due)[:\s]*(\d{1,2}\s+\w+\s+\d{2,4})',
            r'in\s+(\d+)\s+days?',
            r'(\d+)\s+days?\s+(?:from now|later)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    if 'days' in pattern:
                        # Handle relative dates
                        days = int(date_str)
                        return datetime.now() + timedelta(days=days)
                    else:
                        # Handle absolute dates (simplified parsing)
                        if '/' in date_str or '-' in date_str:
                            # Try common date formats
                            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                                try:
                                    return datetime.strptime(date_str, fmt)
                                except ValueError:
                                    continue
                except (ValueError, IndexError):
                    continue
        
        return None

    def _extract_invoice_number_from_description(self, description: str) -> Optional[str]:
        """
        Extract invoice number from description
        """
        # Pattern for invoice numbers
        number_patterns = [
            r'(?:invoice|inv|facture)[:\s#]*([A-Z0-9-]+)',
            r'(?:number|num|no)[:\s#]*([A-Z0-9-]+)',
            r'#([A-Z0-9-]+)'
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None

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