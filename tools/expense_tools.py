"""
Expense management tools for Semantic Kernel
These tools handle expense creation, modification, and data retrieval
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
from models.expenses import Expense, ExpenseCreate, ExpenseUpdate, ExpenseResponse, ExpenseCategory

class ExpenseTools:
    """
    Semantic Kernel tools for expense management
    Provides AI-powered expense creation and management from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize expense tools with application settings"""
        self.settings = settings
        self.default_vat_rate = settings.default_vat_rate
        self.currency = settings.default_currency

    # ===== CREATE/UPDATE/DELETE TOOLS (Return structured responses for frontend verification) =====

    @kernel_function(
        description="Create a new expense from natural language description",
        name="create_expense"
    )
    def create_expense(self, description: str) -> str:
        """
        Create a new expense from text description
        
        Args:
            description: Natural language description of the expense
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Extract expense information from description
            expense_data = self._extract_expense_from_description(description)
            
            # Calculate VAT amount
            vat_amount = expense_data["amount"] * (expense_data["vat_rate"] / 100)
            
            # Create response matching API format
            response = {
                "action": "create_expense",
                "endpoint": "/api/expenses/",
                "method": "POST",
                "data": {
                    "description": expense_data["description"],
                    "amount": expense_data["amount"],
                    "vat_rate": expense_data["vat_rate"],
                    "category": expense_data["category"],
                    "date": expense_data["date"].isoformat(),
                    "notes": expense_data.get("notes"),
                    "receipt_url": expense_data.get("receipt_url")
                },
                "preview": {
                    "expense": {
                        "id": str(uuid.uuid4()),
                        "description": expense_data["description"],
                        "amount": expense_data["amount"],
                        "vat_amount": round(vat_amount, 2),
                        "vat_rate": expense_data["vat_rate"],
                        "category": expense_data["category"],
                        "date": expense_data["date"].isoformat(),
                        "notes": expense_data.get("notes"),
                        "receipt_url": expense_data.get("receipt_url"),
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create expense: {str(e)}"})

    @kernel_function(
        description="Update an existing expense",
        name="update_expense"
    )
    def update_expense(self, expense_id: str, description: str) -> str:
        """
        Update an existing expense based on description
        
        Args:
            expense_id: ID of the expense to update
            description: Natural language description of changes
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Parse what needs to be updated from description
            update_data = {}
            
            # Extract expense information from description
            expense_info = self._extract_expense_from_description(description)
            
            # Only include fields that have actual values
            for field, value in expense_info.items():
                if value is not None and str(value).strip():
                    update_data[field] = value
            
            # Convert datetime to ISO string for JSON serialization
            if "date" in update_data and isinstance(update_data["date"], datetime):
                update_data["date"] = update_data["date"].isoformat()
            
            # Calculate preview VAT if amount or rate changed
            preview_totals = {}
            if "amount" in update_data or "vat_rate" in update_data:
                amount = update_data.get("amount", 0)
                vat_rate = update_data.get("vat_rate", self.default_vat_rate)
                vat_amount = amount * (vat_rate / 100)
                preview_totals = {
                    "vat_amount": round(vat_amount, 2)
                }
            
            response = {
                "action": "update_expense",
                "endpoint": f"/api/expenses/{expense_id}",
                "method": "PUT",
                "data": update_data,
                "preview": {
                    "expense": {
                        "id": expense_id,
                        **update_data,
                        **preview_totals,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to update expense: {str(e)}"})

    @kernel_function(
        description="Delete an expense by ID",
        name="delete_expense"
    )
    def delete_expense(self, expense_id: str, description: str = "") -> str:
        """
        Delete an expense by ID
        
        Args:
            expense_id: ID of the expense to delete
            description: Optional reason for deletion
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            response = {
                "action": "delete_expense",
                "endpoint": f"/api/expenses/{expense_id}",
                "method": "DELETE",
                "data": {},
                "preview": {
                    "message": "Expense will be permanently deleted",
                    "expense_id": expense_id,
                    "reason": description if description else "User requested deletion"
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to prepare expense deletion: {str(e)}"})

    # ===== GET TOOLS (Actually fetch data from database) =====

    @kernel_function(
        description="Get all expenses with optional filtering and search",
        name="get_expenses"
    )
    async def get_expenses(self, search: str = "", category_filter: str = "", start_date: str = "", end_date: str = "", user_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> str:
        """
        Retrieve a list of expenses with optional filtering
        
        Args:
            search: Optional search text to filter by description
            category_filter: Filter by category (Materials, Transport, Equipment, etc.)
            start_date: Filter expenses from this date (YYYY-MM-DD format)
            end_date: Filter expenses until this date (YYYY-MM-DD format)
            user_id: Filter by user ID (required for security)
            skip: Number of expenses to skip
            limit: Maximum number of expenses to return
            
        Returns:
            JSON string containing the list of expenses
        """
        try:
            from database import get_expenses_collection
            from bson import ObjectId
            
            expenses_collection = get_expenses_collection()
            query_dict = {}

            # Add search filter
            if search:
                import re
                regex = re.compile(re.escape(search), re.IGNORECASE)
                query_dict["description"] = {"$regex": regex}

            # Add category filter
            if category_filter:
                valid_categories = [category.value for category in ExpenseCategory]
                if category_filter not in valid_categories:
                    return json.dumps({"error": f"Invalid category filter: {category_filter}"})
                query_dict["category"] = category_filter

            # Add date range filter
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date)
                        date_filter["$gte"] = start_dt
                    except ValueError:
                        return json.dumps({"error": "Invalid start_date format. Use YYYY-MM-DD"})
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date)
                        date_filter["$lte"] = end_dt
                    except ValueError:
                        return json.dumps({"error": "Invalid end_date format. Use YYYY-MM-DD"})
                query_dict["date"] = date_filter

            # Add user ID filter
            if user_id:
                query_dict["userId"] = user_id

            # Get total count
            total = await expenses_collection.count_documents(query_dict)

            # Get expenses with pagination
            expenses_cursor = expenses_collection.find(query_dict).skip(skip).limit(limit).sort("date", -1)
            expenses = []
            async for expense_doc in expenses_cursor:
                # Convert to response format
                expense_response = {
                    "id": str(expense_doc["_id"]),
                    "description": expense_doc.get("description", ""),
                    "amount": expense_doc.get("amount", 0.0),
                    "vat_amount": expense_doc.get("vat_amount", 0.0),
                    "vat_rate": expense_doc.get("vat_rate", 20.0),
                    "category": expense_doc.get("category", "General"),
                    "date": expense_doc.get("date", "").isoformat() if isinstance(expense_doc.get("date"), datetime) else expense_doc.get("date", ""),
                    "notes": expense_doc.get("notes"),
                    "receipt_url": expense_doc.get("receipt_url"),
                    "created_at": expense_doc.get("created_at", "").isoformat() if isinstance(expense_doc.get("created_at"), datetime) else expense_doc.get("created_at", ""),
                    "updated_at": expense_doc.get("updated_at", "").isoformat() if isinstance(expense_doc.get("updated_at"), datetime) else expense_doc.get("updated_at", "")
                }
                expenses.append(expense_response)

            response = {
                "expenses": expenses,
                "total": total
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get expenses: {str(e)}"})

    @kernel_function(
        description="Get a specific expense by ID",
        name="get_expense_by_id"
    )
    async def get_expense_by_id(self, expense_id: str, user_id: Optional[str] = None) -> str:
        """
        Retrieve a specific expense by ID
        
        Args:
            expense_id: Expense ID to retrieve
            user_id: Filter by user ID (required for security)
            
        Returns:
            JSON string containing the expense details
        """
        try:
            from database import get_expenses_collection
            from bson import ObjectId
            
            expenses_collection = get_expenses_collection()

            try:
                query = {"_id": ObjectId(expense_id)}
                if user_id:
                    query["userId"] = user_id
                expense_doc = await expenses_collection.find_one(query)
            except:
                return json.dumps({"error": "Invalid expense ID format"})

            if not expense_doc:
                return json.dumps({"error": "Expense not found"})

            # Convert to response format
            expense_response = {
                "id": str(expense_doc["_id"]),
                "description": expense_doc.get("description", ""),
                "amount": expense_doc.get("amount", 0.0),
                "vat_amount": expense_doc.get("vat_amount", 0.0),
                "vat_rate": expense_doc.get("vat_rate", 20.0),
                "category": expense_doc.get("category", "General"),
                "date": expense_doc.get("date", "").isoformat() if isinstance(expense_doc.get("date"), datetime) else expense_doc.get("date", ""),
                "notes": expense_doc.get("notes"),
                "receipt_url": expense_doc.get("receipt_url"),
                "created_at": expense_doc.get("created_at", "").isoformat() if isinstance(expense_doc.get("created_at"), datetime) else expense_doc.get("created_at", ""),
                "updated_at": expense_doc.get("updated_at", "").isoformat() if isinstance(expense_doc.get("updated_at"), datetime) else expense_doc.get("updated_at", "")
            }

            response = {
                "expense": expense_response
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get expense: {str(e)}"})

    @kernel_function(
        description="Get expenses by category with totals",
        name="get_expenses_by_category"
    )
    async def get_expenses_by_category(self, category: str, start_date: str = "", end_date: str = "", user_id: Optional[str] = None) -> str:
        """
        Get all expenses for a specific category with totals
        
        Args:
            category: Expense category to filter by
            start_date: Optional start date filter (YYYY-MM-DD format)
            end_date: Optional end date filter (YYYY-MM-DD format)
            user_id: Filter by user ID (required for security)
            
        Returns:
            JSON string containing expenses and category totals
        """
        try:
            from database import get_expenses_collection
            
            expenses_collection = get_expenses_collection()
            query_dict = {"category": category}

            # Add date range filter
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date)
                        date_filter["$gte"] = start_dt
                    except ValueError:
                        return json.dumps({"error": "Invalid start_date format. Use YYYY-MM-DD"})
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date)
                        date_filter["$lte"] = end_dt
                    except ValueError:
                        return json.dumps({"error": "Invalid end_date format. Use YYYY-MM-DD"})
                query_dict["date"] = date_filter

            # Add user ID filter
            if user_id:
                query_dict["userId"] = user_id

            # Get expenses
            expenses_cursor = expenses_collection.find(query_dict).sort("date", -1)
            expenses = []
            total_amount = 0
            total_vat = 0
            
            async for expense_doc in expenses_cursor:
                expense_response = {
                    "id": str(expense_doc["_id"]),
                    "description": expense_doc.get("description", ""),
                    "amount": expense_doc.get("amount", 0.0),
                    "vat_amount": expense_doc.get("vat_amount", 0.0),
                    "vat_rate": expense_doc.get("vat_rate", 20.0),
                    "category": expense_doc.get("category", "General"),
                    "date": expense_doc.get("date", "").isoformat() if isinstance(expense_doc.get("date"), datetime) else expense_doc.get("date", ""),
                    "notes": expense_doc.get("notes"),
                    "receipt_url": expense_doc.get("receipt_url")
                }
                expenses.append(expense_response)
                total_amount += expense_doc.get("amount", 0.0)
                total_vat += expense_doc.get("vat_amount", 0.0)

            response = {
                "category": category,
                "expenses": expenses,
                "summary": {
                    "count": len(expenses),
                    "total_amount": round(total_amount, 2),
                    "total_vat": round(total_vat, 2),
                    "total_with_vat": round(total_amount + total_vat, 2)
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get expenses by category: {str(e)}"})

    @kernel_function(
        description="Calculate expense totals with VAT for a date range",
        name="calculate_expense_totals"
    )
    def calculate_expense_totals(self, expenses_json: str) -> str:
        """
        Calculate totals for a list of expenses
        
        Args:
            expenses_json: JSON string containing array of expenses
            
        Returns:
            JSON string containing calculated totals
        """
        try:
            expenses = json.loads(expenses_json)
            if not isinstance(expenses, list):
                raise ValueError("Expenses must be an array")
            
            total_amount = sum(float(expense.get("amount", 0)) for expense in expenses)
            total_vat = sum(float(expense.get("vat_amount", 0)) for expense in expenses)
            total_with_vat = total_amount + total_vat
            
            # Group by category
            category_totals = {}
            for expense in expenses:
                category = expense.get("category", "General")
                if category not in category_totals:
                    category_totals[category] = {
                        "count": 0,
                        "amount": 0,
                        "vat": 0,
                        "total": 0
                    }
                category_totals[category]["count"] += 1
                category_totals[category]["amount"] += float(expense.get("amount", 0))
                category_totals[category]["vat"] += float(expense.get("vat_amount", 0))
                category_totals[category]["total"] += float(expense.get("amount", 0)) + float(expense.get("vat_amount", 0))
            
            result = {
                "summary": {
                    "total_expenses": len(expenses),
                    "total_amount": round(total_amount, 2),
                    "total_vat": round(total_vat, 2),
                    "total_with_vat": round(total_with_vat, 2),
                    "currency": self.currency
                },
                "by_category": {
                    category: {
                        "count": data["count"],
                        "amount": round(data["amount"], 2),
                        "vat": round(data["vat"], 2),
                        "total": round(data["total"], 2)
                    }
                    for category, data in category_totals.items()
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to calculate totals: {str(e)}"})

    # ===== HELPER METHODS =====

    def _extract_expense_from_description(self, description: str) -> Dict[str, Any]:
        """
        Extract expense information from description
        """
        expense_data = {
            "description": "",
            "amount": 0.0,
            "vat_rate": self.default_vat_rate,
            "category": "General",
            "date": datetime.now(),
            "notes": "",
            "receipt_url": None
        }
        
        # Extract amount
        amount_patterns = [
            r'[€$£](\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[€$£]',
            r'(?:cost|price|amount|total)[:\s]*[€$£]?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:euros?|dollars?|pounds?)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                expense_data["amount"] = float(match.group(1))
                break
        
        # Extract VAT rate
        vat_patterns = [
            r'(?:vat|tax)[:\s]*(\d+(?:\.\d+)?)%?',
            r'(\d+(?:\.\d+)?)%?\s*(?:vat|tax)',
            r'tva[:\s]*(\d+(?:\.\d+)?)%?'  # French VAT
        ]
        
        for pattern in vat_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                rate = float(match.group(1))
                expense_data["vat_rate"] = rate if rate <= 100 else rate / 100
                break
        
        # Extract category based on keywords
        description_lower = description.lower()
        category_keywords = {
            "Materials": ["material", "supplies", "parts", "component", "hardware", "lumber", "steel", "concrete"],
            "Transport": ["transport", "travel", "taxi", "uber", "flight", "train", "bus", "fuel", "gas", "petrol"],
            "Equipment": ["equipment", "tool", "machinery", "device", "computer", "laptop", "printer", "scanner"],
            "Labor": ["labor", "labour", "work", "service", "consultation", "wage", "salary", "hourly"],
            "Insurance": ["insurance", "coverage", "premium", "policy", "liability", "protection"],
            "Training": ["training", "course", "education", "seminar", "workshop", "certification", "learning"],
            "Marketing": ["marketing", "advertising", "promotion", "website", "seo", "social media", "campaign"],
            "Others": ["misc", "miscellaneous", "other", "various", "general"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                expense_data["category"] = category
                break
        
        # Extract date
        date_patterns = [
            r'(?:on|date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:today|yesterday)',
            r'(\d{1,2}\s+\w+\s+\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                try:
                    if "today" in match.group(0).lower():
                        expense_data["date"] = datetime.now()
                    elif "yesterday" in match.group(0).lower():
                        expense_data["date"] = datetime.now() - timedelta(days=1)
                    else:
                        date_str = match.group(1)
                        # Try common date formats
                        for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d']:
                            try:
                                expense_data["date"] = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                except (ValueError, AttributeError):
                    continue
                break
        
        # Extract description (clean up the text)
        # Remove amount, date, and category keywords to get clean description
        clean_desc = description
        for pattern in amount_patterns + date_patterns:
            clean_desc = re.sub(pattern, "", clean_desc, flags=re.IGNORECASE)
        
        # Remove common prefixes and clean up
        clean_desc = re.sub(r'^(expense|cost|payment|bill|receipt|purchase)[\s:]*', '', clean_desc, flags=re.IGNORECASE)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        
        if clean_desc:
            expense_data["description"] = clean_desc.title()
        else:
            expense_data["description"] = f"{expense_data['category']} Expense"
        
        # Extract notes
        note_patterns = [
            r'(?:note|notes|comment|comments)[:\s]*([^,\.;]+)',
            r'(?:for|regarding)[:\s]*([^,\.;]+)'
        ]
        
        for pattern in note_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                expense_data["notes"] = match.group(1).strip()
                break
        
        # Extract receipt URL
        url_pattern = r'(?:receipt|url|link)[:\s]*(https?://[^\s]+)'
        url_match = re.search(url_pattern, description, re.IGNORECASE)
        if url_match:
            expense_data["receipt_url"] = url_match.group(1)
        
        return expense_data
        """
        Extract structured expense information from receipt text or description
        
        Args:
            text: Receipt text or expense description
            receipt_date: Optional date string if known
            
        Returns:
            JSON string containing extracted expense data
            
        Example:
            Input: "Office supplies from Staples €45.80 including VAT on 2024-01-15"
            Output: JSON with structured expense data
        """
        try:
            expense_data = {
                "id": str(uuid.uuid4()),
                "description": "",
                "amount": 0.0,
                "vat_amount": 0.0,
                "category": "",
                "date": None,
                "receipt": None,
                "vendor": "",
                "payment_method": "",
                "currency": self.currency,
                "created_at": datetime.now().isoformat()
            }
            
            # Extract basic expense information
            expense_data["description"] = self._extract_description(text)
            expense_data["amount"] = self._extract_amount(text)
            expense_data["vat_amount"] = self._extract_vat_amount(text, expense_data["amount"])
            expense_data["category"] = self._categorize_expense(text)
            expense_data["date"] = self._extract_date(text, receipt_date)
            expense_data["vendor"] = self._extract_vendor(text)
            expense_data["payment_method"] = self._extract_payment_method(text)
            
            # Validate and clean data
            expense_data = self._validate_expense_data(expense_data)
            
            return json.dumps(expense_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to extract expense: {str(e)}"})
    
    @kernel_function(
        description="Automatically categorize an expense based on description and vendor",
        name="categorize_expense"
    )
    def categorize_expense(self, description: str, vendor: str = "") -> str:
        """
        Automatically categorize an expense
        
        Args:
            description: Expense description
            vendor: Vendor name (optional)
            
        Returns:
            JSON string containing categorization results
        """
        try:
            text_to_analyze = f"{description} {vendor}".lower()
            
            category_scores = {}
            
            # Score each category based on keyword matches
            for category, keywords in self.expense_categories.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_to_analyze:
                        score += 1
                
                if score > 0:
                    category_scores[category] = score
            
            # Determine primary category
            if category_scores:
                primary_category = max(category_scores, key=category_scores.get)
                confidence = min(category_scores[primary_category] / 3.0, 1.0)  # Normalize to 0-1
            else:
                primary_category = "miscellaneous"
                confidence = 0.3
            
            # Get suggested categories (top 3)
            suggested_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            result = {
                "primary_category": primary_category,
                "confidence": round(confidence, 2),
                "suggested_categories": [cat for cat, score in suggested_categories],
                "category_scores": category_scores
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to categorize expense: {str(e)}"})
    
    @kernel_function(
        description="Calculate VAT amount from total or net amount",
        name="calculate_vat"
    )
    def calculate_vat(self, amount: float, vat_rate: Optional[float] = None, amount_includes_vat: bool = True) -> str:
        """
        Calculate VAT amount from total or net amount
        
        Args:
            amount: The amount to calculate VAT for
            vat_rate: VAT rate to use (defaults to company default)
            amount_includes_vat: Whether the amount includes VAT or not
            
        Returns:
            JSON string containing VAT calculation results
        """
        try:
            vat_rate = vat_rate or self.default_vat_rate
            
            if amount_includes_vat:
                # Amount includes VAT - extract VAT from total
                net_amount = amount / (1 + vat_rate / 100)
                vat_amount = amount - net_amount
            else:
                # Amount is net - calculate VAT to add
                net_amount = amount
                vat_amount = amount * (vat_rate / 100)
                amount = net_amount + vat_amount
            
            result = {
                "total_amount": round(amount, 2),
                "net_amount": round(net_amount, 2),
                "vat_amount": round(vat_amount, 2),
                "vat_rate": vat_rate,
                "currency": self.currency,
                "amount_includes_vat": amount_includes_vat
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to calculate VAT: {str(e)}"})
    
    @kernel_function(
        description="Parse receipt information from structured receipt text",
        name="parse_receipt"
    )
    def parse_receipt(self, receipt_text: str) -> str:
        """
        Parse structured receipt text to extract detailed information
        
        Args:
            receipt_text: Raw receipt text (OCR output or typed receipt)
            
        Returns:
            JSON string containing parsed receipt data
        """
        try:
            receipt_data = {
                "vendor": "",
                "vendor_address": "",
                "vendor_phone": "",
                "vendor_vat_number": "",
                "receipt_number": "",
                "date": None,
                "time": None,
                "items": [],
                "subtotal": 0.0,
                "vat_amount": 0.0,
                "total": 0.0,
                "payment_method": "",
                "payment_reference": ""
            }
            
            # Extract vendor information (usually at the top)
            lines = receipt_text.split('\\n')
            receipt_data["vendor"] = self._extract_vendor_from_receipt(lines)
            receipt_data["vendor_address"] = self._extract_vendor_address(lines)
            receipt_data["vendor_phone"] = self._extract_vendor_phone(lines)
            receipt_data["vendor_vat_number"] = self._extract_vendor_vat(lines)
            
            # Extract receipt metadata
            receipt_data["receipt_number"] = self._extract_receipt_number(receipt_text)
            receipt_data["date"] = self._extract_receipt_date(receipt_text)
            receipt_data["time"] = self._extract_receipt_time(receipt_text)
            
            # Extract line items
            receipt_data["items"] = self._extract_receipt_items(lines)
            
            # Extract totals
            receipt_data["subtotal"] = self._extract_subtotal(receipt_text)
            receipt_data["vat_amount"] = self._extract_vat_from_receipt(receipt_text)
            receipt_data["total"] = self._extract_total_from_receipt(receipt_text)
            
            # Extract payment information
            receipt_data["payment_method"] = self._extract_payment_method(receipt_text)
            receipt_data["payment_reference"] = self._extract_payment_reference(receipt_text)
            
            return json.dumps(receipt_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to parse receipt: {str(e)}"})
    
    @kernel_function(
        description="Split shared expenses among team members or projects",
        name="split_expense"
    )
    def split_expense(self, expense_json: str, split_method: str = "equal", split_data: str = "{}") -> str:
        """
        Split an expense among multiple people or projects
        
        Args:
            expense_json: JSON string containing expense data
            split_method: Method to use ("equal", "percentage", "amount")
            split_data: JSON string containing split information
            
        Returns:
            JSON string containing split expense data
        """
        try:
            expense = json.loads(expense_json)
            split_info = json.loads(split_data) if split_data != "{}" else {}
            
            total_amount = expense.get("amount", 0)
            vat_amount = expense.get("vat_amount", 0)
            
            split_expenses = []
            
            if split_method == "equal":
                # Equal split among participants
                participants = split_info.get("participants", ["Person 1", "Person 2"])
                amount_per_person = total_amount / len(participants)
                vat_per_person = vat_amount / len(participants)
                
                for participant in participants:
                    split_expense = expense.copy()
                    split_expense["id"] = str(uuid.uuid4())
                    split_expense["description"] = f"{expense['description']} (Split - {participant})"
                    split_expense["amount"] = round(amount_per_person, 2)
                    split_expense["vat_amount"] = round(vat_per_person, 2)
                    split_expense["split_info"] = {
                        "original_expense_id": expense["id"],
                        "participant": participant,
                        "split_method": "equal",
                        "total_participants": len(participants)
                    }
                    split_expenses.append(split_expense)
            
            elif split_method == "percentage":
                # Split by percentage
                percentages = split_info.get("percentages", {})
                total_percentage = sum(percentages.values())
                
                if abs(total_percentage - 100) > 0.01:
                    raise ValueError("Percentages must sum to 100%")
                
                for participant, percentage in percentages.items():
                    amount = total_amount * (percentage / 100)
                    vat = vat_amount * (percentage / 100)
                    
                    split_expense = expense.copy()
                    split_expense["id"] = str(uuid.uuid4())
                    split_expense["description"] = f"{expense['description']} (Split - {participant} {percentage}%)"
                    split_expense["amount"] = round(amount, 2)
                    split_expense["vat_amount"] = round(vat, 2)
                    split_expense["split_info"] = {
                        "original_expense_id": expense["id"],
                        "participant": participant,
                        "split_method": "percentage",
                        "percentage": percentage
                    }
                    split_expenses.append(split_expense)
            
            elif split_method == "amount":
                # Split by specific amounts
                amounts = split_info.get("amounts", {})
                total_split_amount = sum(amounts.values())
                
                if abs(total_split_amount - total_amount) > 0.01:
                    raise ValueError("Split amounts must equal total expense amount")
                
                for participant, amount in amounts.items():
                    vat = vat_amount * (amount / total_amount)
                    
                    split_expense = expense.copy()
                    split_expense["id"] = str(uuid.uuid4())
                    split_expense["description"] = f"{expense['description']} (Split - {participant})"
                    split_expense["amount"] = round(amount, 2)
                    split_expense["vat_amount"] = round(vat, 2)
                    split_expense["split_info"] = {
                        "original_expense_id": expense["id"],
                        "participant": participant,
                        "split_method": "amount",
                        "amount": amount
                    }
                    split_expenses.append(split_expense)
            
            return json.dumps(split_expenses, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to split expense: {str(e)}"})
    
    def _extract_description(self, text: str) -> str:
        """Extract expense description from text"""
        # Look for descriptive patterns
        description_patterns = [
            r'(?:purchase|bought|paid for)\\s+([^,\\.;]+)',
            r'([^,\\.;]+?)\\s+(?:from|at)\\s+[A-Z]',
            r'^([^€$£\\d]{10,50})',  # First 10-50 characters without currency
        ]
        
        for pattern in description_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                if len(description) > 5:
                    return description.title()
        
        # Extract by removing amount and vendor
        clean_text = re.sub(r'[€$£]?\\d+(?:\\.\\d{2})?', '', text)
        clean_text = re.sub(r'\\b(?:from|at|on)\\s+\\w+', '', clean_text)
        clean_text = clean_text.strip(' -.,;:')
        
        if len(clean_text) > 5:
            return clean_text.title()
        
        return "Expense"
    
    def _extract_amount(self, text: str) -> float:
        """Extract monetary amount from text"""
        # Patterns for amounts with currency symbols
        amount_patterns = [
            r'[€$£](\\d+(?:\\.\\d{2})?)',  # €45.80
            r'(\\d+(?:\\.\\d{2})?)\\s*[€$£]',  # 45.80€
            r'(?:total|amount|price)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)',  # total: €45.80
            r'(\\d+\\.\\d{2})\\b',  # Any decimal amount
            r'\\b(\\d+)\\s*euros?\\b',  # 45 euros
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return 0.0
    
    def _extract_vat_amount(self, text: str, total_amount: float) -> float:
        """Extract VAT amount from text"""
        # Look for explicit VAT amounts
        vat_patterns = [
            r'(?:vat|tva|tax)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)',
            r'[€$£]?(\\d+(?:\\.\\d{2})?)\\s*(?:vat|tva|tax)',
            r'(?:including|inc\\.?)\\s+vat\\s+[€$£]?(\\d+(?:\\.\\d{2})?)'
        ]
        
        for pattern in vat_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Check if VAT is included in total
        if any(phrase in text.lower() for phrase in ["including vat", "inc vat", "vat included", "ttc"]):
            # Calculate VAT from total (assuming default rate)
            net_amount = total_amount / (1 + self.default_vat_rate / 100)
            return total_amount - net_amount
        
        return 0.0
    
    def _categorize_expense(self, text: str) -> str:
        """Categorize expense based on description"""
        text_lower = text.lower()
        
        category_scores = {}
        
        for category, keywords in self.expense_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return "miscellaneous"
    
    def _extract_date(self, text: str, provided_date: Optional[str] = None) -> str:
        """Extract date from text"""
        if provided_date:
            try:
                # Validate provided date
                datetime.fromisoformat(provided_date.replace('Z', '+00:00'))
                return provided_date
            except:
                pass
        
        # Date patterns
        date_patterns = [
            r'\\b(\\d{4}-\\d{2}-\\d{2})\\b',  # YYYY-MM-DD
            r'\\b(\\d{2}/\\d{2}/\\d{4})\\b',  # DD/MM/YYYY
            r'\\b(\\d{2}-\\d{2}-\\d{4})\\b',  # DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                try:
                    # Try to parse and convert to ISO format
                    if '-' in date_str and len(date_str.split('-')[0]) == 4:
                        # Already in YYYY-MM-DD format
                        return datetime.fromisoformat(date_str).isoformat()
                    else:
                        # Convert DD/MM/YYYY or DD-MM-YYYY to YYYY-MM-DD
                        parts = re.split(r'[/-]', date_str)
                        if len(parts) == 3:
                            day, month, year = parts
                            iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            return datetime.fromisoformat(iso_date).isoformat()
                except:
                    continue
        
        # Default to today
        return datetime.now().isoformat()
    
    def _extract_vendor(self, text: str) -> str:
        """Extract vendor name from text"""
        vendor_patterns = [
            r'(?:from|at|vendor|store)\\s+([A-Z][a-zA-Z\\s&]+)(?:\\s|$)',
            r'^([A-Z][a-zA-Z\\s&]{3,20})\\s',  # Vendor name at start
            r'\\b([A-Z]{2,}[A-Z\\s&]{2,15})\\b'  # All caps company names
        ]
        
        for pattern in vendor_patterns:
            match = re.search(pattern, text)
            if match:
                vendor = match.group(1).strip()
                if len(vendor) > 2 and len(vendor) < 30:
                    return vendor.title()
        
        return ""
    
    def _extract_payment_method(self, text: str) -> str:
        """Extract payment method from text"""
        payment_methods = {
            "credit card": ["card", "credit", "visa", "mastercard", "amex"],
            "debit card": ["debit", "pin"],
            "cash": ["cash", "espèces"],
            "bank transfer": ["transfer", "virement", "wire"],
            "paypal": ["paypal"],
            "check": ["check", "cheque", "chèque"]
        }
        
        text_lower = text.lower()
        
        for method, keywords in payment_methods.items():
            if any(keyword in text_lower for keyword in keywords):
                return method
        
        return ""
    
    def _validate_expense_data(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean expense data"""
        # Ensure minimum required fields
        if not expense_data.get("description"):
            expense_data["description"] = "Expense"
        
        if expense_data.get("amount", 0) <= 0:
            expense_data["amount"] = 0.0
        
        if not expense_data.get("category"):
            expense_data["category"] = "miscellaneous"
        
        if not expense_data.get("date"):
            expense_data["date"] = datetime.now().isoformat()
        
        # Round monetary values
        expense_data["amount"] = round(expense_data.get("amount", 0), 2)
        expense_data["vat_amount"] = round(expense_data.get("vat_amount", 0), 2)
        
        return expense_data
    
    # Receipt parsing helper methods
    def _extract_vendor_from_receipt(self, lines: List[str]) -> str:
        """Extract vendor from receipt lines (usually first few lines)"""
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and not re.match(r'^[\\d\\s\\-\\.\\(\\)]+$', line):
                # Skip pure numeric lines, addresses, phones
                if not re.search(r'\\d{4,}|\\s+\\d+\\s+|^\\d+$', line):
                    return line.title()
        return ""
    
    def _extract_vendor_address(self, lines: List[str]) -> str:
        """Extract vendor address from receipt lines"""
        address_patterns = [
            r'\\d+\\s+[A-Za-z\\s]+(?:street|st|avenue|ave|road|rd)',
            r'\\d{5}\\s+[A-Za-z\\s]+',  # Postal code + city
        ]
        
        for line in lines[:10]:
            for pattern in address_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return line.strip()
        return ""
    
    def _extract_vendor_phone(self, lines: List[str]) -> str:
        """Extract vendor phone from receipt lines"""
        phone_pattern = r'[\\d\\s\\-\\.\\(\\)]{10,}'
        
        for line in lines[:10]:
            if re.search(phone_pattern, line) and len(re.sub(r'[^\\d]', '', line)) >= 8:
                return line.strip()
        return ""
    
    def _extract_vendor_vat(self, lines: List[str]) -> str:
        """Extract vendor VAT number from receipt lines"""
        vat_patterns = [
            r'(?:vat|tva)\\s*:?\\s*([A-Z0-9]+)',
            r'\\b([A-Z]{2}\\d{8,12})\\b'  # European VAT format
        ]
        
        text = ' '.join(lines)
        for pattern in vat_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""
    
    def _extract_receipt_number(self, text: str) -> str:
        """Extract receipt/invoice number"""
        patterns = [
            r'(?:receipt|invoice|ticket)\\s*#?\\s*:?\\s*(\\w+)',
            r'(?:ref|reference)\\s*:?\\s*(\\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""
    
    def _extract_receipt_date(self, text: str) -> str:
        """Extract date from receipt"""
        return self._extract_date(text)
    
    def _extract_receipt_time(self, text: str) -> str:
        """Extract time from receipt"""
        time_pattern = r'\\b(\\d{1,2}:\\d{2}(?::\\d{2})?)\\b'
        match = re.search(time_pattern, text)
        return match.group(1) if match else ""
    
    def _extract_receipt_items(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract line items from receipt"""
        items = []
        item_pattern = r'^\\s*(.+?)\\s+(\\d+(?:\\.\\d{2})?)\\s*$'
        
        for line in lines:
            match = re.search(item_pattern, line)
            if match:
                description = match.group(1).strip()
                amount = float(match.group(2))
                
                if len(description) > 2 and amount > 0:
                    items.append({
                        "description": description,
                        "amount": amount
                    })
        
        return items
    
    def _extract_subtotal(self, text: str) -> float:
        """Extract subtotal from receipt"""
        patterns = [
            r'(?:subtotal|sub-total)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)',
            r'(?:net|ht)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0
    
    def _extract_vat_from_receipt(self, text: str) -> float:
        """Extract VAT amount from receipt"""
        return self._extract_vat_amount(text, 0)
    
    def _extract_total_from_receipt(self, text: str) -> float:
        """Extract total amount from receipt"""
        patterns = [
            r'(?:total|ttc)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)',
            r'(?:amount due|à payer)\\s*:?\\s*[€$£]?(\\d+(?:\\.\\d{2})?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # Fallback to general amount extraction
        return self._extract_amount(text)
    
    def _extract_payment_reference(self, text: str) -> str:
        """Extract payment reference from receipt"""
        patterns = [
            r'(?:ref|reference|transaction)\\s*:?\\s*(\\w+)',
            r'(?:card|transaction)\\s*#\\s*(\\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""