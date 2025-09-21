"""
Expense tracking tools for Semantic Kernel
These tools handle expense tracking, categorization, and VAT calculations
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
from models import Expense

class ExpenseTools:
    """
    Semantic Kernel tools for expense tracking and management
    Provides AI-powered expense extraction from receipts and descriptions
    """
    
    def __init__(self, settings: Settings):
        """Initialize expense tools with application settings"""
        self.settings = settings
        self.default_vat_rate = settings.default_vat_rate
        self.currency = settings.default_currency
        
        # Common expense categories
        self.expense_categories = {
            "office": ["office", "supplies", "stationery", "paper", "pen", "pencil", "notebook"],
            "travel": ["travel", "transport", "taxi", "uber", "flight", "train", "hotel", "accommodation"],
            "meals": ["meal", "lunch", "dinner", "breakfast", "restaurant", "food", "coffee"],
            "technology": ["computer", "laptop", "software", "hardware", "phone", "internet", "hosting"],
            "marketing": ["advertising", "marketing", "promotion", "website", "seo", "social media"],
            "utilities": ["electricity", "gas", "water", "phone bill", "internet bill", "utility"],
            "insurance": ["insurance", "liability", "coverage", "premium"],
            "professional": ["consultant", "legal", "accounting", "professional", "advisor"],
            "equipment": ["equipment", "tool", "machinery", "vehicle", "furniture"],
            "training": ["training", "course", "education", "seminar", "workshop", "certification"],
            "miscellaneous": ["misc", "other", "various", "general"]
        }
    
    @kernel_function(
        description="Extract expense information from receipt text or description",
        name="extract_expense_from_text"
    )
    def extract_expense_from_text(self, text: str, receipt_date: Optional[str] = None) -> str:
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