"""
Client management tools for Semantic Kernel
These tools handle client creation, modification, and data retrieval
"""

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import List, Dict, Any, Optional
import json
import re
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from config.settings import Settings
from models.clients import Client, ClientCreate, ClientUpdate, ClientResponse, ClientStatus

class ClientTools:
    """
    Semantic Kernel tools for client management
    Provides AI-powered client creation and management from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize client tools with application settings"""
        self.settings = settings
        self.company_name = settings.company_name
        self.currency = settings.default_currency

    # ===== CREATE/UPDATE/DELETE TOOLS (Return structured responses for frontend verification) =====

    @kernel_function(
        description="Create a new client from natural language description",
        name="create_client"
    )
    def create_client(self, description: str) -> str:
        """
        Create a new client from text description
        
        Args:
            description: Natural language description of the client
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Extract client information from description
            client_data = self._extract_client_from_description(description)
            
            # Create response matching API format
            response = {
                "action": "create_client",
                "endpoint": "/api/clients/",
                "method": "POST",
                "data": {
                    "name": client_data.get("name", ""),
                    "email": client_data.get("email", ""),
                    "phone": client_data.get("phone", ""),
                    "address": client_data.get("address", ""),
                    "company": client_data.get("company"),
                    "notes": client_data.get("notes")
                },
                "preview": {
                    "client": {
                        "id": str(uuid.uuid4()),
                        "name": client_data.get("name", ""),
                        "email": client_data.get("email", ""),
                        "phone": client_data.get("phone", ""),
                        "address": client_data.get("address", ""),
                        "company": client_data.get("company"),
                        "balance": 0.0,
                        "status": "active",
                        "notes": client_data.get("notes"),
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create client: {str(e)}"})

    @kernel_function(
        description="Update an existing client",
        name="update_client"
    )
    def update_client(self, client_id: str, description: str) -> str:
        """
        Update an existing client based on description
        
        Args:
            client_id: ID of the client to update
            description: Natural language description of changes
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            # Parse what needs to be updated from description
            update_data = {}
            
            # Extract client information from description
            client_info = self._extract_client_from_description(description)
            
            # Only include fields that have actual values
            for field, value in client_info.items():
                if value and value.strip():
                    update_data[field] = value
            
            # Check for status changes
            status_keywords = {
                "active": ["active", "activate", "enable"],
                "delinquent": ["delinquent", "overdue", "late"],
                "archived": ["archive", "archived", "disable", "inactive"]
            }
            
            for status, keywords in status_keywords.items():
                if any(keyword in description.lower() for keyword in keywords):
                    update_data["status"] = status
                    break
            
            # Check for balance changes
            balance = self._extract_balance_from_description(description)
            if balance is not None:
                update_data["balance"] = balance
            
            response = {
                "action": "update_client",
                "endpoint": f"/api/clients/{client_id}",
                "method": "PUT",
                "data": update_data,
                "preview": {
                    "client": {
                        "id": client_id,
                        **update_data,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to update client: {str(e)}"})

    @kernel_function(
        description="Delete a client by ID",
        name="delete_client"
    )
    def delete_client(self, client_id: str, description: str = "") -> str:
        """
        Delete a client by ID
        
        Args:
            client_id: ID of the client to delete
            description: Optional reason for deletion
            
        Returns:
            JSON string for frontend verification before API call
        """
        try:
            response = {
                "action": "delete_client",
                "endpoint": f"/api/clients/{client_id}",
                "method": "DELETE",
                "data": {},
                "preview": {
                    "message": "Client will be permanently deleted",
                    "client_id": client_id,
                    "reason": description if description else "User requested deletion"
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to prepare client deletion: {str(e)}"})

    # ===== GET TOOLS (Actually fetch data from database) =====

    @kernel_function(
        description="Get all clients with optional filtering and search",
        name="get_clients"
    )
    async def get_clients(self, search: str = "", status_filter: str = "", user_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> str:
        """
        Retrieve a list of clients with optional filtering
        
        Args:
            search: Optional search text to filter by name, email, or company
            status_filter: Filter by status: active, delinquent, archived
            user_id: Filter by user ID (required for security)
            skip: Number of clients to skip
            limit: Maximum number of clients to return
            
        Returns:
            JSON string containing the list of clients
        """
        try:
            from database import get_clients_collection
            from bson import ObjectId
            
            clients_collection = get_clients_collection()
            query_dict = {}

            # Add search filter
            if search:
                import re
                regex = re.compile(re.escape(search), re.IGNORECASE)
                query_dict["$or"] = [
                    {"name": {"$regex": regex}},
                    {"email": {"$regex": regex}},
                    {"company": {"$regex": regex}}
                ]

            # Add status filter
            if status_filter:
                valid_statuses = ["active", "delinquent", "archived"]
                if status_filter not in valid_statuses:
                    return json.dumps({"error": f"Invalid status filter: {status_filter}"})
                query_dict["status"] = status_filter

            # Add user ID filter
            if user_id:
                query_dict["userId"] = user_id

            # Get total count
            total = await clients_collection.count_documents(query_dict)

            # Get clients with pagination
            clients_cursor = clients_collection.find(query_dict).skip(skip).limit(limit).sort("created_at", -1)
            clients = []
            async for client_doc in clients_cursor:
                # Convert to response format
                client_response = {
                    "id": str(client_doc["_id"]),
                    "name": client_doc.get("name", ""),
                    "email": client_doc.get("email", ""),
                    "phone": client_doc.get("phone", ""),
                    "address": client_doc.get("address", ""),
                    "company": client_doc.get("company"),
                    "balance": client_doc.get("balance", 0.0),
                    "status": client_doc.get("status", "active"),
                    "notes": client_doc.get("notes"),
                    "created_at": client_doc.get("created_at", "").isoformat() if isinstance(client_doc.get("created_at"), datetime) else client_doc.get("created_at", ""),
                    "updated_at": client_doc.get("updated_at", "").isoformat() if isinstance(client_doc.get("updated_at"), datetime) else client_doc.get("updated_at", "")
                }
                clients.append(client_response)

            response = {
                "clients": clients,
                "total": total
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get clients: {str(e)}"})

    @kernel_function(
        description="Get a specific client by ID",
        name="get_client_by_id"
    )
    async def get_client_by_id(self, client_id: str, user_id: Optional[str] = None) -> str:
        """
        Retrieve a specific client by ID
        
        Args:
            client_id: Client ID to retrieve
            user_id: Filter by user ID (required for security)
            
        Returns:
            JSON string containing the client details
        """
        try:
            from database import get_clients_collection
            from bson import ObjectId
            
            clients_collection = get_clients_collection()

            try:
                query = {"_id": ObjectId(client_id)}
                if user_id:
                    query["userId"] = user_id
                client_doc = await clients_collection.find_one(query)
            except:
                return json.dumps({"error": "Invalid client ID format"})

            if not client_doc:
                return json.dumps({"error": "Client not found"})

            # Convert to response format
            client_response = {
                "id": str(client_doc["_id"]),
                "name": client_doc.get("name", ""),
                "email": client_doc.get("email", ""),
                "phone": client_doc.get("phone", ""),
                "address": client_doc.get("address", ""),
                "company": client_doc.get("company"),
                "balance": client_doc.get("balance", 0.0),
                "status": client_doc.get("status", "active"),
                "notes": client_doc.get("notes"),
                "created_at": client_doc.get("created_at", "").isoformat() if isinstance(client_doc.get("created_at"), datetime) else client_doc.get("created_at", ""),
                "updated_at": client_doc.get("updated_at", "").isoformat() if isinstance(client_doc.get("updated_at"), datetime) else client_doc.get("updated_at", "")
            }

            response = {
                "client": client_response
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get client: {str(e)}"})

    @kernel_function(
        description="Search clients by various criteria",
        name="search_clients"
    )
    async def search_clients(self, query: str, search_type: str = "all") -> str:
        """
        Search clients by name, email, company, or phone
        
        Args:
            query: Search query text
            search_type: Type of search (name, email, company, phone, all)
            
        Returns:
            JSON string containing matching clients
        """
        try:
            from database import get_clients_collection
            
            clients_collection = get_clients_collection()
            
            # Build search query based on type
            search_dict = {}
            regex = re.compile(re.escape(query), re.IGNORECASE)
            
            if search_type == "name":
                search_dict["name"] = {"$regex": regex}
            elif search_type == "email":
                search_dict["email"] = {"$regex": regex}
            elif search_type == "company":
                search_dict["company"] = {"$regex": regex}
            elif search_type == "phone":
                search_dict["phone"] = {"$regex": regex}
            else:  # search_type == "all"
                search_dict["$or"] = [
                    {"name": {"$regex": regex}},
                    {"email": {"$regex": regex}},
                    {"company": {"$regex": regex}},
                    {"phone": {"$regex": regex}}
                ]

            # Get matching clients
            clients_cursor = clients_collection.find(search_dict).sort("name", 1)
            clients = []
            async for client_doc in clients_cursor:
                client_response = {
                    "id": str(client_doc["_id"]),
                    "name": client_doc.get("name", ""),
                    "email": client_doc.get("email", ""),
                    "phone": client_doc.get("phone", ""),
                    "address": client_doc.get("address", ""),
                    "company": client_doc.get("company"),
                    "balance": client_doc.get("balance", 0.0),
                    "status": client_doc.get("status", "active"),
                    "notes": client_doc.get("notes"),
                    "created_at": client_doc.get("created_at", "").isoformat() if isinstance(client_doc.get("created_at"), datetime) else client_doc.get("created_at", ""),
                    "updated_at": client_doc.get("updated_at", "").isoformat() if isinstance(client_doc.get("updated_at"), datetime) else client_doc.get("updated_at", "")
                }
                clients.append(client_response)

            response = {
                "clients": clients,
                "total": len(clients),
                "query": query,
                "search_type": search_type
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to search clients: {str(e)}"})

    # ===== HELPER METHODS =====

    def _extract_balance_from_description(self, description: str) -> Optional[float]:
        """
        Extract balance amount from description
        """
        # Pattern for balance amounts
        balance_patterns = [
            r'balance[:\s]*[€$£]?(\d+(?:\.\d+)?)',
            r'owes[:\s]*[€$£]?(\d+(?:\.\d+)?)',
            r'debt[:\s]*[€$£]?(\d+(?:\.\d+)?)',
            r'[€$£]?(\d+(?:\.\d+)?)\s*(?:balance|owed|debt)'
        ]
        
        for pattern in balance_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None

    def _extract_client_from_description(self, description: str) -> Dict[str, Any]:
        """
        Extract client information from description
        """
        client_data = {
            "name": "",
            "email": "",
            "phone": "",
            "address": "",
            "company": "",
            "notes": ""
        }
        
        # Extract name patterns
        name_patterns = [
            r'(?:client|customer|person)\s+(?:named\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:for|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+at|\s+from|\s+works)',
            r'name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)'  # Name at start of description
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description)
            if match:
                client_data["name"] = match.group(1).strip()
                break
        
        # Extract email
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, description)
        if email_match:
            client_data["email"] = email_match.group(1)
        
        # Extract phone (more comprehensive patterns)
        phone_patterns = [
            r'(?:phone|tel|mobile|cell)[:\s]*([+\d\s\-\(\)\.]{10,})',
            r'(\+33\s*[1-9](?:\s*\d{2}){4})',  # French format
            r'(\+1\s*\(\d{3}\)\s*\d{3}-\d{4})',  # US format
            r'(\d{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2})',  # French mobile
            r'(\(\d{3}\)\s*\d{3}-\d{4})',  # US format without country code
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'  # General format
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, description, re.IGNORECASE)
            if phone_match:
                client_data["phone"] = phone_match.group(1).strip()
                break
        
        # Extract address (comprehensive patterns)
        address_patterns = [
            r'(?:address|lives?\s+at|located\s+at)[:\s]*([^,\.;]+(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|place|pl|way|lane|ln)[^,\.;]*)',
            r'(?:at|address)[:\s]*(\d+[^,\.;]*(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|place|pl|way|lane|ln)[^,\.;]*)',
            r'(\d+\s+[^,\.;]+(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|place|pl|way|lane|ln)[^,\.;]*)',
            r'(?:address|lives?)[:\s]*([^,\.;]+ \d{5}[^,\.;]*)',  # Address with postal code
            r'(?:address)[:\s]*([^,\.;]{20,})'  # Long address-like strings
        ]
        
        for pattern in address_patterns:
            address_match = re.search(pattern, description, re.IGNORECASE)
            if address_match:
                client_data["address"] = address_match.group(1).strip()
                break
        
        # Extract company (comprehensive patterns)
        company_patterns = [
            r'(?:works?\s+at|employed\s+by|company)[:\s]*([^,\.;]+(?:inc|ltd|llc|corp|sa|sas|sarl|eurl)\.?)',
            r'(?:company|corporation|business)[:\s]*([^,\.;]+)',
            r'([^,\.;]+(?:company|corp|inc|ltd|llc|sa|sas|sarl|eurl)\.?)',
            r'(?:from|at)\s+([A-Z][^,\.;]*(?:company|corp|inc|ltd|llc|sa|sas|sarl|eurl)\.?)',
            r'CEO\s+of\s+([^,\.;]+)',
            r'owner\s+of\s+([^,\.;]+)'
        ]
        
        for pattern in company_patterns:
            company_match = re.search(pattern, description, re.IGNORECASE)
            if company_match:
                client_data["company"] = company_match.group(1).strip()
                break
        
        # Extract notes
        note_patterns = [
            r'(?:note|notes|comment|comments|additional\s+info)[:\s]*([^,\.;]+)',
            r'(?:special|important|remember)[:\s]*([^,\.;]+)',
            r'(?:background|history)[:\s]*([^,\.;]+)'
        ]
        
        for pattern in note_patterns:
            note_match = re.search(pattern, description, re.IGNORECASE)
            if note_match:
                client_data["notes"] = note_match.group(1).strip()
                break
        
        # If no explicit notes found, look for descriptive information
        if not client_data["notes"]:
            descriptive_patterns = [
                r'(?:good|excellent|loyal|reliable|difficult|problematic)\s+client',
                r'(?:since|from)\s+\d{4}',
                r'(?:always|never|often|sometimes)\s+[^,\.;]+'
            ]
            
            for pattern in descriptive_patterns:
                desc_match = re.search(pattern, description, re.IGNORECASE)
                if desc_match:
                    client_data["notes"] = desc_match.group(0).strip()
                    break
        
        return client_data