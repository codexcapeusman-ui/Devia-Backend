"""
Manual Task tools for Semantic Kernel
These tools handle manual task creation, scheduling, and management with database connectivity
"""

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from config.settings import Settings
from database import get_manual_tasks_collection


class ManualTaskTools:
    """
    Semantic Kernel tools for manual task management
    Provides AI-powered manual task creation and management from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize manual task tools with application settings"""
        self.settings = settings
        self.company_name = settings.company_name
        self.default_currency = settings.default_currency

    # =============================================
    # CREATE/UPDATE/DELETE TOOLS (Return structured responses for frontend verification)
    # =============================================

    @kernel_function(
        description="Create a new manual task from natural language description and return API call structure for frontend verification",
        name="create_manual_task_api_call"
    )
    def create_manual_task_api_call(self, description: str, client_id: Optional[str] = None) -> str:
        """
        Parse natural language and create a manual task API call payload
        
        Args:
            description: Natural language description of the manual task
            client_id: Optional client ID if known
            
        Returns:
            JSON string containing API call structure matching ManualTaskCreate model
            
        Example:
            Input: "Create a placo work task for ABC Corp tomorrow from 9 AM to 5 PM, mark it red, bring tools"
            Output: JSON with action, endpoint, method, data, preview, and validation
        """
        try:
            # Extract task information from description
            task_data = {
                "title": self._extract_task_title(description),
                "clientId": client_id or self._extract_client_id_from_description(description),
                "startTime": self._extract_start_time(description),
                "endTime": self._extract_end_time(description),
                "color": self._extract_color_from_description(description) or "#ff0000",
                "notes": self._extract_notes_from_description(description),
                "assignedTo": self._extract_assigned_to(description),
                "location": self._extract_location_from_description(description),
                "isAllDay": self._extract_is_all_day(description)
            }

            # Validate required fields
            validation_result = self._validate_create_task_data(task_data)
            
            if not validation_result["is_valid"]:
                return json.dumps({
                    "error": "Validation failed",
                    "validation_errors": validation_result["errors"],
                    "action": None
                }, indent=2)

            # Create API call structure
            api_call = {
                "action": "create_manual_task",
                "endpoint": "/api/v1/manual-tasks/",
                "method": "POST",
                "data": task_data,
                "preview": {
                    "title": task_data["title"],
                    "start_time": task_data["startTime"],
                    "end_time": task_data["endTime"],
                    "client_id": task_data["clientId"],
                    "color": task_data["color"],
                    "assigned_to": task_data["assignedTo"],
                    "location": task_data["location"],
                    "notes": task_data["notes"],
                    "is_all_day": task_data["isAllDay"],
                    "duration_hours": self._calculate_duration_hours(task_data["startTime"], task_data["endTime"])
                },
                "validation": validation_result
            }

            return json.dumps(api_call, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to create manual task API call: {str(e)}"}, indent=2)

    @kernel_function(
        description="Update an existing manual task from natural language description and return API call structure",
        name="update_manual_task_api_call"
    )
    def update_manual_task_api_call(self, task_id: str, description: str) -> str:
        """
        Parse natural language and create a manual task update API call payload
        
        Args:
            task_id: The ID of the task to update
            description: Natural language description of updates
            
        Returns:
            JSON string containing API call structure matching ManualTaskUpdate model
            
        Example:
            Input: task_id="507f1f77bcf86cd799439011", description="Change time to 2 PM to 6 PM and make it yellow"
            Output: JSON with update structure
        """
        try:
            # Extract only fields that should be updated
            update_data = {}
            
            # Check if each field is mentioned in description
            title = self._extract_task_title(description)
            if title:
                update_data["title"] = title
            
            start_time = self._extract_start_time(description)
            if start_time:
                update_data["startTime"] = start_time
            
            end_time = self._extract_end_time(description)
            if end_time:
                update_data["endTime"] = end_time
            
            color = self._extract_color_from_description(description)
            if color:
                update_data["color"] = color
            
            notes = self._extract_notes_from_description(description)
            if notes:
                update_data["notes"] = notes
            
            assigned_to = self._extract_assigned_to(description)
            if assigned_to:
                update_data["assignedTo"] = assigned_to
            
            location = self._extract_location_from_description(description)
            if location:
                update_data["location"] = location
            
            is_all_day = self._extract_is_all_day(description)
            if is_all_day is not None:
                update_data["isAllDay"] = is_all_day
            
            client_id = self._extract_client_id_from_description(description)
            if client_id:
                update_data["clientId"] = client_id

            # Validate task ID
            if not self._is_valid_task_id(task_id):
                return json.dumps({
                    "error": "Invalid task ID format"
                }, indent=2)

            # Create API call structure
            api_call = {
                "action": "update_manual_task",
                "endpoint": f"/api/v1/manual-tasks/{task_id}",
                "method": "PUT",
                "task_id": task_id,
                "data": update_data if update_data else {"notes": f"Updated at {datetime.utcnow().isoformat()}"},
                "preview": {
                    "fields_updated": list(update_data.keys()),
                    "update_count": len(update_data)
                }
            }

            return json.dumps(api_call, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to create manual task update API call: {str(e)}"}, indent=2)

    @kernel_function(
        description="Delete a manual task and return API call structure for frontend verification",
        name="delete_manual_task_api_call"
    )
    def delete_manual_task_api_call(self, task_id: str, description: str = "") -> str:
        """
        Create a manual task delete API call structure
        
        Args:
            task_id: The ID of the task to delete
            description: Optional description for confirmation
            
        Returns:
            JSON string containing API call structure for deletion
        """
        try:
            # Validate task ID
            if not self._is_valid_task_id(task_id):
                return json.dumps({
                    "error": "Invalid task ID format"
                }, indent=2)

            # Create API call structure
            api_call = {
                "action": "delete_manual_task",
                "endpoint": f"/api/v1/manual-tasks/{task_id}",
                "method": "DELETE",
                "task_id": task_id,
                "preview": {
                    "message": "This will permanently delete the manual task",
                    "task_id": task_id,
                    "description": description
                }
            }

            return json.dumps(api_call, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to create manual task delete API call: {str(e)}"}, indent=2)

    # =============================================
    # DATABASE-CONNECTED GET OPERATIONS (Return actual data)
    # =============================================

    @kernel_function(
        description="Get all manual tasks from database with optional filtering - returns actual data",
        name="get_manual_tasks"
    )
    async def get_manual_tasks(self, 
                user_id: str,
                skip: int = 0, 
                limit: int = 100, 
                client_id: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> str:
        """
        Retrieve manual tasks from database with filtering
        
        Args:
            user_id: User ID (required for security)
            skip: Number of tasks to skip for pagination
            limit: Maximum number of tasks to return
            client_id: Filter by client ID
            start_date: Filter tasks starting from this date (ISO format)
            end_date: Filter tasks ending before this date (ISO format)
            
        Returns:
            JSON string containing actual manual tasks data from database
        """
        try:
            print(f"[DEBUG] ManualTaskTools.get_manual_tasks() - User ID: {user_id} (type: {type(user_id)})")
            from database import is_connected
            print(f"[DEBUG] ManualTaskTools.get_manual_tasks() - Database connected: {is_connected()}")
            return await self._get_manual_tasks_async(skip, limit, user_id, client_id, start_date, end_date)
        except Exception as e:
            return json.dumps({"error": f"Failed to get manual tasks: {str(e)}", "tasks": [], "total": 0})

    async def _get_manual_tasks_async(self, skip, limit, user_id, client_id, start_date, end_date):
        """Async implementation for getting manual tasks"""
        try:
            tasks_collection = get_manual_tasks_collection()
            
            # Build query filter
            query = {"userId": user_id}
            
            print(f"[DEBUG] Manual task query: {query}")
            
            # Check total documents in collection
            total_docs = await tasks_collection.count_documents({})
            print(f"[DEBUG] Total manual tasks in collection: {total_docs}")
            
            if client_id:
                query["clientId"] = client_id
            
            # Date range filtering
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["startTime"] = date_filter
            
            # Get total count for pagination
            total = await tasks_collection.count_documents(query)
            
            # Get tasks with pagination
            cursor = tasks_collection.find(query).skip(skip).limit(limit).sort("startTime", 1)
            tasks_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            tasks_data = []
            for task in tasks_list:
                task_data = {
                    "id": str(task["_id"]),
                    "title": task.get("title", ""),
                    "clientId": task.get("clientId", ""),
                    "assignedTo": task.get("assignedTo", ""),
                    "startTime": task.get("startTime").isoformat() if task.get("startTime") else None,
                    "endTime": task.get("endTime").isoformat() if task.get("endTime") else None,
                    "color": task.get("color", "#ff0000"),
                    "notes": task.get("notes", ""),
                    "location": task.get("location", ""),
                    "isAllDay": task.get("isAllDay", False),
                    "createdAt": task.get("createdAt").isoformat() if task.get("createdAt") else datetime.now().isoformat(),
                    "updatedAt": task.get("updatedAt").isoformat() if task.get("updatedAt") else datetime.now().isoformat()
                }
                tasks_data.append(task_data)
            
            result = {
                "tasks": tasks_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "tasks": [], "total": 0})

    @kernel_function(
        description="Get a specific manual task by ID from database - returns actual data",
        name="get_manual_task_by_id"
    )
    def get_manual_task_by_id(self, task_id: str, user_id: str) -> str:
        """
        Retrieve a specific manual task by its ID from database
        
        Args:
            task_id: The manual task ID to retrieve
            user_id: User ID (required for security)
            
        Returns:
            JSON string containing actual manual task data from database
        """
        try:
            return asyncio.run(self._get_manual_task_by_id_async(task_id, user_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to get manual task: {str(e)}", "task": None})

    async def _get_manual_task_by_id_async(self, task_id, user_id):
        """Async implementation for getting manual task by ID"""
        try:
            tasks_collection = get_manual_tasks_collection()
            
            # Convert string ID to ObjectId if valid
            try:
                object_id = ObjectId(task_id)
            except:
                return json.dumps({"error": "Invalid task ID format", "task": None})
            
            # Build query with user_id filter
            query = {"_id": object_id, "userId": user_id}
            
            task = await tasks_collection.find_one(query)
            
            if not task:
                return json.dumps({"error": "Manual task not found", "task": None})
            
            # Format task data
            task_data = {
                "id": str(task["_id"]),
                "title": task.get("title", ""),
                "clientId": task.get("clientId", ""),
                "assignedTo": task.get("assignedTo", ""),
                "startTime": task.get("startTime").isoformat() if task.get("startTime") else None,
                "endTime": task.get("endTime").isoformat() if task.get("endTime") else None,
                "color": task.get("color", "#ff0000"),
                "notes": task.get("notes", ""),
                "location": task.get("location", ""),
                "isAllDay": task.get("isAllDay", False),
                "createdAt": task.get("createdAt").isoformat() if task.get("createdAt") else datetime.now().isoformat(),
                "updatedAt": task.get("updatedAt").isoformat() if task.get("updatedAt") else datetime.now().isoformat()
            }
            
            result = {"task": task_data}
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "task": None})

    @kernel_function(
        description="Get all manual tasks for a specific date range - returns actual data",
        name="get_manual_tasks_by_date_range"
    )
    def get_manual_tasks_by_date_range(self, user_id: str, start_date: str, end_date: str) -> str:
        """
        Retrieve manual tasks within a specific date range
        
        Args:
            user_id: User ID (required for security)
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            JSON string containing manual tasks within the date range
        """
        try:
            return asyncio.run(self._get_manual_tasks_by_date_range_async(user_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to get manual tasks: {str(e)}", "tasks": [], "total": 0})

    async def _get_manual_tasks_by_date_range_async(self, user_id, start_date, end_date):
        """Async implementation for getting manual tasks by date range"""
        try:
            tasks_collection = get_manual_tasks_collection()
            
            # Parse dates
            try:
                start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return json.dumps({"error": "Invalid date format. Use ISO format.", "tasks": [], "total": 0})
            
            # Build query
            query = {
                "userId": user_id,
                "startTime": {
                    "$gte": start,
                    "$lte": end
                }
            }
            
            # Get total count
            total = await tasks_collection.count_documents(query)
            
            # Get tasks
            cursor = tasks_collection.find(query).sort("startTime", 1)
            tasks_list = await cursor.to_list(length=None)
            
            # Format response
            tasks_data = []
            for task in tasks_list:
                task_data = {
                    "id": str(task["_id"]),
                    "title": task.get("title", ""),
                    "clientId": task.get("clientId", ""),
                    "startTime": task.get("startTime").isoformat() if task.get("startTime") else None,
                    "endTime": task.get("endTime").isoformat() if task.get("endTime") else None,
                    "color": task.get("color", "#ff0000"),
                    "notes": task.get("notes", ""),
                    "location": task.get("location", ""),
                    "isAllDay": task.get("isAllDay", False),
                    "assignedTo": task.get("assignedTo", "")
                }
                tasks_data.append(task_data)
            
            return json.dumps({
                "tasks": tasks_data,
                "total": total,
                "date_range": {
                    "start": start.isoformat(),
                    "end": end.isoformat()
                }
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "tasks": [], "total": 0})

    # =============================================
    # HELPER METHODS FOR TEXT EXTRACTION
    # =============================================

    def _extract_task_title(self, text: str) -> str:
        """Extract task title from natural language text"""
        # Look for explicit title patterns
        title_patterns = [
            r'(?:title|task|for)\s*:?\s*"([^"]+)"',
            r'(?:create|schedule|plan|add)\s+(?:task|work)?\s*:?\s*([^,\.;]+?)(?:\s+(?:for|at|on|tomorrow|today|next))',
            r'(?:placo|electric|plumb|paint|cleaning|repair|maintenance|installation)\s+(?:work|job|task)?',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title() if '(' in pattern else match.group(0).strip().title()
        
        # Fallback: use first 50 characters or sentence
        sentences = text.split('.')
        return sentences[0].strip().title()[:100]

    def _extract_start_time(self, text: str) -> str:
        """Extract start time from text and return ISO format"""
        try:
            # Time patterns: "2 PM", "14:00", "tomorrow at 2 PM", "next Monday at 9 AM"
            time_patterns = [
                r'(?:start|from|at)\s+(\d{1,2}:\d{2}(?:\s*(?:AM|PM))?)',
                r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)',
                r'(?:tomorrow|next|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            ]
            
            # Check for relative dates
            if re.search(r'tomorrow', text, re.IGNORECASE):
                base_date = datetime.utcnow() + timedelta(days=1)
            elif re.search(r'today', text, re.IGNORECASE):
                base_date = datetime.utcnow()
            elif re.search(r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', text, re.IGNORECASE):
                day_name = re.search(r'next\s+(\w+)', text, re.IGNORECASE).group(1)
                days_ahead = self._get_next_day(day_name)
                base_date = datetime.utcnow() + timedelta(days=days_ahead)
            else:
                base_date = datetime.utcnow()
            
            # Extract time
            time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', text, re.IGNORECASE)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                am_pm = time_match.group(3)
                
                if am_pm and 'pm' in am_pm.lower() and hour != 12:
                    hour += 12
                elif am_pm and 'am' in am_pm.lower() and hour == 12:
                    hour = 0
                
                start_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return start_time.isoformat()
            
            # Default to 9 AM
            return base_date.replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
        
        except Exception:
            return datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0).isoformat()

    def _extract_end_time(self, text: str) -> str:
        """Extract end time from text and return ISO format"""
        try:
            # Similar logic to start time
            start_time_str = self._extract_start_time(text)
            start_time = datetime.fromisoformat(start_time_str)
            
            # Look for duration patterns
            duration_patterns = [
                r'(?:for|duration)\s+(\d+)\s*(?:hours?)',
                r'(?:until|to|till)\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?',
            ]
            
            for pattern in duration_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'hours' in pattern:
                        hours = int(match.group(1))
                        end_time = start_time + timedelta(hours=hours)
                    else:
                        hour = int(match.group(1))
                        minute = int(match.group(2)) if match.group(2) else 0
                        am_pm = match.group(3)
                        
                        if am_pm and 'pm' in am_pm.lower() and hour != 12:
                            hour += 12
                        elif am_pm and 'am' in am_pm.lower() and hour == 12:
                            hour = 0
                        
                        end_time = start_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    return end_time.isoformat()
            
            # Default: 8 hours after start time
            end_time = start_time + timedelta(hours=8)
            return end_time.isoformat()
        
        except Exception:
            start_time = datetime.fromisoformat(self._extract_start_time(text))
            end_time = start_time + timedelta(hours=8)
            return end_time.isoformat()

    def _extract_color_from_description(self, text: str) -> Optional[str]:
        """Extract hex color from text"""
        # Look for hex color patterns
        hex_pattern = r'#[0-9a-fA-F]{6}\b'
        match = re.search(hex_pattern, text)
        if match:
            return match.group(0)
        
        # Look for color names and convert to hex
        color_map = {
            'red': '#ff0000',
            'blue': '#0000ff',
            'green': '#00ff00',
            'yellow': '#ffff00',
            'orange': '#ff8800',
            'purple': '#800080',
            'pink': '#ff69b4',
            'black': '#000000',
            'white': '#ffffff',
            'gray': '#808080',
        }
        
        for color_name, hex_value in color_map.items():
            if re.search(rf'\b{color_name}\b', text, re.IGNORECASE):
                return hex_value
        
        return None

    def _extract_notes_from_description(self, text: str) -> str:
        """Extract notes from text"""
        # Look for patterns like "note:", "notes:", "remember:", "important:"
        note_patterns = [
            r'(?:note|notes|remember|important|attention|note to self)\s*:?\s*(.+?)(?:\.|$)',
            r'(?:bring|take|don\'t forget|make sure)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in note_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_assigned_to(self, text: str) -> Optional[str]:
        """Extract assigned worker/team from text"""
        # Look for patterns like "assign to:", "assigned to:", "for team:"
        patterns = [
            r'(?:assign|assigned)\s+(?:to|for)\s*:?\s*([^,\.;]+)',
            r'(?:team|worker|person)\s*:?\s*([^,\.;]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_location_from_description(self, text: str) -> str:
        """Extract location/address from text"""
        # Look for location patterns
        location_patterns = [
            r'(?:at|location|address|site|place)\s*:?\s*([^,\.;]+)',
            r'(?:job\s+site|worksite|work\s+at)\s*:?\s*([^,\.;]+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_is_all_day(self, text: str) -> Optional[bool]:
        """Check if task should be marked as all-day"""
        if re.search(r'\ball[\s-]?day\b', text, re.IGNORECASE):
            return True
        if re.search(r'\bwhole\s+day\b', text, re.IGNORECASE):
            return True
        return False

    def _extract_client_id_from_description(self, text: str) -> Optional[str]:
        """Extract client ID from text"""
        # Look for patterns like "for client:", "for:", "client ID:"
        patterns = [
            r'(?:for|client|company)\s*:?\s*([a-zA-Z0-9]+)',
            r'(?:ABC|client)\s+([A-Z][a-zA-Z0-9]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _calculate_duration_hours(self, start_time: str, end_time: str) -> float:
        """Calculate duration in hours between two ISO time strings"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = (end - start).total_seconds() / 3600
            return round(duration, 2)
        except Exception:
            return 0.0

    def _get_next_day(self, day_name: str) -> int:
        """Get days ahead for next occurrence of day name"""
        days = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6,
        }
        
        day_index = days.get(day_name.lower(), 0)
        today = datetime.utcnow().weekday()
        days_ahead = day_index - today
        
        if days_ahead <= 0:
            days_ahead += 7
        
        return days_ahead

    def _is_valid_task_id(self, task_id: str) -> bool:
        """Validate if task ID is a valid MongoDB ObjectId"""
        try:
            ObjectId(task_id)
            return True
        except:
            return False

    # =============================================
    # VALIDATION METHODS
    # =============================================

    def _validate_create_task_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task data for creation"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "required_fields": ["title", "startTime", "endTime"]
        }
        
        # Check required fields
        if not task_data.get("title") or task_data["title"].strip() == "":
            validation_result["errors"].append("Task title is required")
            validation_result["is_valid"] = False
        
        if not task_data.get("startTime"):
            validation_result["errors"].append("Start time is required")
            validation_result["is_valid"] = False
        
        if not task_data.get("endTime"):
            validation_result["errors"].append("End time is required")
            validation_result["is_valid"] = False
        
        # Validate time relationship
        try:
            if task_data.get("startTime") and task_data.get("endTime"):
                start = datetime.fromisoformat(task_data["startTime"].replace('Z', '+00:00'))
                end = datetime.fromisoformat(task_data["endTime"].replace('Z', '+00:00'))
                
                if start >= end:
                    validation_result["errors"].append("Start time must be before end time")
                    validation_result["is_valid"] = False
        except Exception:
            validation_result["errors"].append("Invalid time format")
            validation_result["is_valid"] = False
        
        # Warnings
        if not task_data.get("clientId"):
            validation_result["warnings"].append("Client ID not specified")
        
        if not task_data.get("notes"):
            validation_result["warnings"].append("Consider adding notes for better documentation")
        
        return validation_result
