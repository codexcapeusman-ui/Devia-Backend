"""
Job scheduling tools for Semantic Kernel
These tools handle job creation, scheduling, and management with database connectivity
"""

from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import List, Dict, Any, Optional
import json
import re
import uuid
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import calendar
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from config.settings import Settings
from models import Job, Client, JobStatus
from database import get_jobs_collection, get_clients_collection, get_meetings_collection, get_invoices_collection, get_quotes_collection, get_expenses_collection

class JobTools:
    """
    Semantic Kernel tools for job scheduling and management
    Provides AI-powered job creation from natural language prompts
    """
    
    def __init__(self, settings: Settings):
        """Initialize job tools with application settings"""
        self.settings = settings
        self.company_name = settings.company_name
    
    @kernel_function(
        description="Create a job from natural language text description (legacy format - use create_job_api_call for API structure)",
        name="create_job_from_text"
    )
    def create_job_from_text(self, description: str, client_id: Optional[str] = None) -> str:
        """
        Create a complete job from text description (legacy format)
        
        Args:
            description: Natural language description of the job
            client_id: Optional client ID if known
            
        Returns:
            JSON string containing the created job data (legacy format)
            
        Example:
            Input: "Schedule website maintenance for ABC Corp next Tuesday at 2 PM, should take 3 hours"
            Output: JSON with complete job structure
            
        Note: For API integration, use create_job_api_call instead
        """
        try:
            # Extract job information from description
            job_data = {
                "id": str(uuid.uuid4()),
                "title": "",
                "client_id": client_id or str(uuid.uuid4()),
                "client": None,
                "assigned_to": None,
                "assigned_worker": None,
                "start_time": None,
                "end_time": None,
                "status": "scheduled",
                "location": "",
                "description": description,
                "notes": "",
                "images": [],
                "created_at": datetime.now().isoformat()
            }
            
            # Extract job title
            job_data["title"] = self._extract_job_title(description)
            
            # Extract client information if not provided
            if not client_id:
                client_data = self._extract_client_from_description(description)
                job_data["client"] = client_data
                job_data["client_id"] = client_data.get("id", str(uuid.uuid4()))
            
            # Extract timing information
            start_time, end_time = self._extract_timing_from_description(description)
            job_data["start_time"] = start_time.isoformat() if start_time else None
            job_data["end_time"] = end_time.isoformat() if end_time else None
            
            # Extract location
            job_data["location"] = self._extract_location_from_description(description)
            
            # Extract assigned worker
            job_data["assigned_to"] = self._extract_assigned_worker(description)
            
            # Extract notes
            job_data["notes"] = self._extract_job_notes(description)
            
            # Validate the job data
            job_data = self._validate_job_data(job_data)
            
            return json.dumps(job_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create job: {str(e)}"})
    
    @kernel_function(
        description="Parse scheduling information from natural language text",
        name="parse_schedule_info"
    )
    def parse_schedule_info(self, text: str) -> str:
        """
        Parse date and time information from natural language
        
        Args:
            text: Text containing date/time expressions
            
        Returns:
            JSON string containing parsed scheduling information
            
        Example:
            Input: "next Tuesday at 2 PM for 3 hours"
            Output: JSON with start_time, end_time, duration info
        """
        try:
            schedule_info = {
                "start_time": None,
                "end_time": None,
                "duration_hours": None,
                "parsed_expressions": [],
                "confidence": "medium"
            }
            
            # Extract and parse time expressions
            time_expressions = self._find_time_expressions(text)
            schedule_info["parsed_expressions"] = time_expressions
            
            if time_expressions:
                # Parse the most specific time expression
                start_time, end_time, duration = self._parse_primary_time_expression(time_expressions[0], text)
                
                if start_time:
                    schedule_info["start_time"] = start_time.isoformat()
                    schedule_info["confidence"] = "high"
                
                if end_time:
                    schedule_info["end_time"] = end_time.isoformat()
                elif start_time and duration:
                    end_time = start_time + timedelta(hours=duration)
                    schedule_info["end_time"] = end_time.isoformat()
                
                if duration:
                    schedule_info["duration_hours"] = duration
            
            return json.dumps(schedule_info, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to parse schedule info: {str(e)}"})
    
    @kernel_function(
        description="Validate schedule feasibility and check for conflicts",
        name="validate_schedule"
    )
    def validate_schedule(self, job_json: str, existing_jobs_json: str = "[]") -> str:
        """
        Validate job schedule and check for potential conflicts
        
        Args:
            job_json: JSON string containing job data
            existing_jobs_json: JSON string containing array of existing jobs
            
        Returns:
            JSON string containing validation results
        """
        try:
            job_data = json.loads(job_json)
            existing_jobs = json.loads(existing_jobs_json) if existing_jobs_json != "[]" else []
            
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "suggestions": [],
                "conflicts": []
            }
            
            start_time_str = job_data.get("start_time")
            end_time_str = job_data.get("end_time")
            
            if not start_time_str:
                validation_result["errors"].append("Start time is required")
                validation_result["is_valid"] = False
                return json.dumps(validation_result, indent=2)
            
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00')) if end_time_str else None
            
            # Check if job is in the past
            if start_time < datetime.now():
                validation_result["warnings"].append("Job is scheduled in the past")
            
            # Check if job is too far in the future
            if start_time > datetime.now() + timedelta(days=365):
                validation_result["warnings"].append("Job is scheduled more than a year in advance")
            
            # Check business hours
            if start_time.hour < 8 or start_time.hour > 18:
                validation_result["warnings"].append("Job is scheduled outside normal business hours")
            
            # Check if it's a weekend
            if start_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                validation_result["warnings"].append("Job is scheduled on a weekend")
            
            # Check duration
            if end_time:
                duration = (end_time - start_time).total_seconds() / 3600
                if duration > 8:
                    validation_result["warnings"].append("Job duration exceeds 8 hours")
                elif duration < 0.5:
                    validation_result["warnings"].append("Job duration is very short (less than 30 minutes)")
            
            # Check for conflicts with existing jobs
            assigned_to = job_data.get("assigned_to")
            if assigned_to and existing_jobs:
                conflicts = self._check_schedule_conflicts(start_time, end_time, assigned_to, existing_jobs)
                validation_result["conflicts"] = conflicts
                if conflicts:
                    validation_result["warnings"].append(f"Found {len(conflicts)} potential scheduling conflicts")
            
            # Add suggestions
            if validation_result["warnings"]:
                validation_result["suggestions"].append("Consider adjusting the schedule to avoid potential issues")
            
            return json.dumps(validation_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to validate schedule: {str(e)}"})
    
    @kernel_function(
        description="Suggest optimal scheduling times based on preferences and constraints",
        name="suggest_optimal_times"
    )
    def suggest_optimal_times(self, preferences_json: str, duration_hours: float = 2.0) -> str:
        """
        Suggest optimal scheduling times based on preferences
        
        Args:
            preferences_json: JSON string containing scheduling preferences
            duration_hours: Expected duration of the job in hours
            
        Returns:
            JSON string containing suggested time slots
        """
        try:
            preferences = json.loads(preferences_json) if preferences_json else {}
            
            suggestions = {
                "suggested_times": [],
                "reasoning": [],
                "alternatives": []
            }
            
            # Get base parameters
            preferred_days = preferences.get("preferred_days", ["monday", "tuesday", "wednesday", "thursday", "friday"])
            preferred_hours = preferences.get("preferred_hours", [9, 10, 11, 14, 15, 16])
            avoid_hours = preferences.get("avoid_hours", [12, 13])  # Lunch time
            min_notice_days = preferences.get("min_notice_days", 1)
            
            # Generate suggestions for the next 14 days
            start_date = datetime.now() + timedelta(days=min_notice_days)
            
            for days_ahead in range(14):
                check_date = start_date + timedelta(days=days_ahead)
                day_name = check_date.strftime("%A").lower()
                
                if day_name in preferred_days:
                    for hour in preferred_hours:
                        if hour not in avoid_hours:
                            suggested_start = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                            suggested_end = suggested_start + timedelta(hours=duration_hours)
                            
                            # Check if end time is within business hours
                            if suggested_end.hour <= 18:
                                suggestion = {
                                    "start_time": suggested_start.isoformat(),
                                    "end_time": suggested_end.isoformat(),
                                    "day_name": day_name.title(),
                                    "confidence": "high" if hour in [9, 10, 14, 15] else "medium"
                                }
                                suggestions["suggested_times"].append(suggestion)
                                
                                if len(suggestions["suggested_times"]) >= 5:
                                    break
                
                if len(suggestions["suggested_times"]) >= 5:
                    break
            
            # Add reasoning
            suggestions["reasoning"] = [
                f"Suggested times are within preferred days: {', '.join(preferred_days)}",
                f"Times avoid lunch hours and late afternoon slots",
                f"Duration of {duration_hours} hours is accommodated within business hours",
                f"Minimum notice period of {min_notice_days} days is respected"
            ]
            
            # Add alternatives for weekends if no weekday preferences
            if len(suggestions["suggested_times"]) < 3:
                weekend_times = self._generate_weekend_alternatives(start_date, duration_hours)
                suggestions["alternatives"] = weekend_times
            
            return json.dumps(suggestions, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to suggest optimal times: {str(e)}"})
    
    @kernel_function(
        description="Reschedule an existing job to a new time",
        name="reschedule_job"
    )
    def reschedule_job(self, job_json: str, new_schedule_text: str) -> str:
        """
        Reschedule an existing job to a new time
        
        Args:
            job_json: JSON string containing existing job data
            new_schedule_text: Natural language description of new schedule
            
        Returns:
            JSON string containing updated job data
        """
        try:
            job_data = json.loads(job_json)
            
            # Parse new scheduling information
            schedule_info = json.loads(self.parse_schedule_info(new_schedule_text))
            
            # Store original schedule as backup
            original_schedule = {
                "start_time": job_data.get("start_time"),
                "end_time": job_data.get("end_time")
            }
            
            # Update job with new schedule
            if schedule_info.get("start_time"):
                job_data["start_time"] = schedule_info["start_time"]
            
            if schedule_info.get("end_time"):
                job_data["end_time"] = schedule_info["end_time"]
            
            # Add rescheduling note
            reschedule_note = f"Rescheduled from {original_schedule['start_time']} to {job_data.get('start_time')}"
            existing_notes = job_data.get("notes", "")
            
            if existing_notes:
                job_data["notes"] = f"{existing_notes}\\n\\n{reschedule_note}"
            else:
                job_data["notes"] = reschedule_note
            
            # Update status if needed
            if job_data.get("status") == "completed":
                job_data["status"] = "scheduled"
            
            return json.dumps(job_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to reschedule job: {str(e)}"})

    # =============================================
    # DATABASE-CONNECTED GET OPERATIONS (Return actual data)
    # =============================================

    @kernel_function(
        description="Get all jobs from database with optional filtering - returns actual data",
        name="get_jobs"
    )
    async def get_jobs(self, 
                skip: int = 0, 
                limit: int = 100, 
                client_id: Optional[str] = None,
                assigned_to: Optional[str] = None,
                status: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> str:
        """
        Retrieve jobs from database with filtering
        
        Args:
            skip: Number of jobs to skip for pagination
            limit: Maximum number of jobs to return
            client_id: Filter by client ID
            assigned_to: Filter by assigned user ID
            status: Filter by job status (scheduled, in_progress, completed, cancelled)
            start_date: Filter jobs starting from this date (ISO format)
            end_date: Filter jobs ending before this date (ISO format)
            
        Returns:
            JSON string containing actual jobs data from database
        """
        try:
            # Run async function in sync context
            return await self._get_jobs_async(skip, limit, client_id, assigned_to, status, start_date, end_date)
        except Exception as e:
            return json.dumps({"error": f"Failed to get jobs: {str(e)}", "jobs": [], "total": 0})

    async def _get_jobs_async(self, skip, limit, client_id, assigned_to, status, start_date, end_date):
        """Async implementation for getting jobs"""
        try:
            jobs_collection = get_jobs_collection()
            
            # Build query filter
            query = {}
            
            if client_id:
                query["clientId"] = client_id
            if assigned_to:
                query["assignedTo"] = assigned_to
            if status:
                query["status"] = status
            
            # Date range filtering
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["startTime"] = date_filter
            
            # Get total count for pagination
            total = await jobs_collection.count_documents(query)
            
            # Get jobs with pagination
            cursor = jobs_collection.find(query).skip(skip).limit(limit).sort("startTime", 1)
            jobs_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            jobs_data = []
            for job in jobs_list:
                job_data = {
                    "id": str(job["_id"]),
                    "title": job.get("title", ""),
                    "clientId": job.get("clientId", ""),
                    "assignedTo": job.get("assignedTo", ""),
                    "startTime": job.get("startTime").isoformat() if job.get("startTime") else None,
                    "endTime": job.get("endTime").isoformat() if job.get("endTime") else None,
                    "status": job.get("status", "scheduled"),
                    "location": job.get("location", ""),
                    "description": job.get("description", ""),
                    "createdAt": job.get("createdAt").isoformat() if job.get("createdAt") else datetime.now().isoformat(),
                    "updatedAt": job.get("updatedAt").isoformat() if job.get("updatedAt") else datetime.now().isoformat()
                }
                jobs_data.append(job_data)
            
            result = {
                "jobs": jobs_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "jobs": [], "total": 0})

    @kernel_function(
        description="Get a specific job by ID from database - returns actual data",
        name="get_job_by_id"
    )
    def get_job_by_id(self, job_id: str) -> str:
        """
        Retrieve a specific job by its ID from database
        
        Args:
            job_id: The job ID to retrieve
            
        Returns:
            JSON string containing actual job data from database
        """
        try:
            return asyncio.run(self._get_job_by_id_async(job_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to get job: {str(e)}", "job": None})

    async def _get_job_by_id_async(self, job_id):
        """Async implementation for getting job by ID"""
        try:
            jobs_collection = get_jobs_collection()
            
            # Convert string ID to ObjectId if valid
            try:
                object_id = ObjectId(job_id)
            except:
                return json.dumps({"error": "Invalid job ID format", "job": None})
            
            job = await jobs_collection.find_one({"_id": object_id})
            
            if not job:
                return json.dumps({"error": "Job not found", "job": None})
            
            # Format job data
            job_data = {
                "id": str(job["_id"]),
                "title": job.get("title", ""),
                "clientId": job.get("clientId", ""),
                "assignedTo": job.get("assignedTo", ""),
                "startTime": job.get("startTime").isoformat() if job.get("startTime") else None,
                "endTime": job.get("endTime").isoformat() if job.get("endTime") else None,
                "status": job.get("status", "scheduled"),
                "location": job.get("location", ""),
                "description": job.get("description", ""),
                "createdAt": job.get("createdAt").isoformat() if job.get("createdAt") else datetime.now().isoformat(),
                "updatedAt": job.get("updatedAt").isoformat() if job.get("updatedAt") else datetime.now().isoformat()
            }
            
            result = {"job": job_data}
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "job": None})

    @kernel_function(
        description="Get calendar overview with jobs, meetings, invoices, and quotes - returns actual data",
        name="get_calendar_overview"
    )
    def get_calendar_overview(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            event_types: Optional[List[str]] = None,
                            client_id: Optional[str] = None) -> str:
        """
        Get calendar overview combining all event types from database
        
        Args:
            start_date: Start date for filtering (ISO format)
            end_date: End date for filtering (ISO format)
            event_types: List of event types to include (job, meeting, invoice, quote)
            client_id: Filter by client ID
            
        Returns:
            JSON string containing actual calendar data from database
        """
        try:
            return asyncio.run(self._get_calendar_overview_async(start_date, end_date, event_types, client_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to get calendar overview: {str(e)}", "events": [], "total": 0})

    async def _get_calendar_overview_async(self, start_date, end_date, event_types, client_id):
        """Async implementation for getting calendar overview"""
        try:
            all_events = []
            
            # Default to all event types if none specified
            if not event_types:
                event_types = ["job", "meeting", "invoice", "quote"]
            
            # Set default date range if not provided (next 30 days)
            if not start_date:
                start_date = datetime.now().isoformat()
            if not end_date:
                end_date = (datetime.now() + timedelta(days=30)).isoformat()
            
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Get Jobs
            if "job" in event_types:
                jobs_collection = get_jobs_collection()
                job_query = {"startTime": {"$gte": start_dt, "$lte": end_dt}}
                if client_id:
                    job_query["clientId"] = client_id
                
                jobs_cursor = jobs_collection.find(job_query)
                jobs = await jobs_cursor.to_list(length=None)
                
                for job in jobs:
                    event = {
                        "id": str(job["_id"]),
                        "title": job.get("title", ""),
                        "type": "job",
                        "date": job.get("startTime").isoformat() if job.get("startTime") else None,
                        "clientId": job.get("clientId"),
                        "status": job.get("status"),
                        "description": job.get("description", "")
                    }
                    all_events.append(event)
            
            # Get Meetings
            if "meeting" in event_types:
                meetings_collection = get_meetings_collection()
                meeting_query = {"startTime": {"$gte": start_dt, "$lte": end_dt}}
                
                meetings_cursor = meetings_collection.find(meeting_query)
                meetings = await meetings_cursor.to_list(length=None)
                
                for meeting in meetings:
                    event = {
                        "id": str(meeting["_id"]),
                        "title": meeting.get("title", ""),
                        "type": "meeting",
                        "date": meeting.get("startTime").isoformat() if meeting.get("startTime") else None,
                        "clientId": None,  # Meetings don't have direct client association
                        "status": meeting.get("status"),
                        "description": meeting.get("description", "")
                    }
                    all_events.append(event)
            
            # Get Invoices
            if "invoice" in event_types:
                invoices_collection = get_invoices_collection()
                invoice_query = {}
                if client_id:
                    invoice_query["client_id"] = client_id
                
                # Filter invoices by due date or creation date
                if "dueDate" in invoice_query:
                    invoice_query["dueDate"] = {"$gte": start_dt, "$lte": end_dt}
                else:
                    invoice_query["createdAt"] = {"$gte": start_dt, "$lte": end_dt}
                
                invoices_cursor = invoices_collection.find(invoice_query)
                invoices = await invoices_cursor.to_list(length=None)
                
                for invoice in invoices:
                    event = {
                        "id": str(invoice["_id"]),
                        "title": f"Invoice #{invoice.get('invoice_number', 'N/A')}",
                        "type": "invoice",
                        "date": invoice.get("dueDate").isoformat() if invoice.get("dueDate") else invoice.get("createdAt").isoformat(),
                        "clientId": invoice.get("client_id"),
                        "status": invoice.get("status"),
                        "description": f"Amount: ${invoice.get('total_amount', 0)}"
                    }
                    all_events.append(event)
            
            # Get Quotes
            if "quote" in event_types:
                quotes_collection = get_quotes_collection()
                quote_query = {}
                if client_id:
                    quote_query["client_id"] = client_id
                
                quote_query["createdAt"] = {"$gte": start_dt, "$lte": end_dt}
                
                quotes_cursor = quotes_collection.find(quote_query)
                quotes = await quotes_cursor.to_list(length=None)
                
                for quote in quotes:
                    event = {
                        "id": str(quote["_id"]),
                        "title": f"Quote #{quote.get('quote_number', 'N/A')}",
                        "type": "quote",
                        "date": quote.get("createdAt").isoformat() if quote.get("createdAt") else None,
                        "clientId": quote.get("client_id"),
                        "status": quote.get("status"),
                        "description": f"Amount: ${quote.get('total_amount', 0)}"
                    }
                    all_events.append(event)
            
            # Sort events by date
            all_events.sort(key=lambda x: x.get("date", ""))
            
            result = {
                "events": all_events,
                "total": len(all_events)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "events": [], "total": 0})

    # =============================================
    # STRUCTURED API RESPONSE TOOLS (For CREATE/UPDATE/DELETE operations)
    # =============================================

    @kernel_function(
        description="Create a job and return API call structure for frontend verification",
        name="create_job_api_call"
    )
    def create_job_api_call(self, description: str, client_id: Optional[str] = None) -> str:
        """
        Create job data and return structured API call for frontend
        
        Args:
            description: Natural language description of the job
            client_id: Optional client ID if known
            
        Returns:
            JSON string containing API endpoint, method, data, and preview
        """
        try:
            # Use existing logic to extract job data
            job_data_json = self.create_job_from_text(description, client_id)
            job_data = json.loads(job_data_json)
            
            if "error" in job_data:
                return job_data_json
            
            # Convert to API format (JobCreate model)
            api_data = {
                "title": job_data.get("title", "Service Appointment"),
                "clientId": job_data.get("client_id", client_id or ""),
                "assignedTo": job_data.get("assigned_to", ""),
                "startTime": job_data.get("start_time"),
                "endTime": job_data.get("end_time"),
                "location": job_data.get("location", ""),
                "description": job_data.get("description", "")
            }
            
            # Structure API call response
            api_call = {
                "action": "create_job",
                "endpoint": "/api/v1/calendar/",
                "method": "POST",
                "data": api_data,
                "sync_google": False,
                "preview": {
                    "title": api_data["title"],
                    "start_time": api_data["startTime"],
                    "end_time": api_data["endTime"],
                    "location": api_data["location"],
                    "description": f"Job creation from: {description}"
                },
                "validation": {
                    "required_fields": ["title", "clientId", "assignedTo", "startTime", "endTime"],
                    "missing_fields": [field for field in ["title", "clientId", "assignedTo", "startTime", "endTime"] 
                                     if not api_data.get(field)]
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create job API call: {str(e)}"})

    @kernel_function(
        description="Update a job and return API call structure for frontend verification",
        name="update_job_api_call"
    )
    def update_job_api_call(self, job_id: str, update_description: str) -> str:
        """
        Update job data and return structured API call for frontend
        
        Args:
            job_id: ID of the job to update
            update_description: Natural language description of updates
            
        Returns:
            JSON string containing API endpoint, method, data, and preview
        """
        try:
            # Parse update information from description
            updates = {}
            
            # Extract new timing if mentioned
            schedule_info_json = self.parse_schedule_info(update_description)
            schedule_info = json.loads(schedule_info_json)
            
            if schedule_info.get("start_time"):
                updates["startTime"] = schedule_info["start_time"]
            if schedule_info.get("end_time"):
                updates["endTime"] = schedule_info["end_time"]
            
            # Extract other updates
            if "title" in update_description.lower() or "name" in update_description.lower():
                new_title = self._extract_job_title(update_description)
                if new_title != "Service Appointment":
                    updates["title"] = new_title
            
            if "location" in update_description.lower():
                new_location = self._extract_location_from_description(update_description)
                if new_location != "Client Location":
                    updates["location"] = new_location
            
            if "assign" in update_description.lower():
                new_worker = self._extract_assigned_worker(update_description)
                if new_worker:
                    updates["assignedTo"] = new_worker
            
            # Check for status updates
            status_keywords = {
                "complete": "completed",
                "finish": "completed", 
                "done": "completed",
                "cancel": "cancelled",
                "start": "in_progress",
                "begin": "in_progress"
            }
            
            for keyword, status in status_keywords.items():
                if keyword in update_description.lower():
                    updates["status"] = status
                    break
            
            # Structure API call response
            api_call = {
                "action": "update_job",
                "endpoint": f"/api/v1/calendar/{job_id}",
                "method": "PUT", 
                "data": updates,
                "sync_google": False,
                "preview": {
                    "job_id": job_id,
                    "updates": updates,
                    "description": f"Job update from: {update_description}"
                },
                "validation": {
                    "has_updates": len(updates) > 0,
                    "update_fields": list(updates.keys())
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create update API call: {str(e)}"})

    @kernel_function(
        description="Delete a job and return API call structure for frontend verification", 
        name="delete_job_api_call"
    )
    def delete_job_api_call(self, job_id: str, remove_from_google: bool = False) -> str:
        """
        Delete job and return structured API call for frontend
        
        Args:
            job_id: ID of the job to delete
            remove_from_google: Whether to also remove from Google Calendar
            
        Returns:
            JSON string containing API endpoint, method, and preview
        """
        try:
            api_call = {
                "action": "delete_job",
                "endpoint": f"/api/v1/calendar/{job_id}",
                "method": "DELETE",
                "params": {
                    "remove_google": remove_from_google
                },
                "preview": {
                    "job_id": job_id,
                    "will_remove_from_google": remove_from_google,
                    "description": f"Delete job {job_id}"
                },
                "validation": {
                    "job_id_valid": len(job_id) == 24,  # ObjectId length check
                    "confirmation_required": True
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create delete API call: {str(e)}"})

    # =============================================
    # MEETING MANAGEMENT TOOLS
    # =============================================

    @kernel_function(
        description="Get all meetings from database with optional filtering - returns actual data",
        name="get_meetings"
    )
    async def get_meetings(self,
                    skip: int = 0,
                    limit: int = 100,
                    organizer_id: Optional[str] = None,
                    status: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> str:
        """
        Retrieve meetings from database with filtering
        
        Args:
            skip: Number of meetings to skip for pagination
            limit: Maximum number of meetings to return
            organizer_id: Filter by organizer user ID
            status: Filter by meeting status (scheduled, confirmed, cancelled)
            start_date: Filter meetings starting from this date (ISO format)
            end_date: Filter meetings ending before this date (ISO format)
            
        Returns:
            JSON string containing actual meetings data from database
        """
        try:
            return await self._get_meetings_async(skip, limit, organizer_id, status, start_date, end_date)
        except Exception as e:
            return json.dumps({"error": f"Failed to get meetings: {str(e)}", "meetings": [], "total": 0})

    async def _get_meetings_async(self, skip, limit, organizer_id, status, start_date, end_date):
        """Async implementation for getting meetings"""
        try:
            meetings_collection = get_meetings_collection()
            
            # Build query filter
            query = {}
            
            if organizer_id:
                query["organizerId"] = organizer_id
            if status:
                query["status"] = status
            
            # Date range filtering
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["startTime"] = date_filter
            
            # Get total count for pagination
            total = await meetings_collection.count_documents(query)
            
            # Get meetings with pagination
            cursor = meetings_collection.find(query).skip(skip).limit(limit).sort("startTime", 1)
            meetings_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            meetings_data = []
            for meeting in meetings_list:
                meeting_data = {
                    "id": str(meeting["_id"]),
                    "title": meeting.get("title", ""),
                    "description": meeting.get("description", ""),
                    "startTime": meeting.get("startTime").isoformat() if meeting.get("startTime") else None,
                    "endTime": meeting.get("endTime").isoformat() if meeting.get("endTime") else None,
                    "location": meeting.get("location", ""),
                    "attendees": meeting.get("attendees", []),
                    "organizerId": meeting.get("organizerId", ""),
                    "status": meeting.get("status", "scheduled"),
                    "googleEventId": meeting.get("googleEventId"),
                    "createdAt": meeting.get("createdAt").isoformat() if meeting.get("createdAt") else datetime.now().isoformat(),
                    "updatedAt": meeting.get("updatedAt").isoformat() if meeting.get("updatedAt") else datetime.now().isoformat()
                }
                meetings_data.append(meeting_data)
            
            result = {
                "meetings": meetings_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "meetings": [], "total": 0})

    @kernel_function(
        description="Get a specific meeting by ID from database - returns actual data",
        name="get_meeting_by_id"
    )
    def get_meeting_by_id(self, meeting_id: str) -> str:
        """
        Retrieve a specific meeting by its ID from database
        
        Args:
            meeting_id: The meeting ID to retrieve
            
        Returns:
            JSON string containing actual meeting data from database
        """
        try:
            return asyncio.run(self._get_meeting_by_id_async(meeting_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to get meeting: {str(e)}", "meeting": None})

    async def _get_meeting_by_id_async(self, meeting_id):
        """Async implementation for getting meeting by ID"""
        try:
            meetings_collection = get_meetings_collection()
            
            # Convert string ID to ObjectId if valid
            try:
                object_id = ObjectId(meeting_id)
            except:
                return json.dumps({"error": "Invalid meeting ID format", "meeting": None})
            
            meeting = await meetings_collection.find_one({"_id": object_id})
            
            if not meeting:
                return json.dumps({"error": "Meeting not found", "meeting": None})
            
            # Format meeting data
            meeting_data = {
                "id": str(meeting["_id"]),
                "title": meeting.get("title", ""),
                "description": meeting.get("description", ""),
                "startTime": meeting.get("startTime").isoformat() if meeting.get("startTime") else None,
                "endTime": meeting.get("endTime").isoformat() if meeting.get("endTime") else None,
                "location": meeting.get("location", ""),
                "attendees": meeting.get("attendees", []),
                "organizerId": meeting.get("organizerId", ""),
                "status": meeting.get("status", "scheduled"),
                "googleEventId": meeting.get("googleEventId"),
                "createdAt": meeting.get("createdAt").isoformat() if meeting.get("createdAt") else datetime.now().isoformat(),
                "updatedAt": meeting.get("updatedAt").isoformat() if meeting.get("updatedAt") else datetime.now().isoformat()
            }
            
            result = {"meeting": meeting_data}
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "meeting": None})

    @kernel_function(
        description="Create a meeting from natural language and return API call structure",
        name="create_meeting_from_text"
    )
    def create_meeting_from_text(self, description: str, organizer_id: Optional[str] = None) -> str:
        """
        Create a meeting from text description and return API call structure
        
        Args:
            description: Natural language description of the meeting
            organizer_id: Optional organizer ID
            
        Returns:
            JSON string containing API endpoint, method, data, and preview
        """
        try:
            # Extract meeting information from description
            meeting_data = {
                "title": self._extract_meeting_title(description),
                "description": self._extract_meeting_description(description),
                "startTime": None,
                "endTime": None,
                "location": self._extract_location_from_description(description),
                "attendeeEmails": self._extract_attendee_emails(description)
            }
            
            # Extract timing information
            start_time, end_time = self._extract_timing_from_description(description)
            meeting_data["startTime"] = start_time.isoformat() if start_time else None
            meeting_data["endTime"] = end_time.isoformat() if end_time else None
            
            # Structure API call response
            api_call = {
                "action": "create_meeting",
                "endpoint": "/api/v1/calendar/meetings/",
                "method": "POST",
                "data": meeting_data,
                "sync_google": False,
                "preview": {
                    "title": meeting_data["title"],
                    "start_time": meeting_data["startTime"],
                    "end_time": meeting_data["endTime"],
                    "location": meeting_data["location"],
                    "attendees_count": len(meeting_data["attendeeEmails"]),
                    "description": f"Meeting creation from: {description}"
                },
                "validation": {
                    "required_fields": ["title", "startTime", "endTime"],
                    "missing_fields": [field for field in ["title", "startTime", "endTime"] 
                                     if not meeting_data.get(field)]
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create meeting: {str(e)}"})

    @kernel_function(
        description="Update a meeting and return API call structure for frontend verification",
        name="update_meeting_api_call"
    )
    def update_meeting_api_call(self, meeting_id: str, update_description: str) -> str:
        """
        Update meeting data and return structured API call for frontend
        
        Args:
            meeting_id: ID of the meeting to update
            update_description: Natural language description of updates
            
        Returns:
            JSON string containing API endpoint, method, data, and preview
        """
        try:
            updates = {}
            
            # Extract new timing if mentioned
            schedule_info_json = self.parse_schedule_info(update_description)
            schedule_info = json.loads(schedule_info_json)
            
            if schedule_info.get("start_time"):
                updates["startTime"] = schedule_info["start_time"]
            if schedule_info.get("end_time"):
                updates["endTime"] = schedule_info["end_time"]
            
            # Extract other updates
            if "title" in update_description.lower():
                new_title = self._extract_meeting_title(update_description)
                if new_title != "Meeting":
                    updates["title"] = new_title
            
            if "location" in update_description.lower():
                new_location = self._extract_location_from_description(update_description)
                updates["location"] = new_location
            
            if "attendee" in update_description.lower() or "email" in update_description.lower():
                attendee_emails = self._extract_attendee_emails(update_description)
                if attendee_emails:
                    updates["attendeeEmails"] = attendee_emails
            
            # Check for status updates
            status_keywords = {
                "confirm": "confirmed",
                "cancel": "cancelled",
                "schedule": "scheduled"
            }
            
            for keyword, status in status_keywords.items():
                if keyword in update_description.lower():
                    updates["status"] = status
                    break
            
            # Structure API call response
            api_call = {
                "action": "update_meeting",
                "endpoint": f"/api/v1/calendar/meetings/{meeting_id}",
                "method": "PUT",
                "data": updates,
                "sync_google": False,
                "preview": {
                    "meeting_id": meeting_id,
                    "updates": updates,
                    "description": f"Meeting update from: {update_description}"
                },
                "validation": {
                    "has_updates": len(updates) > 0,
                    "update_fields": list(updates.keys())
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create update API call: {str(e)}"})

    @kernel_function(
        description="Delete a meeting and return API call structure for frontend verification",
        name="delete_meeting_api_call"
    )
    def delete_meeting_api_call(self, meeting_id: str, remove_from_google: bool = False) -> str:
        """
        Delete meeting and return structured API call for frontend
        
        Args:
            meeting_id: ID of the meeting to delete
            remove_from_google: Whether to also remove from Google Calendar
            
        Returns:
            JSON string containing API endpoint, method, and preview
        """
        try:
            api_call = {
                "action": "delete_meeting",
                "endpoint": f"/api/v1/calendar/meetings/{meeting_id}",
                "method": "DELETE",
                "params": {
                    "remove_google": remove_from_google
                },
                "preview": {
                    "meeting_id": meeting_id,
                    "will_remove_from_google": remove_from_google,
                    "description": f"Delete meeting {meeting_id}"
                },
                "validation": {
                    "meeting_id_valid": len(meeting_id) == 24,  # ObjectId length check
                    "confirmation_required": True
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create delete API call: {str(e)}"})

    @kernel_function(
        description="Sync jobs with Google Calendar and return API call structure",
        name="sync_google_calendar_api_call"
    )
    def sync_google_calendar_api_call(self, start_date: str, end_date: str) -> str:
        """
        Sync jobs with Google Calendar
        
        Args:
            start_date: Start date for sync range (ISO format)
            end_date: End date for sync range (ISO format)
            
        Returns:
            JSON string containing API endpoint, method, and preview
        """
        try:
            api_call = {
                "action": "sync_google_calendar",
                "endpoint": "/api/v1/calendar/sync-google-calendar",
                "method": "POST",
                "params": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "preview": {
                    "sync_range": f"{start_date} to {end_date}",
                    "description": "Sync all jobs in date range with Google Calendar"
                },
                "validation": {
                    "dates_valid": True,
                    "google_auth_required": True
                }
            }
            
            return json.dumps(api_call, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to create sync API call: {str(e)}"})

    @kernel_function(
        description="Search jobs by various criteria - returns actual data from database",
        name="search_jobs"
    )
    def search_jobs(self, search_query: str, limit: int = 50) -> str:
        """
        Search jobs by title, description, location, or client information
        
        Args:
            search_query: Search terms to look for
            limit: Maximum number of results to return
            
        Returns:
            JSON string containing matching jobs from database
        """
        try:
            return asyncio.run(self._search_jobs_async(search_query, limit))
        except Exception as e:
            return json.dumps({"error": f"Failed to search jobs: {str(e)}", "jobs": [], "total": 0})

    async def _search_jobs_async(self, search_query, limit):
        """Async implementation for searching jobs"""
        try:
            jobs_collection = get_jobs_collection()
            
            # Create text search query
            search_filter = {
                "$or": [
                    {"title": {"$regex": search_query, "$options": "i"}},
                    {"description": {"$regex": search_query, "$options": "i"}},
                    {"location": {"$regex": search_query, "$options": "i"}},
                    {"clientId": {"$regex": search_query, "$options": "i"}}
                ]
            }
            
            # Get matching jobs
            cursor = jobs_collection.find(search_filter).limit(limit).sort("startTime", -1)
            jobs_list = await cursor.to_list(length=limit)
            
            # Format results
            jobs_data = []
            for job in jobs_list:
                job_data = {
                    "id": str(job["_id"]),
                    "title": job.get("title", ""),
                    "clientId": job.get("clientId", ""),
                    "startTime": job.get("startTime").isoformat() if job.get("startTime") else None,
                    "endTime": job.get("endTime").isoformat() if job.get("endTime") else None,
                    "status": job.get("status", "scheduled"),
                    "location": job.get("location", ""),
                    "description": job.get("description", "")
                }
                jobs_data.append(job_data)
            
            result = {
                "jobs": jobs_data,
                "total": len(jobs_data),
                "search_query": search_query,
                "message": f"Found {len(jobs_data)} jobs matching '{search_query}'"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}", "jobs": [], "total": 0})

    @kernel_function(
        description="Get job statistics and summary from database - returns actual data",
        name="get_job_statistics"
    )
    def get_job_statistics(self, date_range_days: int = 30) -> str:
        """
        Get job statistics and summary
        
        Args:
            date_range_days: Number of days to include in statistics
            
        Returns:
            JSON string containing job statistics from database
        """
        try:
            return asyncio.run(self._get_job_statistics_async(date_range_days))
        except Exception as e:
            return json.dumps({"error": f"Failed to get statistics: {str(e)}"})

    async def _get_job_statistics_async(self, date_range_days):
        """Async implementation for getting job statistics"""
        try:
            jobs_collection = get_jobs_collection()
            
            # Date range for statistics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range_days)
            
            # Aggregation pipeline for statistics
            pipeline = [
                {
                    "$match": {
                        "startTime": {"$gte": start_date, "$lte": end_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "jobs": {"$push": {
                            "id": {"$toString": "$_id"},
                            "title": "$title",
                            "startTime": "$startTime"
                        }}
                    }
                }
            ]
            
            # Execute aggregation
            stats_cursor = jobs_collection.aggregate(pipeline)
            stats_list = await stats_cursor.to_list(length=None)
            
            # Calculate totals and format results
            total_jobs = sum(stat["count"] for stat in stats_list)
            status_breakdown = {stat["_id"]: stat["count"] for stat in stats_list}
            
            # Get upcoming jobs (next 7 days)
            upcoming_date = datetime.now() + timedelta(days=7)
            upcoming_cursor = jobs_collection.find({
                "startTime": {"$gte": datetime.now(), "$lte": upcoming_date},
                "status": {"$in": ["scheduled", "in_progress"]}
            }).sort("startTime", 1).limit(10)
            
            upcoming_jobs = await upcoming_cursor.to_list(length=10)
            upcoming_data = []
            for job in upcoming_jobs:
                upcoming_data.append({
                    "id": str(job["_id"]),
                    "title": job.get("title", ""),
                    "startTime": job.get("startTime").isoformat() if job.get("startTime") else None,
                    "status": job.get("status", "")
                })
            
            result = {
                "summary": {
                    "total_jobs": total_jobs,
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "status_breakdown": status_breakdown
                },
                "upcoming_jobs": {
                    "count": len(upcoming_data),
                    "jobs": upcoming_data
                },
                "statistics_date": datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Statistics query failed: {str(e)}"})

    # =============================================
    # CLIENT MANAGEMENT TOOLS
    # =============================================

    @kernel_function(
        description="Get all clients from database with optional filtering - returns actual data",
        name="get_clients"
    )
    async def get_clients(self,
                   skip: int = 0,
                   limit: int = 100,
                   search: Optional[str] = None,
                   status_filter: Optional[str] = None) -> str:
        """
        Retrieve clients from database with filtering
        
        Args:
            skip: Number of clients to skip for pagination
            limit: Maximum number of clients to return
            search: Search text to filter by name, email, or company
            status_filter: Filter by status (active, delinquent, archived)
            
        Returns:
            JSON string containing actual clients data from database
        """
        try:
            return await self._get_clients_async(skip, limit, search, status_filter)
        except Exception as e:
            return json.dumps({"error": f"Failed to get clients: {str(e)}", "clients": [], "total": 0})

    async def _get_clients_async(self, skip, limit, search, status_filter):
        """Async implementation for getting clients"""
        try:
            clients_collection = get_clients_collection()
            
            # Build query filter
            query = {}
            
            if search:
                query["$or"] = [
                    {"name": {"$regex": search, "$options": "i"}},
                    {"email": {"$regex": search, "$options": "i"}},
                    {"company": {"$regex": search, "$options": "i"}}
                ]
            
            if status_filter:
                query["status"] = status_filter
            
            # Get total count for pagination
            total = await clients_collection.count_documents(query)
            
            # Get clients with pagination
            cursor = clients_collection.find(query).skip(skip).limit(limit).sort("name", 1)
            clients_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            clients_data = []
            for client in clients_list:
                client_data = {
                    "id": str(client["_id"]),
                    "name": client.get("name", ""),
                    "email": client.get("email", ""),
                    "phone": client.get("phone", ""),
                    "address": client.get("address", ""),
                    "company": client.get("company", ""),
                    "balance": client.get("balance", 0.0),
                    "status": client.get("status", "active"),
                    "notes": client.get("notes", ""),
                    "created_at": client.get("created_at").isoformat() if client.get("created_at") else datetime.now().isoformat(),
                    "updated_at": client.get("updated_at").isoformat() if client.get("updated_at") else datetime.now().isoformat()
                }
                clients_data.append(client_data)
            
            result = {
                "clients": clients_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "clients": [], "total": 0})

    @kernel_function(
        description="Get a specific client by ID from database - returns actual data",
        name="get_client_by_id"
    )
    def get_client_by_id(self, client_id: str) -> str:
        """
        Retrieve a specific client by their ID from database
        
        Args:
            client_id: The client ID to retrieve
            
        Returns:
            JSON string containing actual client data from database
        """
        try:
            return asyncio.run(self._get_client_by_id_async(client_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to get client: {str(e)}", "client": None})

    async def _get_client_by_id_async(self, client_id):
        """Async implementation for getting client by ID"""
        try:
            clients_collection = get_clients_collection()
            
            # Convert string ID to ObjectId if valid
            try:
                object_id = ObjectId(client_id)
            except:
                return json.dumps({"error": "Invalid client ID format", "client": None})
            
            client = await clients_collection.find_one({"_id": object_id})
            
            if not client:
                return json.dumps({"error": "Client not found", "client": None})
            
            # Format client data
            client_data = {
                "id": str(client["_id"]),
                "name": client.get("name", ""),
                "email": client.get("email", ""),
                "phone": client.get("phone", ""),
                "address": client.get("address", ""),
                "company": client.get("company", ""),
                "balance": client.get("balance", 0.0),
                "status": client.get("status", "active"),
                "notes": client.get("notes", ""),
                "created_at": client.get("created_at").isoformat() if client.get("created_at") else datetime.now().isoformat(),
                "updated_at": client.get("updated_at").isoformat() if client.get("updated_at") else datetime.now().isoformat()
            }
            
            result = {"client": client_data}
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "client": None})

    # =============================================
    # EXPENSE MANAGEMENT TOOLS
    # =============================================

    @kernel_function(
        description="Get all expenses from database with optional filtering - returns actual data",
        name="get_expenses"
    )
    async def get_expenses(self,
                    skip: int = 0,
                    limit: int = 100,
                    search: Optional[str] = None,
                    category_filter: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> str:
        """
        Retrieve expenses from database with filtering
        
        Args:
            skip: Number of expenses to skip for pagination
            limit: Maximum number of expenses to return
            search: Search text to filter by description
            category_filter: Filter by category
            start_date: Filter expenses from this date (ISO format)
            end_date: Filter expenses until this date (ISO format)
            
        Returns:
            JSON string containing actual expenses data from database
        """
        try:
            return await self._get_expenses_async(skip, limit, search, category_filter, start_date, end_date)
        except Exception as e:
            return json.dumps({"error": f"Failed to get expenses: {str(e)}", "expenses": [], "total": 0})

    async def _get_expenses_async(self, skip, limit, search, category_filter, start_date, end_date):
        """Async implementation for getting expenses"""
        try:
            expenses_collection = get_expenses_collection()
            
            # Build query filter
            query = {}
            
            if search:
                query["description"] = {"$regex": search, "$options": "i"}
            
            if category_filter:
                query["category"] = category_filter
            
            # Date range filtering
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["date"] = date_filter
            
            # Get total count for pagination
            total = await expenses_collection.count_documents(query)
            
            # Get expenses with pagination
            cursor = expenses_collection.find(query).skip(skip).limit(limit).sort("date", -1)
            expenses_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            expenses_data = []
            for expense in expenses_list:
                expense_data = {
                    "id": str(expense["_id"]),
                    "description": expense.get("description", ""),
                    "amount": expense.get("amount", 0.0),
                    "vat_amount": expense.get("vat_amount", 0.0),
                    "vat_rate": expense.get("vat_rate", 20.0),
                    "category": expense.get("category", ""),
                    "date": expense.get("date").isoformat() if expense.get("date") else datetime.now().isoformat(),
                    "notes": expense.get("notes", ""),
                    "receipt_url": expense.get("receipt_url"),
                    "created_at": expense.get("created_at").isoformat() if expense.get("created_at") else datetime.now().isoformat(),
                    "updated_at": expense.get("updated_at").isoformat() if expense.get("updated_at") else datetime.now().isoformat()
                }
                expenses_data.append(expense_data)
            
            result = {
                "expenses": expenses_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "expenses": [], "total": 0})

    # =============================================
    # INVOICE MANAGEMENT TOOLS
    # =============================================

    @kernel_function(
        description="Get all invoices from database with optional filtering - returns actual data",
        name="get_invoices"
    )
    def get_invoices(self,
                    skip: int = 0,
                    limit: int = 100,
                    search: Optional[str] = None,
                    status_filter: Optional[str] = None,
                    client_id: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> str:
        """
        Retrieve invoices from database with filtering
        
        Args:
            skip: Number of invoices to skip for pagination
            limit: Maximum number of invoices to return
            search: Search text to filter by number or description
            status_filter: Filter by status (draft, sent, paid, overdue, cancelled)
            client_id: Filter by client ID
            start_date: Filter invoices from this date (ISO format)
            end_date: Filter invoices until this date (ISO format)
            
        Returns:
            JSON string containing actual invoices data from database
        """
        try:
            return asyncio.run(self._get_invoices_async(skip, limit, search, status_filter, client_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to get invoices: {str(e)}", "invoices": [], "total": 0})

    async def _get_invoices_async(self, skip, limit, search, status_filter, client_id, start_date, end_date):
        """Async implementation for getting invoices"""
        try:
            invoices_collection = get_invoices_collection()
            
            # Build query filter
            query = {}
            
            if search:
                query["$or"] = [
                    {"number": {"$regex": search, "$options": "i"}},
                    {"notes": {"$regex": search, "$options": "i"}}
                ]
            
            if status_filter:
                query["status"] = status_filter
            
            if client_id:
                query["clientId"] = client_id
            
            # Date range filtering on dueDate
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["dueDate"] = date_filter
            
            # Get total count for pagination
            total = await invoices_collection.count_documents(query)
            
            # Get invoices with pagination
            cursor = invoices_collection.find(query).skip(skip).limit(limit).sort("createdAt", -1)
            invoices_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            invoices_data = []
            for invoice in invoices_list:
                invoice_data = {
                    "id": str(invoice["_id"]),
                    "clientId": invoice.get("clientId", ""),
                    "number": invoice.get("number", ""),
                    "items": invoice.get("items", []),
                    "subtotal": invoice.get("subtotal", 0.0),
                    "discount": invoice.get("discount", 0.0),
                    "vatRate": invoice.get("vatRate", 20.0),
                    "vatAmount": invoice.get("vatAmount", 0.0),
                    "total": invoice.get("total", 0.0),
                    "status": invoice.get("status", "draft"),
                    "dueDate": invoice.get("dueDate").isoformat() if invoice.get("dueDate") else None,
                    "eInvoiceStatus": invoice.get("eInvoiceStatus"),
                    "notes": invoice.get("notes", ""),
                    "createdAt": invoice.get("createdAt").isoformat() if invoice.get("createdAt") else datetime.now().isoformat(),
                    "updatedAt": invoice.get("updatedAt").isoformat() if invoice.get("updatedAt") else datetime.now().isoformat()
                }
                invoices_data.append(invoice_data)
            
            result = {
                "invoices": invoices_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "invoices": [], "total": 0})

    # =============================================
    # QUOTE MANAGEMENT TOOLS  
    # =============================================

    @kernel_function(
        description="Get all quotes from database with optional filtering - returns actual data",
        name="get_quotes"
    )
    def get_quotes(self,
                  skip: int = 0,
                  limit: int = 100,
                  search: Optional[str] = None,
                  status_filter: Optional[str] = None,
                  client_id: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> str:
        """
        Retrieve quotes from database with filtering
        
        Args:
            skip: Number of quotes to skip for pagination
            limit: Maximum number of quotes to return
            search: Search text to filter by number or description
            status_filter: Filter by status (draft, sent, accepted, rejected, expired)
            client_id: Filter by client ID
            start_date: Filter quotes from this date (ISO format)
            end_date: Filter quotes until this date (ISO format)
            
        Returns:
            JSON string containing actual quotes data from database
        """
        try:
            return asyncio.run(self._get_quotes_async(skip, limit, search, status_filter, client_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to get quotes: {str(e)}", "quotes": [], "total": 0})

    async def _get_quotes_async(self, skip, limit, search, status_filter, client_id, start_date, end_date):
        """Async implementation for getting quotes"""
        try:
            quotes_collection = get_quotes_collection()
            
            # Build query filter
            query = {}
            
            if search:
                query["$or"] = [
                    {"number": {"$regex": search, "$options": "i"}},
                    {"notes": {"$regex": search, "$options": "i"}}
                ]
            
            if status_filter:
                query["status"] = status_filter
            
            if client_id:
                query["clientId"] = client_id
            
            # Date range filtering on validUntil
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query["validUntil"] = date_filter
            
            # Get total count for pagination
            total = await quotes_collection.count_documents(query)
            
            # Get quotes with pagination
            cursor = quotes_collection.find(query).skip(skip).limit(limit).sort("createdAt", -1)
            quotes_list = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format response
            quotes_data = []
            for quote in quotes_list:
                quote_data = {
                    "id": str(quote["_id"]),
                    "clientId": quote.get("clientId", ""),
                    "number": quote.get("number", ""),
                    "items": quote.get("items", []),
                    "subtotal": quote.get("subtotal", 0.0),
                    "discount": quote.get("discount", 0.0),
                    "vatRate": quote.get("vatRate", 20.0),
                    "vatAmount": quote.get("vatAmount", 0.0),
                    "total": quote.get("total", 0.0),
                    "status": quote.get("status", "draft"),
                    "validUntil": quote.get("validUntil").isoformat() if quote.get("validUntil") else None,
                    "notes": quote.get("notes", ""),
                    "createdAt": quote.get("createdAt").isoformat() if quote.get("createdAt") else datetime.now().isoformat(),
                    "updatedAt": quote.get("updatedAt").isoformat() if quote.get("updatedAt") else datetime.now().isoformat()
                }
                quotes_data.append(quote_data)
            
            result = {
                "quotes": quotes_data,
                "total": total
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Database query failed: {str(e)}", "quotes": [], "total": 0})

    # =============================================
    # HELPER METHODS FOR MEETING EXTRACTION
    # =============================================

    def _extract_meeting_title(self, description: str) -> str:
        """Extract meeting title from description"""
        # Look for explicit meeting titles
        title_patterns = [
            r'(?:meeting|call|conference)\\s*:?\\s*([^,\\.;]+)',
            r'(?:schedule|book|plan)\\s+(?:a\\s+)?(?:meeting|call)\\s+(?:about|for|with)?\\s*([^,\\.;]+)',
            r'^([^,\\.;]+?)\\s+(?:meeting|call|conference)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 3 and len(title) < 100:
                    return title.title()
        
        # Generate title from meeting type keywords
        meeting_types = {
            "kickoff": "Project Kickoff Meeting",
            "standup": "Daily Standup Meeting", 
            "review": "Review Meeting",
            "planning": "Planning Meeting",
            "retrospective": "Retrospective Meeting",
            "demo": "Demo Meeting",
            "training": "Training Session",
            "interview": "Interview",
            "consultation": "Consultation Call"
        }
        
        for keyword, title in meeting_types.items():
            if keyword in description.lower():
                return title
        
        return "Meeting"

    def _extract_meeting_description(self, description: str) -> str:
        """Extract meeting description/agenda from text"""
        # Look for agenda or description patterns
        desc_patterns = [
            r'(?:agenda|about|discuss|regarding)\\s*:?\\s*([^,\\.;]+)',
            r'(?:to discuss|will cover|topics?)\\s*:?\\s*([^,\\.;]+)'
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                if len(desc) > 3:
                    return desc
        
        return description[:200] + "..." if len(description) > 200 else description

    def _extract_attendee_emails(self, description: str) -> List[str]:
        """Extract attendee email addresses from description"""
        # Email pattern
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        emails = re.findall(email_pattern, description)
        
        return emails

    def _extract_job_title(self, description: str) -> str:
        """Extract job title from description"""
        # Look for explicit titles
        title_patterns = [
            r'(?:title|job|task)\\s*:?\\s*([^,\\.;]+)',
            r'(?:schedule|book|plan)\\s+([^,\\.;]+?)\\s+(?:for|with|at)',
            r'^([^,\\.;]+?)\\s+(?:for|with|at|on)',  # First part before preposition
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5 and len(title) < 100:
                    return title.title()
        
        # Generate title from job type keywords
        job_types = {
            "maintenance": "Maintenance Service",
            "installation": "Installation Service",
            "consultation": "Consultation Meeting",
            "meeting": "Client Meeting",
            "repair": "Repair Service",
            "setup": "Setup Service",
            "training": "Training Session",
            "review": "Review Meeting",
            "inspection": "Inspection Service",
            "deployment": "Deployment Service"
        }
        
        for keyword, title in job_types.items():
            if keyword in description.lower():
                return title
        
        return "Service Appointment"
    
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
            r'(?:for|with|client)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+(?:corp|company|inc|ltd)',
            r'(?:at|for)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)(?:\\s+office|\\s+location)?'
        ]
        
        for pattern in client_patterns:
            match = re.search(pattern, description)
            if match:
                name = match.group(1).strip()
                if "corp" in description.lower() or "company" in description.lower():
                    client_data["company"] = name
                    client_data["name"] = name
                else:
                    client_data["name"] = name
                break
        
        return client_data
    
    def _extract_timing_from_description(self, description: str) -> tuple:
        """Extract start and end times from description"""
        try:
            time_expressions = self._find_time_expressions(description)
            
            if time_expressions:
                return self._parse_primary_time_expression(time_expressions[0], description)
            
            return None, None
            
        except Exception:
            return None, None
    
    def _find_time_expressions(self, text: str) -> List[str]:
        """Find time-related expressions in text"""
        time_patterns = [
            # Relative day expressions
            r'\\b(?:next|this)\\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\b',
            r'\\b(?:tomorrow|today)\\b',
            r'\\b(?:in|after)\\s+\\d+\\s+(?:days?|weeks?|months?)\\b',
            
            # Specific date patterns
            r'\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b',
            r'\\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\\s+\\d{1,2}\\b',
            
            # Time patterns
            r'\\b\\d{1,2}\\s*:?\\s*\\d{0,2}\\s*(?:am|pm|AM|PM)\\b',
            r'\\b(?:at|around)\\s+\\d{1,2}\\s*(?:am|pm|AM|PM|:00|:30)?\\b',
            
            # Duration patterns
            r'\\b(?:for|during)\\s+\\d+\\s*(?:hours?|minutes?|hrs?)\\b',
            
            # Combined expressions
            r'\\b(?:next|this)\\s+\\w+\\s+at\\s+\\d{1,2}\\s*(?:am|pm|AM|PM)\\b'
        ]
        
        expressions = []
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expressions.append(match.group(0))
        
        return expressions
    
    def _parse_primary_time_expression(self, expression: str, full_text: str) -> tuple:
        """Parse a time expression to get start time, end time, and duration"""
        start_time = None
        end_time = None
        duration = None
        
        try:
            # Extract duration first
            duration_match = re.search(r'(?:for|during)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:hours?|hrs?)', full_text, re.IGNORECASE)
            if duration_match:
                duration = float(duration_match.group(1))
            
            # Parse relative day expressions
            now = datetime.now()
            
            if "tomorrow" in expression.lower():
                target_date = now + timedelta(days=1)
            elif "today" in expression.lower():
                target_date = now
            elif "next" in expression.lower():
                # Parse "next Tuesday", etc.
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                for i, day in enumerate(days):
                    if day in expression.lower():
                        days_ahead = (i - now.weekday() + 7) % 7
                        if days_ahead == 0:
                            days_ahead = 7  # Next week
                        target_date = now + timedelta(days=days_ahead)
                        break
                else:
                    target_date = now + timedelta(days=1)
            else:
                # Try to parse absolute date
                try:
                    target_date = parse_date(expression, fuzzy=True)
                except:
                    target_date = now + timedelta(days=1)
            
            # Extract time
            time_match = re.search(r'\\b(\\d{1,2})\\s*(?::(\\d{2}))?\\s*(am|pm|AM|PM)?\\b', expression)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                am_pm = time_match.group(3)
                
                if am_pm and am_pm.lower() == "pm" and hour != 12:
                    hour += 12
                elif am_pm and am_pm.lower() == "am" and hour == 12:
                    hour = 0
                
                start_time = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                # Default to 9 AM if no time specified
                start_time = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
            
            # Calculate end time
            if start_time and duration:
                end_time = start_time + timedelta(hours=duration)
            elif start_time and not duration:
                # Default duration of 2 hours
                end_time = start_time + timedelta(hours=2)
                duration = 2
            
        except Exception:
            # Fallback to tomorrow at 9 AM
            start_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            end_time = start_time + timedelta(hours=2)
            duration = 2
        
        return start_time, end_time, duration
    
    def _extract_location_from_description(self, description: str) -> str:
        """Extract location from description"""
        location_patterns = [
            r'(?:at|location|address)\\s*:?\\s*([^,\\.;]+(?:street|st|avenue|ave|office|building)[^,\\.;]*)',
            r'(?:visit|go to|travel to)\\s+([^,\\.;]+)',
            r'(?:on-?site|onsite)\\s+(?:at)?\\s*([^,\\.;]+)',
            r'(?:office|location|facility)\\s*:?\\s*([^,\\.;]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 3:
                    return location.title()
        
        # Check for remote work indicators
        remote_indicators = ["remote", "online", "video call", "zoom", "teams"]
        if any(indicator in description.lower() for indicator in remote_indicators):
            return "Remote"
        
        return "Client Location"
    
    def _extract_assigned_worker(self, description: str) -> Optional[str]:
        """Extract assigned worker from description"""
        worker_patterns = [
            r'(?:assign to|assigned to|worker|technician)\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)',
            r'(?:with|by)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)\\s+(?:handling|doing|performing)'
        ]
        
        for pattern in worker_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                worker = match.group(1).strip()
                if len(worker) > 2:
                    return worker.title()
        
        return None
    
    def _extract_job_notes(self, description: str) -> str:
        """Extract additional notes from description"""
        note_patterns = [
            r'(?:note|notes|special|important)\\s*:?\\s*([^,\\.;]+)',
            r'(?:requirement|requirements)\\s*:?\\s*([^,\\.;]+)',
            r'(?:prepare|bring|equipment needed)\\s*:?\\s*([^,\\.;]+)'
        ]
        
        notes = []
        for pattern in note_patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                note = match.group(1).strip()
                if note and len(note) > 3:
                    notes.append(note)
        
        return "; ".join(notes) if notes else ""
    
    def _validate_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean job data"""
        # Ensure required fields have defaults
        if not job_data.get("title"):
            job_data["title"] = "Service Appointment"
        
        if not job_data.get("location"):
            job_data["location"] = "Client Location"
        
        # Validate times
        start_time = job_data.get("start_time")
        end_time = job_data.get("end_time")
        
        if start_time and end_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            # Ensure end time is after start time
            if end_dt <= start_dt:
                job_data["end_time"] = (start_dt + timedelta(hours=2)).isoformat()
        
        return job_data
    
    def _check_schedule_conflicts(self, start_time: datetime, end_time: Optional[datetime], 
                                 assigned_to: str, existing_jobs: List[Dict]) -> List[Dict]:
        """Check for scheduling conflicts with existing jobs"""
        conflicts = []
        
        if not end_time:
            end_time = start_time + timedelta(hours=2)
        
        for job in existing_jobs:
            if job.get("assigned_to") != assigned_to:
                continue
            
            job_start = datetime.fromisoformat(job.get("start_time", "").replace('Z', '+00:00'))
            job_end_str = job.get("end_time")
            job_end = datetime.fromisoformat(job_end_str.replace('Z', '+00:00')) if job_end_str else job_start + timedelta(hours=2)
            
            # Check for overlap
            if start_time < job_end and end_time > job_start:
                conflict = {
                    "job_id": job.get("id"),
                    "job_title": job.get("title", "Unknown"),
                    "conflict_start": max(start_time, job_start).isoformat(),
                    "conflict_end": min(end_time, job_end).isoformat(),
                    "severity": "high" if (end_time - start_time).total_seconds() > 1800 else "low"  # >30 min overlap
                }
                conflicts.append(conflict)
        
        return conflicts
    
    def _generate_weekend_alternatives(self, start_date: datetime, duration_hours: float) -> List[Dict]:
        """Generate weekend alternative time slots"""
        alternatives = []
        
        # Find next Saturday and Sunday
        days_to_saturday = 5 - start_date.weekday()  # Saturday = 5
        if days_to_saturday <= 0:
            days_to_saturday += 7
        
        saturday = start_date + timedelta(days=days_to_saturday)
        sunday = saturday + timedelta(days=1)
        
        # Weekend hours (more flexible)
        weekend_hours = [9, 10, 11, 14, 15]
        
        for day in [saturday, sunday]:
            for hour in weekend_hours:
                suggested_start = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                suggested_end = suggested_start + timedelta(hours=duration_hours)
                
                alternative = {
                    "start_time": suggested_start.isoformat(),
                    "end_time": suggested_end.isoformat(),
                    "day_name": suggested_start.strftime("%A"),
                    "note": "Weekend appointment - may incur additional charges"
                }
                alternatives.append(alternative)
        
        return alternatives[:3]  # Return top 3 weekend alternatives