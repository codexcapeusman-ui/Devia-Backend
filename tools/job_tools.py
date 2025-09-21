"""
Job scheduling tools for Semantic Kernel
These tools handle job creation, scheduling, and management
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

from config.settings import Settings
from models import Job, Client, JobStatus

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
        description="Create a job from natural language text description",
        name="create_job_from_text"
    )
    def create_job_from_text(self, description: str, client_id: Optional[str] = None) -> str:
        """
        Create a complete job from text description
        
        Args:
            description: Natural language description of the job
            client_id: Optional client ID if known
            
        Returns:
            JSON string containing the created job data
            
        Example:
            Input: "Schedule website maintenance for ABC Corp next Tuesday at 2 PM, should take 3 hours"
            Output: JSON with complete job structure
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