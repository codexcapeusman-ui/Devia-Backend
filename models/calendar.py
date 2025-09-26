from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from bson import ObjectId
from datetime import datetime


class JobStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class MeetingAttendee(BaseModel):
    userId: str = Field(..., description="User ID of the attendee")
    name: str = Field(..., description="Attendee name")
    email: str = Field(..., description="Attendee email")
    status: str = Field(default="pending", description="RSVP status")


class Job(BaseModel):
    id: str = Field(alias="_id")
    title: str = Field(..., description="Job title")
    clientId: str = Field(..., description="Client ID this job belongs to")
    assignedTo: str = Field(..., description="User ID of the assigned worker")
    startTime: datetime = Field(..., description="Job start date and time")
    endTime: datetime = Field(..., description="Job end date and time")
    status: JobStatus = Field(default=JobStatus.SCHEDULED, description="Job status")
    location: Optional[str] = Field(None, description="Job location")
    description: Optional[str] = Field(None, description="Job description")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Job last update timestamp")

    class Config:
        validate_by_name = True
        json_encoders = {
            ObjectId: str
        }


class Meeting(BaseModel):
    id: str = Field(alias="_id")
    title: str = Field(..., description="Meeting title")
    description: Optional[str] = Field(None, description="Meeting description")
    startTime: datetime = Field(..., description="Meeting start date and time")
    endTime: datetime = Field(..., description="Meeting end date and time")
    location: Optional[str] = Field(None, description="Meeting location")
    attendees: List[MeetingAttendee] = Field(default_factory=list, description="Meeting attendees")
    organizerId: str = Field(..., description="User ID of the meeting organizer")
    status: MeetingStatus = Field(default=MeetingStatus.SCHEDULED, description="Meeting status")
    googleEventId: Optional[str] = Field(None, description="Google Calendar event ID")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Meeting creation timestamp")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Meeting last update timestamp")

    class Config:
        validate_by_name = True
        json_encoders = {
            ObjectId: str
        }