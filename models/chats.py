"""
Chat and Conversation models for AI Agent interactions
Stores conversation history for record-keeping and context
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from bson import ObjectId
from datetime import datetime


class MessageRole(str, Enum):
    """Role of the message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationState(str, Enum):
    """States of conversation flow"""
    INTENT_DETECTION = "intent_detection"
    DATA_EXTRACTION = "data_extraction"
    DATA_COMPLETION = "data_completion"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"


class ChatMessage(BaseModel):
    """Individual chat message within a conversation"""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    """
    Conversation model for storing AI agent interactions
    Tracks the full conversation flow including intent, state, and extracted data
    """
    id: str = Field(alias="_id")
    user_id: str = Field(..., description="User ID who owns this conversation")
    
    # Conversation state
    state: ConversationState = Field(
        default=ConversationState.INTENT_DETECTION, 
        description="Current conversation state"
    )
    intent: Optional[str] = Field(default=None, description="Detected intent (invoice, quote, customer, etc.)")
    operation: Optional[str] = Field(default=None, description="Detected operation (get, create, update, delete)")
    confidence: float = Field(default=0.0, description="Intent detection confidence score")
    
    # Messages history
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")
    
    # Extracted data
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Data extracted from conversation")
    
    # Per-field attempt tracking (max 2 attempts per field)
    field_attempts: Dict[str, int] = Field(
        default_factory=dict, 
        description="Number of attempts made to collect each missing field"
    )
    
    # Language preference
    language: str = Field(default="en", description="Conversation language (en/fr)")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Status
    is_active: bool = Field(default=True, description="Whether conversation is still active")

    class Config:
        validate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ConversationCreate(BaseModel):
    """Model for creating a new conversation"""
    user_id: str = Field(..., description="User ID who owns this conversation")
    language: str = Field(default="en", description="Conversation language")
    
    # Optional initial state
    intent: Optional[str] = Field(default=None)
    operation: Optional[str] = Field(default=None)
    context: Optional[Dict[str, Any]] = Field(default=None, description="Initial context from request")


class ConversationUpdate(BaseModel):
    """Model for updating an existing conversation"""
    state: Optional[ConversationState] = None
    intent: Optional[str] = None
    operation: Optional[str] = None
    confidence: Optional[float] = None
    extracted_data: Optional[Dict[str, Any]] = None
    field_attempts: Optional[Dict[str, int]] = None
    is_active: Optional[bool] = None


class ConversationResponse(BaseModel):
    """Response model for conversation data"""
    id: str = Field(alias="_id")
    user_id: str
    state: ConversationState
    intent: Optional[str] = None
    operation: Optional[str] = None
    confidence: float
    messages: List[ChatMessage]
    extracted_data: Dict[str, Any]
    field_attempts: Dict[str, int]
    language: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        validate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationListResponse(BaseModel):
    """Response model for listing conversations"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int
    has_more: bool
