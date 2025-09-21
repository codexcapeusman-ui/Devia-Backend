"""
Data models for the Devia backend system
Based on the frontend TypeScript interfaces
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Union
from datetime import datetime
from enum import Enum

# Enums
class UserRole(str, Enum):
    ADMIN = "admin"

class ClientStatus(str, Enum):
    ACTIVE = "active"
    DELINQUENT = "delinquent"
    ARCHIVED = "archived"

class QuoteStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"

class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

class JobStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ItemType(str, Enum):
    MATERIAL = "material"
    LABOR = "labor"
    SERVICE = "service"

class EInvoiceStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

# Core Models
class User(BaseModel):
    """User model"""
    id: str
    email: str
    name: str
    role: UserRole = UserRole.ADMIN
    company: Optional[str] = None
    avatar: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class Client(BaseModel):
    """Client model"""
    id: str
    name: str
    email: str
    phone: str
    address: str
    company: Optional[str] = None
    balance: float = 0.0
    status: ClientStatus = ClientStatus.ACTIVE
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

class QuoteItem(BaseModel):
    """Quote/Invoice item model"""
    id: str
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)
    type: ItemType = ItemType.SERVICE
    
    @validator('total', always=True)
    def calculate_total(cls, v, values):
        """Calculate total automatically"""
        if 'quantity' in values and 'unit_price' in values:
            return values['quantity'] * values['unit_price']
        return v

class Quote(BaseModel):
    """Quote model"""
    id: str
    client_id: str
    client: Optional[Client] = None
    number: str
    items: List[QuoteItem] = []
    subtotal: float = Field(ge=0)
    discount: float = Field(ge=0, default=0.0)
    vat_rate: float = Field(ge=0, le=100, default=20.0)
    vat_amount: float = Field(ge=0)
    total: float = Field(ge=0)
    status: QuoteStatus = QuoteStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    valid_until: datetime
    notes: Optional[str] = None
    
    @validator('subtotal', always=True)
    def calculate_subtotal(cls, v, values):
        """Calculate subtotal from items"""
        if 'items' in values:
            return sum(item.total for item in values['items'])
        return v
    
    @validator('vat_amount', always=True)
    def calculate_vat_amount(cls, v, values):
        """Calculate VAT amount"""
        if 'subtotal' in values and 'discount' in values and 'vat_rate' in values:
            discounted_subtotal = values['subtotal'] - values['discount']
            return discounted_subtotal * (values['vat_rate'] / 100)
        return v
    
    @validator('total', always=True)
    def calculate_total(cls, v, values):
        """Calculate total amount"""
        if 'subtotal' in values and 'discount' in values and 'vat_amount' in values:
            return values['subtotal'] - values['discount'] + values['vat_amount']
        return v

class Invoice(BaseModel):
    """Invoice model (extends Quote functionality)"""
    id: str
    client_id: str
    client: Optional[Client] = None
    number: str
    items: List[QuoteItem] = []
    subtotal: float = Field(ge=0)
    discount: float = Field(ge=0, default=0.0)
    vat_rate: float = Field(ge=0, le=100, default=20.0)
    vat_amount: float = Field(ge=0)
    total: float = Field(ge=0)
    status: InvoiceStatus = InvoiceStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    due_date: datetime
    paid_at: Optional[datetime] = None
    payment_method: Optional[str] = None
    e_invoice_status: Optional[EInvoiceStatus] = None
    notes: Optional[str] = None
    
    # Same validators as Quote
    @validator('subtotal', always=True)
    def calculate_subtotal(cls, v, values):
        if 'items' in values:
            return sum(item.total for item in values['items'])
        return v
    
    @validator('vat_amount', always=True)
    def calculate_vat_amount(cls, v, values):
        if 'subtotal' in values and 'discount' in values and 'vat_rate' in values:
            discounted_subtotal = values['subtotal'] - values['discount']
            return discounted_subtotal * (values['vat_rate'] / 100)
        return v
    
    @validator('total', always=True)
    def calculate_total(cls, v, values):
        if 'subtotal' in values and 'discount' in values and 'vat_amount' in values:
            return values['subtotal'] - values['discount'] + values['vat_amount']
        return v

class Job(BaseModel):
    """Job/Appointment model"""
    id: str
    title: str
    client_id: str
    client: Optional[Client] = None
    assigned_to: Optional[str] = None
    assigned_worker: Optional[User] = None
    start_time: datetime
    end_time: datetime
    status: JobStatus = JobStatus.SCHEDULED
    location: str
    description: str
    notes: Optional[str] = None
    images: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

class Expense(BaseModel):
    """Expense model"""
    id: str
    description: str
    amount: float = Field(gt=0)
    vat_amount: float = Field(ge=0)
    category: str
    date: datetime = Field(default_factory=datetime.now)
    receipt: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class DashboardStats(BaseModel):
    """Dashboard statistics model"""
    total_revenue: float = 0.0
    pending_invoices: int = 0
    active_clients: int = 0
    completed_jobs: int = 0
    overdue_invoices: int = 0
    monthly_revenue: float = 0.0
    quotes_this_month: int = 0
    jobs_this_week: int = 0

# Request/Response Models for AI Agent API
class AgentRequest(BaseModel):
    """Base request model for AI agent interactions"""
    prompt: str = Field(..., description="Natural language prompt for the AI agent")
    context: Optional[dict] = Field(default=None, description="Additional context data")
    language: str = Field(default="en", description="Response language (en/fr)")

class AgentResponse(BaseModel):
    """Base response model for AI agent interactions"""
    success: bool
    message: str
    data: Optional[dict] = None
    errors: Optional[List[str]] = None

# Specific Agent Request Models
class InvoiceGenerationRequest(AgentRequest):
    """Request model for invoice generation"""
    client_id: Optional[str] = None
    quote_id: Optional[str] = None  # If converting from quote

class CustomerExtractionRequest(AgentRequest):
    """Request model for customer data extraction"""
    pass

class QuoteGenerationRequest(AgentRequest):
    """Request model for quote generation"""
    client_id: Optional[str] = None

class JobSchedulingRequest(AgentRequest):
    """Request model for job scheduling"""
    client_id: Optional[str] = None

class ExpenseTrackingRequest(AgentRequest):
    """Request model for expense tracking"""
    receipt_text: Optional[str] = None

class AgentResponse(BaseModel):
    """Standard response model for AI agent interactions"""
    success: bool
    message: str
    data: Optional[dict] = None
    errors: Optional[List[str]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }