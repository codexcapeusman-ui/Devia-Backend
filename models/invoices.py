from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from bson import ObjectId
from datetime import datetime


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class EInvoiceStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ItemType(str, Enum):
    MATERIAL = "material"
    LABOR = "labor"
    SERVICE = "service"


class InvoiceItem(BaseModel):
    id: str = Field(..., description="Item unique identifier")
    description: str = Field(..., description="Item description")
    quantity: float = Field(..., gt=0, description="Item quantity")
    unitPrice: float = Field(..., ge=0, description="Unit price before VAT in euros")
    total: float = Field(..., ge=0, description="Total price for this item before VAT in euros")
    type: ItemType = Field(..., description="Item type")


class Invoice(BaseModel):
    id: str = Field(alias="_id")
    clientId: str = Field(..., description="Client ID this invoice belongs to")
    number: str = Field(..., description="Invoice number")
    items: List[InvoiceItem] = Field(..., description="List of invoice items")
    subtotal: float = Field(..., ge=0, description="Subtotal before discount and VAT in euros")
    discount: float = Field(default=0.0, ge=0, description="Discount amount in euros")
    vatRate: float = Field(..., ge=0, le=100, description="VAT rate percentage")
    vatAmount: float = Field(..., ge=0, description="VAT amount in euros")
    total: float = Field(..., ge=0, description="Total amount including VAT in euros")
    status: InvoiceStatus = Field(default=InvoiceStatus.DRAFT, description="Invoice status")
    dueDate: datetime = Field(..., description="Invoice due date")
    eInvoiceStatus: Optional[EInvoiceStatus] = Field(None, description="E-invoice submission status")
    notes: Optional[str] = Field(None, description="Optional notes about the invoice")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Invoice creation timestamp")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Invoice last update timestamp")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }


class InvoiceCreate(BaseModel):
    clientId: str = Field(..., description="Client ID this invoice belongs to")
    number: str = Field(..., description="Invoice number")
    items: List[InvoiceItem] = Field(..., description="List of invoice items")
    discount: float = Field(default=0.0, ge=0, description="Discount amount in euros")
    vatRate: float = Field(default=20.0, ge=0, le=100, description="VAT rate percentage")
    dueDate: datetime = Field(..., description="Invoice due date")
    notes: Optional[str] = Field(None, description="Optional notes about the invoice")


class InvoiceUpdate(BaseModel):
    number: Optional[str] = Field(None, description="Invoice number")
    items: Optional[List[InvoiceItem]] = Field(None, description="List of invoice items")
    discount: Optional[float] = Field(None, ge=0, description="Discount amount in euros")
    vatRate: Optional[float] = Field(None, ge=0, le=100, description="VAT rate percentage")
    dueDate: Optional[datetime] = Field(None, description="Invoice due date")
    status: Optional[InvoiceStatus] = Field(None, description="Invoice status")
    eInvoiceStatus: Optional[EInvoiceStatus] = Field(None, description="E-invoice submission status")
    notes: Optional[str] = Field(None, description="Optional notes about the invoice")


class InvoiceResponse(BaseModel):
    id: str
    clientId: str
    number: str
    items: List[InvoiceItem]
    subtotal: float
    discount: float
    vatRate: float
    vatAmount: float
    total: float
    status: InvoiceStatus
    dueDate: datetime
    eInvoiceStatus: Optional[EInvoiceStatus]
    notes: Optional[str]
    createdAt: datetime
    updatedAt: datetime

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }