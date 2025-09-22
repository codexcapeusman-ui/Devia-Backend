from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from bson import ObjectId
from datetime import datetime


class QuoteStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class QuoteItemType(str, Enum):
    MATERIAL = "material"
    LABOR = "labor"
    JOB = "job"


class QuoteItem(BaseModel):
    id: str = Field(..., description="Item unique identifier")
    description: str = Field(..., description="Item description")
    quantity: float = Field(..., gt=0, description="Item quantity")
    unitPrice: float = Field(..., ge=0, description="Unit price before VAT in euros")
    total: float = Field(..., ge=0, description="Total price for this item before VAT in euros")
    type: QuoteItemType = Field(..., description="Item type")


class Quote(BaseModel):
    id: str = Field(alias="_id")
    clientId: str = Field(..., description="Client ID this quote belongs to")
    number: str = Field(..., description="Quote number")
    items: List[QuoteItem] = Field(..., description="List of quote items")
    subtotal: float = Field(..., ge=0, description="Subtotal before discount and VAT in euros")
    discount: float = Field(default=0.0, ge=0, description="Discount amount in euros")
    vatRate: float = Field(..., ge=0, le=100, description="VAT rate percentage")
    vatAmount: float = Field(..., ge=0, description="VAT amount in euros")
    total: float = Field(..., ge=0, description="Total amount including VAT in euros")
    status: QuoteStatus = Field(default=QuoteStatus.DRAFT, description="Quote status")
    validUntil: datetime = Field(..., description="Quote validity date")
    notes: Optional[str] = Field(None, description="Optional notes about the quote")
    createdAt: datetime = Field(default_factory=datetime.utcnow, description="Quote creation timestamp")
    updatedAt: datetime = Field(default_factory=datetime.utcnow, description="Quote last update timestamp")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }


class QuoteCreate(BaseModel):
    clientId: str = Field(..., description="Client ID this quote belongs to")
    number: str = Field(..., description="Quote number")
    items: List[QuoteItem] = Field(..., description="List of quote items")
    discount: float = Field(default=0.0, ge=0, description="Discount amount in euros")
    vatRate: float = Field(default=20.0, ge=0, le=100, description="VAT rate percentage")
    validUntil: datetime = Field(..., description="Quote validity date")
    notes: Optional[str] = Field(None, description="Optional notes about the quote")


class QuoteUpdate(BaseModel):
    number: Optional[str] = Field(None, description="Quote number")
    items: Optional[List[QuoteItem]] = Field(None, description="List of quote items")
    discount: Optional[float] = Field(None, ge=0, description="Discount amount in euros")
    vatRate: Optional[float] = Field(None, ge=0, le=100, description="VAT rate percentage")
    validUntil: Optional[datetime] = Field(None, description="Quote validity date")
    status: Optional[QuoteStatus] = Field(None, description="Quote status")
    notes: Optional[str] = Field(None, description="Optional notes about the quote")


class QuoteResponse(BaseModel):
    id: str
    clientId: str
    number: str
    items: List[QuoteItem]
    subtotal: float
    discount: float
    vatRate: float
    vatAmount: float
    total: float
    status: QuoteStatus
    validUntil: datetime
    notes: Optional[str]
    createdAt: datetime
    updatedAt: datetime

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }