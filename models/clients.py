from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from enum import Enum
from bson import ObjectId
from datetime import datetime


class ClientStatus(str, Enum):
    ACTIVE = "active"
    DELINQUENT = "delinquent"
    ARCHIVED = "archived"


class Client(BaseModel):
    id: str = Field(alias="_id")
    name: str = Field(..., description="Client full name")
    email: EmailStr = Field(..., description="Client email address")
    phone: str = Field(..., description="Client phone number")
    address: str = Field(..., description="Client complete address")
    company: Optional[str] = Field(None, description="Client company name")
    balance: float = Field(default=0.0, description="Client account balance in euros")
    status: ClientStatus = Field(default=ClientStatus.ACTIVE, description="Client status")
    notes: Optional[str] = Field(None, description="Private notes about the client")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Client creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Client last update timestamp")

    class Config:
        validate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ClientCreate(BaseModel):
    name: str = Field(..., description="Client full name")
    email: EmailStr = Field(..., description="Client email address")
    phone: str = Field(..., description="Client phone number")
    address: str = Field(..., description="Client complete address")
    company: Optional[str] = Field(None, description="Client company name")
    notes: Optional[str] = Field(None, description="Private notes about the client")


class ClientUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Client full name")
    email: Optional[EmailStr] = Field(None, description="Client email address")
    phone: Optional[str] = Field(None, description="Client phone number")
    address: Optional[str] = Field(None, description="Client complete address")
    company: Optional[str] = Field(None, description="Client company name")
    balance: Optional[float] = Field(None, description="Client account balance in euros")
    status: Optional[ClientStatus] = Field(None, description="Client status")
    notes: Optional[str] = Field(None, description="Private notes about the client")


class ClientResponse(BaseModel):
    id: str = Field(alias="_id")
    name: str
    email: EmailStr
    phone: str
    address: str
    company: Optional[str] = None
    balance: float
    status: ClientStatus
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }