from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from bson import ObjectId
from datetime import datetime


class ExpenseCategory(str, Enum):
    MATERIALS = "Materials"
    TRANSPORT = "Transport"
    EQUIPMENT = "Equipment"
    LABOR = "Labor"
    INSURANCE = "Insurance"
    GENERAL = "General"
    TRAINING = "Training"
    MARKETING = "Marketing"
    OTHERS = "Others"


class Expense(BaseModel):
    id: str = Field(alias="_id")
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., gt=0, description="Expense amount before VAT in euros")
    vat_amount: float = Field(..., ge=0, description="VAT amount in euros")
    vat_rate: float = Field(..., ge=0, le=100, description="VAT rate percentage")
    category: ExpenseCategory = Field(..., description="Expense category")
    date: datetime = Field(..., description="Expense date")
    notes: Optional[str] = Field(None, description="Optional notes about the expense")
    receipt_url: Optional[str] = Field(None, description="URL to receipt file if uploaded")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Expense creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Expense last update timestamp")

    class Config:
        validate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ExpenseCreate(BaseModel):
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., gt=0, description="Expense amount before VAT in euros")
    vat_rate: float = Field(default=20.0, ge=0, le=100, description="VAT rate percentage")
    category: ExpenseCategory = Field(..., description="Expense category")
    date: datetime = Field(..., description="Expense date")
    notes: Optional[str] = Field(None, description="Optional notes about the expense")
    receipt_url: Optional[str] = Field(None, description="URL to receipt file if uploaded")


class ExpenseUpdate(BaseModel):
    description: Optional[str] = Field(None, description="Expense description")
    amount: Optional[float] = Field(None, gt=0, description="Expense amount before VAT in euros")
    vat_rate: Optional[float] = Field(None, ge=0, le=100, description="VAT rate percentage")
    category: Optional[ExpenseCategory] = Field(None, description="Expense category")
    date: Optional[datetime] = Field(None, description="Expense date")
    notes: Optional[str] = Field(None, description="Optional notes about the expense")
    receipt_url: Optional[str] = Field(None, description="URL to receipt file if uploaded")


class ExpenseResponse(BaseModel):
    id: str
    description: str
    amount: float
    vat_amount: float
    vat_rate: float
    category: ExpenseCategory
    date: datetime
    notes: Optional[str] = None
    receipt_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }