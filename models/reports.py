from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ReportType(str, Enum):
    FINANCIAL = "financial"
    SALES = "sales"
    CLIENTS = "clients"
    PROJECTS = "projects"
    EXPENSES = "expenses"
    VAT = "vat"


class ReportPeriod(str, Enum):
    THIS_MONTH = "thisMonth"
    LAST_MONTH = "lastMonth"
    THIS_QUARTER = "thisQuarter"
    THIS_YEAR = "thisYear"
    CUSTOM = "custom"


class FinancialMetrics(BaseModel):
    totalRevenue: float = Field(..., description="Total revenue from all invoices in euros")
    totalExpenses: float = Field(..., description="Total expenses in euros")
    netProfit: float = Field(..., description="Net profit (revenue - expenses) in euros")
    profitMargin: float = Field(..., description="Profit margin percentage")
    revenueChangePercent: float = Field(..., description="Revenue change percentage vs previous period")
    expensesChangePercent: float = Field(..., description="Expenses change percentage vs previous period")
    profitChangePercent: float = Field(..., description="Profit change percentage vs previous period")


class MonthlyDataPoint(BaseModel):
    month: str = Field(..., description="Month name (e.g., 'Jan', 'Fév')")
    revenue: float = Field(..., description="Revenue for the month in euros")
    expenses: float = Field(..., description="Expenses for the month in euros")
    profit: float = Field(..., description="Profit for the month in euros")


class InvoiceStatusSummary(BaseModel):
    status: str = Field(..., description="Invoice status (paid, pending, overdue)")
    count: int = Field(..., description="Number of invoices with this status")
    totalAmount: float = Field(..., description="Total amount for invoices with this status in euros")


class VatSummary(BaseModel):
    vatCollected: float = Field(..., description="Total VAT collected from invoices in euros")
    vatDeductible: float = Field(..., description="Total VAT deductible from expenses in euros")
    vatBalance: float = Field(..., description="VAT balance (collected - deductible) in euros")
    vatToPay: bool = Field(..., description="Whether there's VAT to pay (balance > 0)")


class TopClient(BaseModel):
    id: str = Field(..., description="Client ID")
    name: str = Field(..., description="Client name")
    email: str = Field(..., description="Client email")
    balance: float = Field(..., description="Client balance in euros")
    status: str = Field(..., description="Client status")


class FinancialReport(BaseModel):
    metrics: FinancialMetrics = Field(..., description="Key financial metrics")
    monthlyData: List[MonthlyDataPoint] = Field(..., description="Monthly revenue/expenses/profit data")
    invoiceStatusSummary: List[InvoiceStatusSummary] = Field(..., description="Invoice status breakdown")
    vatSummary: VatSummary = Field(..., description="VAT summary information")
    topClients: List[TopClient] = Field(..., description="Top 5 clients by balance")


class SalesMetrics(BaseModel):
    totalQuotes: int = Field(..., description="Total number of quotes")
    totalQuotesValue: float = Field(..., description="Total value of all quotes in euros")
    acceptedQuotes: int = Field(..., description="Number of accepted quotes")
    acceptedQuotesValue: float = Field(..., description="Total value of accepted quotes in euros")
    conversionRate: float = Field(..., description="Quote to invoice conversion rate percentage")
    averageQuoteValue: float = Field(..., description="Average quote value in euros")


class SalesReport(BaseModel):
    metrics: SalesMetrics = Field(..., description="Key sales metrics")
    quotesByStatus: List[Dict[str, Any]] = Field(..., description="Quotes grouped by status")
    monthlyQuotes: List[MonthlyDataPoint] = Field(..., description="Monthly quotes data")


class ClientMetrics(BaseModel):
    totalClients: int = Field(..., description="Total number of clients")
    activeClients: int = Field(..., description="Number of active clients")
    delinquentClients: int = Field(..., description="Number of delinquent clients")
    archivedClients: int = Field(..., description="Number of archived clients")
    totalOutstanding: float = Field(..., description="Total outstanding balance in euros")


class ClientStatusData(BaseModel):
    status: str = Field(..., description="Client status")
    count: int = Field(..., description="Number of clients with this status")
    percentage: float = Field(..., description="Percentage of total clients")


class ClientsReport(BaseModel):
    metrics: ClientMetrics = Field(..., description="Key client metrics")
    clientsByStatus: List[ClientStatusData] = Field(..., description="Clients grouped by status")
    topClients: List[TopClient] = Field(..., description="Top clients by balance")


class ExpenseMetrics(BaseModel):
    totalExpenses: float = Field(..., description="Total expenses in euros")
    totalVatDeductible: float = Field(..., description="Total VAT deductible in euros")
    expensesByCategory: List[Dict[str, Any]] = Field(..., description="Expenses grouped by category")
    monthlyExpenses: List[MonthlyDataPoint] = Field(..., description="Monthly expenses data")


class ExpensesReport(BaseModel):
    metrics: ExpenseMetrics = Field(..., description="Key expense metrics")
    expensesByCategory: List[Dict[str, Any]] = Field(..., description="Detailed expenses by category")


class VatReport(BaseModel):
    vatSummary: VatSummary = Field(..., description="VAT summary information")
    monthlyVat: List[Dict[str, Any]] = Field(..., description="Monthly VAT data")


class ReportRequest(BaseModel):
    reportType: ReportType = Field(..., description="Type of report to generate")
    period: ReportPeriod = Field(..., description="Time period for the report")
    startDate: Optional[datetime] = Field(None, description="Start date for custom period")
    endDate: Optional[datetime] = Field(None, description="End date for custom period")


class ReportResponse(BaseModel):
    reportType: ReportType = Field(..., description="Type of report generated")
    period: ReportPeriod = Field(..., description="Time period of the report")
    startDate: datetime = Field(..., description="Actual start date of the report period")
    endDate: datetime = Field(..., description="Actual end date of the report period")
    generatedAt: datetime = Field(default_factory=datetime.utcnow, description="Report generation timestamp")
    data: Dict[str, Any] = Field(..., description="Report data based on report type")
