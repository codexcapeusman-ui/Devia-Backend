"""
Report generation tools for Devia backend
Provides AI-powered report generation and data aggregation
"""

import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId

from semantic_kernel.functions.kernel_function_decorator import kernel_function
from motor.motor_asyncio import AsyncIOMotorCollection

from ..database import get_invoices_collection, get_expenses_collection, get_clients_collection, get_quotes_collection
from ..models.reports import (
    ReportType, ReportPeriod, FinancialReport, FinancialMetrics,
    SalesReport, SalesMetrics, ClientsReport, ClientMetrics,
    ExpensesReport, ExpenseMetrics, VatReport, VatSummary,
    MonthlyDataPoint, InvoiceStatusSummary, TopClient, ClientStatusData
)


class ReportTools:
    """Tools for generating various business reports"""

    @kernel_function(
        description="Generate a financial report with revenue, expenses, and profit metrics - returns actual data",
        name="get_financial_report"
    )
    def get_financial_report(self,
                           period: str = "thisMonth",
                           user_id: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> str:
        """
        Generate a comprehensive financial report

        Args:
            period: Report period (thisMonth, lastMonth, thisQuarter, thisYear, custom)
            user_id: Filter by user ID (required for security)
            start_date: Start date for custom period (ISO format)
            end_date: End date for custom period (ISO format)

        Returns:
            JSON string containing financial report data
        """
        try:
            return asyncio.run(self._get_financial_report_async(period, user_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to generate financial report: {str(e)}"})

    async def _get_financial_report_async(self, period, user_id, start_date, end_date):
        """Async implementation for financial report generation"""
        try:
            # Determine date range
            start_dt, end_dt = self._get_date_range(period, start_date, end_date)
            prev_start_dt, prev_end_dt = self._get_previous_period(start_dt, end_dt)

            # Build user filter
            user_filter = {"userId": user_id} if user_id else {}

            # Get current period data
            current_revenue = await self._calculate_total_revenue(start_dt, end_dt, user_filter)
            current_expenses = await self._calculate_total_expenses(start_dt, end_dt, user_filter)

            # Get previous period data for comparison
            prev_revenue = await self._calculate_total_revenue(prev_start_dt, prev_end_dt, user_filter)
            prev_expenses = await self._calculate_total_expenses(prev_start_dt, prev_end_dt, user_filter)

            # Calculate metrics
            net_profit = current_revenue - current_expenses
            profit_margin = (net_profit / current_revenue * 100) if current_revenue > 0 else 0

            revenue_change = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            expenses_change = ((current_expenses - prev_expenses) / prev_expenses * 100) if prev_expenses > 0 else 0
            profit_change = ((net_profit - (prev_revenue - prev_expenses)) / (prev_revenue - prev_expenses) * 100) if (prev_revenue - prev_expenses) != 0 else 0

            # Get monthly data
            monthly_data = await self._get_monthly_financial_data(start_dt, end_dt, user_filter)

            # Get invoice status summary
            invoice_status_summary = await self._get_invoice_status_summary(start_dt, end_dt, user_filter)

            # Get VAT summary
            vat_summary = await self._get_vat_summary(start_dt, end_dt, user_filter)

            # Get top clients
            top_clients = await self._get_top_clients(user_filter)

            # Build financial metrics
            metrics = FinancialMetrics(
                totalRevenue=current_revenue,
                totalExpenses=current_expenses,
                netProfit=net_profit,
                profitMargin=profit_margin,
                revenueChangePercent=revenue_change,
                expensesChangePercent=expenses_change,
                profitChangePercent=profit_change
            )

            # Create report
            report = FinancialReport(
                metrics=metrics,
                monthlyData=monthly_data,
                invoiceStatusSummary=invoice_status_summary,
                vatSummary=vat_summary,
                topClients=top_clients
            )

            return json.dumps({
                "reportType": "financial",
                "period": period,
                "startDate": start_dt.isoformat(),
                "endDate": end_dt.isoformat(),
                "generatedAt": datetime.now().isoformat(),
                "data": report.dict()
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to generate financial report: {str(e)}"})

    async def _calculate_total_revenue(self, start_date, end_date, user_filter):
        """Calculate total revenue from paid invoices"""
        invoices_collection = get_invoices_collection()

        pipeline = [
            {
                "$match": {
                    **user_filter,
                    "status": "paid",
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": "$total_amount"}
                }
            }
        ]

        result = await invoices_collection.aggregate(pipeline).to_list(length=1)
        return result[0]["total"] if result else 0

    async def _calculate_total_expenses(self, start_date, end_date, user_filter):
        """Calculate total expenses"""
        expenses_collection = get_expenses_collection()

        pipeline = [
            {
                "$match": {
                    **user_filter,
                    "date": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": "$amount"}
                }
            }
        ]

        result = await expenses_collection.aggregate(pipeline).to_list(length=1)
        return result[0]["total"] if result else 0

    async def _get_monthly_financial_data(self, start_date, end_date, user_filter):
        """Get monthly revenue and expenses data"""
        # This is a simplified implementation - in a real system you'd aggregate by month
        # For now, return a single data point for the period
        revenue = await self._calculate_total_revenue(start_date, end_date, user_filter)
        expenses = await self._calculate_total_expenses(start_date, end_date, user_filter)

        return [
            MonthlyDataPoint(
                month=start_date.strftime("%b %Y"),
                revenue=revenue,
                expenses=expenses,
                profit=revenue - expenses
            )
        ]

    async def _get_invoice_status_summary(self, start_date, end_date, user_filter):
        """Get invoice status summary"""
        invoices_collection = get_invoices_collection()

        pipeline = [
            {
                "$match": {
                    **user_filter,
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "totalAmount": {"$sum": "$total_amount"}
                }
            }
        ]

        results = await invoices_collection.aggregate(pipeline).to_list(length=None)

        return [
            InvoiceStatusSummary(
                status=result["_id"],
                count=result["count"],
                totalAmount=result["totalAmount"]
            )
            for result in results
        ]

    async def _get_vat_summary(self, start_date, end_date, user_filter):
        """Get VAT summary"""
        # Calculate VAT collected from invoices
        invoices_collection = get_invoices_collection()
        vat_collected_pipeline = [
            {
                "$match": {
                    **user_filter,
                    "status": "paid",
                    "createdAt": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$unwind": "$items"
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": {"$multiply": ["$items.quantity", "$items.unit_price", {"$divide": ["$items.vat_rate", 100]}]}}
                }
            }
        ]

        vat_collected_result = await invoices_collection.aggregate(vat_collected_pipeline).to_list(length=1)
        vat_collected = vat_collected_result[0]["total"] if vat_collected_result else 0

        # Calculate VAT deductible from expenses
        expenses_collection = get_expenses_collection()
        vat_deductible_pipeline = [
            {
                "$match": {
                    **user_filter,
                    "date": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": "$vat_amount"}
                }
            }
        ]

        vat_deductible_result = await expenses_collection.aggregate(vat_deductible_pipeline).to_list(length=1)
        vat_deductible = vat_deductible_result[0]["total"] if vat_deductible_result else 0

        vat_balance = vat_collected - vat_deductible

        return VatSummary(
            vatCollected=vat_collected,
            vatDeductible=vat_deductible,
            vatBalance=vat_balance,
            vatToPay=vat_balance > 0
        )

    async def _get_top_clients(self, user_filter, limit=5):
        """Get top clients by balance"""
        clients_collection = get_clients_collection()

        pipeline = [
            {
                "$match": user_filter
            },
            {
                "$sort": {"balance": -1}
            },
            {
                "$limit": limit
            }
        ]

        clients = await clients_collection.aggregate(pipeline).to_list(length=limit)

        return [
            TopClient(
                id=str(client["_id"]),
                name=client.get("name", ""),
                email=client.get("email", ""),
                balance=client.get("balance", 0),
                status=client.get("status", "active")
            )
            for client in clients
        ]

    def _get_date_range(self, period, start_date_str, end_date_str):
        """Get date range based on period"""
        now = datetime.now()

        if period == "thisMonth":
            start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_dt = (start_dt + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        elif period == "lastMonth":
            end_dt = now.replace(day=1) - timedelta(days=1)
            start_dt = end_dt.replace(day=1)
        elif period == "thisQuarter":
            quarter = (now.month - 1) // 3 + 1
            start_dt = datetime(now.year, (quarter - 1) * 3 + 1, 1)
            end_dt = datetime(now.year, quarter * 3 + 1, 1) - timedelta(days=1)
        elif period == "thisYear":
            start_dt = datetime(now.year, 1, 1)
            end_dt = datetime(now.year, 12, 31)
        elif period == "custom" and start_date_str and end_date_str:
            start_dt = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            # Default to this month
            start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_dt = (start_dt + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        return start_dt, end_dt

    def _get_previous_period(self, start_date, end_date):
        """Get the previous period for comparison"""
        period_length = end_date - start_date
        prev_end_dt = start_date - timedelta(days=1)
        prev_start_dt = prev_end_dt - period_length
        return prev_start_dt, prev_end_dt

    @kernel_function(
        description="Generate a sales report with quotes and conversion metrics - returns actual data",
        name="get_sales_report"
    )
    def get_sales_report(self,
                        period: str = "thisMonth",
                        user_id: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> str:
        """
        Generate a sales report with quote metrics

        Args:
            period: Report period
            user_id: Filter by user ID (required for security)
            start_date: Start date for custom period
            end_date: End date for custom period

        Returns:
            JSON string containing sales report data
        """
        try:
            return asyncio.run(self._get_sales_report_async(period, user_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to generate sales report: {str(e)}"})

    async def _get_sales_report_async(self, period, user_id, start_date, end_date):
        """Async implementation for sales report generation"""
        try:
            start_dt, end_dt = self._get_date_range(period, start_date, end_date)
            user_filter = {"userId": user_id} if user_id else {}

            quotes_collection = get_quotes_collection()

            # Get all quotes in period
            quotes_pipeline = [
                {
                    "$match": {
                        **user_filter,
                        "createdAt": {"$gte": start_dt, "$lte": end_dt}
                    }
                }
            ]

            quotes = await quotes_collection.aggregate(quotes_pipeline).to_list(length=None)

            total_quotes = len(quotes)
            total_quotes_value = sum(quote.get("total_amount", 0) for quote in quotes)

            # Get accepted quotes
            accepted_quotes = [q for q in quotes if q.get("status") == "accepted"]
            accepted_quotes_count = len(accepted_quotes)
            accepted_quotes_value = sum(q.get("total_amount", 0) for q in accepted_quotes)

            conversion_rate = (accepted_quotes_count / total_quotes * 100) if total_quotes > 0 else 0
            average_quote_value = total_quotes_value / total_quotes if total_quotes > 0 else 0

            # Build metrics
            metrics = SalesMetrics(
                totalQuotes=total_quotes,
                totalQuotesValue=total_quotes_value,
                acceptedQuotes=accepted_quotes_count,
                acceptedQuotesValue=accepted_quotes_value,
                conversionRate=conversion_rate,
                averageQuoteValue=average_quote_value
            )

            # Get quotes by status
            status_pipeline = [
                {
                    "$match": {
                        **user_filter,
                        "createdAt": {"$gte": start_dt, "$lte": end_dt}
                    }
                },
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "totalValue": {"$sum": "$total_amount"}
                    }
                }
            ]

            status_results = await quotes_collection.aggregate(status_pipeline).to_list(length=None)
            quotes_by_status = [
                {"status": r["_id"], "count": r["count"], "totalValue": r["totalValue"]}
                for r in status_results
            ]

            # Monthly quotes data (simplified)
            monthly_quotes = [
                MonthlyDataPoint(
                    month=start_dt.strftime("%b %Y"),
                    revenue=total_quotes_value,
                    expenses=0,  # Not applicable for quotes
                    profit=accepted_quotes_value
                )
            ]

            report = SalesReport(
                metrics=metrics,
                quotesByStatus=quotes_by_status,
                monthlyQuotes=monthly_quotes
            )

            return json.dumps({
                "reportType": "sales",
                "period": period,
                "startDate": start_dt.isoformat(),
                "endDate": end_dt.isoformat(),
                "generatedAt": datetime.now().isoformat(),
                "data": report.dict()
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to generate sales report: {str(e)}"})

    @kernel_function(
        description="Generate a clients report with client metrics and status breakdown - returns actual data",
        name="get_clients_report"
    )
    def get_clients_report(self, user_id: Optional[str] = None) -> str:
        """
        Generate a clients report

        Args:
            user_id: Filter by user ID (required for security)

        Returns:
            JSON string containing clients report data
        """
        try:
            return asyncio.run(self._get_clients_report_async(user_id))
        except Exception as e:
            return json.dumps({"error": f"Failed to generate clients report: {str(e)}"})

    async def _get_clients_report_async(self, user_id):
        """Async implementation for clients report generation"""
        try:
            user_filter = {"userId": user_id} if user_id else {}
            clients_collection = get_clients_collection()

            # Get all clients
            clients = await clients_collection.find(user_filter).to_list(length=None)

            total_clients = len(clients)
            active_clients = len([c for c in clients if c.get("status") == "active"])
            delinquent_clients = len([c for c in clients if c.get("status") == "delinquent"])
            archived_clients = len([c for c in clients if c.get("status") == "archived"])

            total_outstanding = sum(c.get("balance", 0) for c in clients)

            # Build metrics
            metrics = ClientMetrics(
                totalClients=total_clients,
                activeClients=active_clients,
                delinquentClients=delinquent_clients,
                archivedClients=archived_clients,
                totalOutstanding=total_outstanding
            )

            # Get clients by status
            status_counts = {}
            for client in clients:
                status = client.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            clients_by_status = [
                ClientStatusData(
                    status=status,
                    count=count,
                    percentage=(count / total_clients * 100) if total_clients > 0 else 0
                )
                for status, count in status_counts.items()
            ]

            # Get top clients
            top_clients = await self._get_top_clients(user_filter, limit=10)

            report = ClientsReport(
                metrics=metrics,
                clientsByStatus=clients_by_status,
                topClients=top_clients
            )

            return json.dumps({
                "reportType": "clients",
                "generatedAt": datetime.now().isoformat(),
                "data": report.dict()
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to generate clients report: {str(e)}"})

    @kernel_function(
        description="Generate an expenses report with expense metrics and category breakdown - returns actual data",
        name="get_expenses_report"
    )
    def get_expenses_report(self,
                           period: str = "thisMonth",
                           user_id: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> str:
        """
        Generate an expenses report

        Args:
            period: Report period
            user_id: Filter by user ID (required for security)
            start_date: Start date for custom period
            end_date: End date for custom period

        Returns:
            JSON string containing expenses report data
        """
        try:
            return asyncio.run(self._get_expenses_report_async(period, user_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to generate expenses report: {str(e)}"})

    async def _get_expenses_report_async(self, period, user_id, start_date, end_date):
        """Async implementation for expenses report generation"""
        try:
            start_dt, end_dt = self._get_date_range(period, start_date, end_date)
            user_filter = {"userId": user_id} if user_id else {}

            expenses_collection = get_expenses_collection()

            # Get expenses in period
            expenses = await expenses_collection.find({
                **user_filter,
                "date": {"$gte": start_dt, "$lte": end_dt}
            }).to_list(length=None)

            total_expenses = sum(expense.get("amount", 0) for expense in expenses)
            total_vat_deductible = sum(expense.get("vat_amount", 0) for expense in expenses)

            # Group by category
            category_totals = {}
            for expense in expenses:
                category = expense.get("category", "uncategorized")
                amount = expense.get("amount", 0)
                category_totals[category] = category_totals.get(category, 0) + amount

            expenses_by_category = [
                {"category": category, "total": total}
                for category, total in category_totals.items()
            ]

            # Monthly expenses data (simplified)
            monthly_expenses = [
                MonthlyDataPoint(
                    month=start_dt.strftime("%b %Y"),
                    revenue=0,  # Not applicable
                    expenses=total_expenses,
                    profit=-total_expenses
                )
            ]

            # Build metrics
            metrics = ExpenseMetrics(
                totalExpenses=total_expenses,
                totalVatDeductible=total_vat_deductible,
                expensesByCategory=expenses_by_category,
                monthlyExpenses=monthly_expenses
            )

            report = ExpensesReport(
                metrics=metrics,
                expensesByCategory=expenses_by_category
            )

            return json.dumps({
                "reportType": "expenses",
                "period": period,
                "startDate": start_dt.isoformat(),
                "endDate": end_dt.isoformat(),
                "generatedAt": datetime.now().isoformat(),
                "data": report.dict()
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to generate expenses report: {str(e)}"})

    @kernel_function(
        description="Generate a VAT report with VAT collected, deductible, and balance - returns actual data",
        name="get_vat_report"
    )
    def get_vat_report(self,
                      period: str = "thisMonth",
                      user_id: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> str:
        """
        Generate a VAT report

        Args:
            period: Report period
            user_id: Filter by user ID (required for security)
            start_date: Start date for custom period
            end_date: End date for custom period

        Returns:
            JSON string containing VAT report data
        """
        try:
            return asyncio.run(self._get_vat_report_async(period, user_id, start_date, end_date))
        except Exception as e:
            return json.dumps({"error": f"Failed to generate VAT report: {str(e)}"})

    async def _get_vat_report_async(self, period, user_id, start_date, end_date):
        """Async implementation for VAT report generation"""
        try:
            start_dt, end_dt = self._get_date_range(period, start_date, end_date)
            user_filter = {"userId": user_id} if user_id else {}

            # Get VAT summary
            vat_summary = await self._get_vat_summary(start_dt, end_dt, user_filter)

            # Monthly VAT data (simplified - would need more complex aggregation for real monthly data)
            monthly_vat = [
                {
                    "month": start_dt.strftime("%b %Y"),
                    "vatCollected": vat_summary.vatCollected,
                    "vatDeductible": vat_summary.vatDeductible,
                    "vatBalance": vat_summary.vatBalance
                }
            ]

            report = VatReport(
                vatSummary=vat_summary,
                monthlyVat=monthly_vat
            )

            return json.dumps({
                "reportType": "vat",
                "period": period,
                "startDate": start_dt.isoformat(),
                "endDate": end_dt.isoformat(),
                "generatedAt": datetime.now().isoformat(),
                "data": report.dict()
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to generate VAT report: {str(e)}"})


# Import asyncio at the top level for the async functions
import asyncio
