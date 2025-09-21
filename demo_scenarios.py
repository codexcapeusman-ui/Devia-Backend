"""
Demo Scenarios for Devia AI Agent System
Comprehensive testing scenarios with expected outcomes
"""

import json
from typing import Dict, List, Any

class DemoScenarios:
    """Collection of demo scenarios for testing the AI agent system"""
    
    @staticmethod
    def get_invoice_scenarios() -> List[Dict[str, Any]]:
        """Get invoice creation scenarios"""
        return [
            {
                "title": "Complete Invoice Request",
                "description": "Invoice with all required information",
                "prompt": "Create invoice for John Doe at ABC Corporation, email john@abc.com, for website development: 40 hours at €60 per hour, total €2400 including 20% VAT",
                "expected_intent": "invoice",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should create complete invoice immediately"
            },
            {
                "title": "Incomplete Invoice Request",
                "description": "Invoice missing customer email",
                "prompt": "Create invoice for John Doe at ABC Corp for website development 40 hours at €60/hour",
                "expected_intent": "invoice",
                "expected_outcome": "missing_data",
                "missing_fields": ["customer_email"],
                "notes": "Should ask for customer email"
            },
            {
                "title": "Minimal Invoice Request",
                "description": "Very basic invoice request",
                "prompt": "Invoice for website work",
                "expected_intent": "invoice",
                "expected_outcome": "missing_data",
                "missing_fields": ["customer_name", "customer_email", "items", "total_amount"],
                "notes": "Should ask for all major details"
            },
            {
                "title": "Multi-Item Invoice",
                "description": "Invoice with multiple line items",
                "prompt": "Invoice for Tech Solutions Ltd (contact@techsolutions.com): Website development 30 hours at €70/hour, Logo design €500, Hosting setup €200",
                "expected_intent": "invoice",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should handle multiple items correctly"
            }
        ]
    
    @staticmethod
    def get_quote_scenarios() -> List[Dict[str, Any]]:
        """Get quote generation scenarios"""
        return [
            {
                "title": "Website Redesign Quote",
                "description": "Complete quote for website services",
                "prompt": "Generate quote for Jane Smith at XYZ Company (jane@xyz.com) for website redesign including 5 pages, responsive design, SEO optimization, estimated at €3500",
                "expected_intent": "quote",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should create complete quote"
            },
            {
                "title": "E-commerce Quote",
                "description": "Quote for e-commerce solution",
                "prompt": "Quote for e-commerce website with product catalog, shopping cart, payment integration, and admin panel for startup company",
                "expected_intent": "quote",
                "expected_outcome": "missing_data",
                "missing_fields": ["customer_name", "customer_email", "estimated_total"],
                "notes": "Should ask for customer details and pricing"
            },
            {
                "title": "Mobile App Quote",
                "description": "Quote for mobile application",
                "prompt": "Create quote for Sarah Wilson at MobileTech (sarah@mobiletech.com) for iOS and Android app development, estimated 3 months at €8000 per month",
                "expected_intent": "quote",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should handle long-term project quote"
            }
        ]
    
    @staticmethod
    def get_customer_scenarios() -> List[Dict[str, Any]]:
        """Get customer management scenarios"""
        return [
            {
                "title": "Complete Customer Profile",
                "description": "Add customer with all details",
                "prompt": "Add new customer: Mike Johnson from TechStart Solutions, email mike@techstart.com, phone +33 1 45 67 89 12, address 45 Innovation Street, Paris 75015, France",
                "expected_intent": "customer",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should create complete customer profile"
            },
            {
                "title": "Basic Customer Info",
                "description": "Minimal customer information",
                "prompt": "Add customer Lisa Chen, email lisa@example.com",
                "expected_intent": "customer",
                "expected_outcome": "missing_data",
                "missing_fields": ["phone", "address"],
                "notes": "Should ask for phone and address"
            },
            {
                "title": "Customer with Company",
                "description": "Business customer with company details",
                "prompt": "New client: Robert Brown, CEO of Digital Solutions Inc, robert@digitalsolutions.com, office phone +33 1 23 45 67 89, company address 123 Business Avenue, Lyon 69000",
                "expected_intent": "customer",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should handle business customer properly"
            }
        ]
    
    @staticmethod
    def get_job_scenarios() -> List[Dict[str, Any]]:
        """Get job scheduling scenarios"""
        return [
            {
                "title": "Maintenance Appointment",
                "description": "Schedule routine maintenance",
                "prompt": "Schedule website maintenance for ABC Corporation next Tuesday at 2:00 PM, estimated duration 3 hours, contact person John Doe",
                "expected_intent": "job",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should schedule complete appointment"
            },
            {
                "title": "Consultation Meeting",
                "description": "Schedule client consultation",
                "prompt": "Book consultation meeting for new project discussion tomorrow at 10 AM",
                "expected_intent": "job",
                "expected_outcome": "missing_data",
                "missing_fields": ["customer_name", "duration"],
                "notes": "Should ask for customer name and duration"
            },
            {
                "title": "System Upgrade Job",
                "description": "Technical system upgrade",
                "prompt": "Schedule system upgrade for TechCorp on Friday September 27th at 6 PM, after business hours, duration 4 hours, contact Sarah at sarah@techcorp.com",
                "expected_intent": "job",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should handle specific date and after-hours scheduling"
            }
        ]
    
    @staticmethod
    def get_expense_scenarios() -> List[Dict[str, Any]]:
        """Get expense tracking scenarios"""
        return [
            {
                "title": "Office Supplies Expense",
                "description": "Basic expense with VAT",
                "prompt": "Track expense: Office supplies from Staples on September 20, 2025, total €75.60 including 20% VAT, category: office supplies",
                "expected_intent": "expense",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should calculate VAT correctly"
            },
            {
                "title": "Business Meal Expense",
                "description": "Client entertainment expense",
                "prompt": "Business lunch with client at Le Grand Restaurant, September 21, 2025, €125.50 including VAT",
                "expected_intent": "expense",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should categorize as business meal"
            },
            {
                "title": "Software License Expense",
                "description": "Recurring software subscription",
                "prompt": "Record monthly software license fee for Adobe Creative Suite, €59.99 per month, paid September 21, 2025",
                "expected_intent": "expense",
                "expected_outcome": "success",
                "missing_fields": [],
                "notes": "Should handle subscription expenses"
            },
            {
                "title": "Travel Expense",
                "description": "Business travel costs",
                "prompt": "Business trip expense: train ticket Paris to Lyon €95, hotel 2 nights €180, meals €65",
                "expected_intent": "expense",
                "expected_outcome": "missing_data",
                "missing_fields": ["date"],
                "notes": "Should ask for travel dates"
            }
        ]
    
    @staticmethod
    def get_conversation_flow_scenarios() -> List[Dict[str, Any]]:
        """Get multi-step conversation scenarios"""
        return [
            {
                "title": "Invoice Creation Flow",
                "description": "Multi-step invoice creation",
                "steps": [
                    {
                        "prompt": "I need to create an invoice",
                        "expected": "Ask for details (customer, services, amount)"
                    },
                    {
                        "prompt": "It's for ABC Company",
                        "expected": "Ask for email, services, and amount"
                    },
                    {
                        "prompt": "Email is contact@abc.com, website development 25 hours",
                        "expected": "Ask for hourly rate or total amount"
                    },
                    {
                        "prompt": "Rate is €65 per hour",
                        "expected": "Create complete invoice with calculations"
                    }
                ],
                "notes": "Tests conversational flow and context retention"
            },
            {
                "title": "Quote to Invoice Flow",
                "description": "Convert quote to invoice",
                "steps": [
                    {
                        "prompt": "Create quote for website development for TechStart, contact mike@techstart.com, estimated €2500",
                        "expected": "Create quote successfully"
                    },
                    {
                        "prompt": "Convert that quote to an invoice",
                        "expected": "Create invoice based on previous quote"
                    }
                ],
                "notes": "Tests context awareness across different intents"
            }
        ]
    
    @staticmethod
    def get_edge_case_scenarios() -> List[Dict[str, Any]]:
        """Get edge case and error scenarios"""
        return [
            {
                "title": "Unclear Intent",
                "description": "Ambiguous request",
                "prompt": "I need help with something",
                "expected_intent": "unknown",
                "expected_outcome": "clarify_intent",
                "notes": "Should ask for clarification"
            },
            {
                "title": "Mixed Intent",
                "description": "Request with multiple intents",
                "prompt": "Create invoice and also add new customer John Doe",
                "expected_intent": "invoice",  # Should prioritize one
                "expected_outcome": "success or missing_data",
                "notes": "Should handle one intent at a time"
            },
            {
                "title": "Invalid Data",
                "description": "Request with invalid information",
                "prompt": "Create invoice for customer with email 'invalid-email' for €-100",
                "expected_intent": "invoice",
                "expected_outcome": "missing_data",
                "notes": "Should validate data and ask for corrections"
            },
            {
                "title": "Very Long Prompt",
                "description": "Extremely detailed request",
                "prompt": "Create a comprehensive invoice for our client ABC Corporation located at 123 Business Street, Paris, France, contact person John Doe with email john@abc.com and phone +33 1 23 45 67 89, for the following services provided during September 2025: website development including frontend design with React framework 40 hours at €70 per hour totaling €2800, backend API development with Node.js 25 hours at €75 per hour totaling €1875, database design and setup 15 hours at €80 per hour totaling €1200, testing and quality assurance 10 hours at €65 per hour totaling €650, project management and client communication 8 hours at €60 per hour totaling €480, with a total subtotal of €7005 plus 20% VAT of €1401 for a grand total of €8406",
                "expected_intent": "invoice",
                "expected_outcome": "success",
                "notes": "Should handle complex, detailed requests"
            }
        ]
    
    @staticmethod
    def get_multilingual_scenarios() -> List[Dict[str, Any]]:
        """Get multilingual testing scenarios"""
        return [
            {
                "title": "French Invoice Request",
                "description": "Invoice request in French",
                "prompt": "Créer une facture pour Jean Dupont à la société ABC, email jean@abc.com, pour développement web 30 heures à 65€ de l'heure",
                "language": "fr",
                "expected_intent": "invoice",
                "expected_outcome": "success",
                "notes": "Should handle French language correctly"
            },
            {
                "title": "French Customer Addition",
                "description": "Add customer in French",
                "prompt": "Ajouter nouveau client: Marie Martin, email marie@exemple.com, téléphone +33 1 23 45 67 89, adresse 45 rue de la Paix, Paris",
                "language": "fr",
                "expected_intent": "customer",
                "expected_outcome": "success",
                "notes": "Should create customer with French interface"
            }
        ]
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, List[Dict[str, Any]]]:
        """Get all scenarios organized by category"""
        return {
            "Invoice Creation": DemoScenarios.get_invoice_scenarios(),
            "Quote Generation": DemoScenarios.get_quote_scenarios(),
            "Customer Management": DemoScenarios.get_customer_scenarios(),
            "Job Scheduling": DemoScenarios.get_job_scenarios(),
            "Expense Tracking": DemoScenarios.get_expense_scenarios(),
            "Conversation Flows": DemoScenarios.get_conversation_flow_scenarios(),
            "Edge Cases": DemoScenarios.get_edge_case_scenarios(),
            "Multilingual": DemoScenarios.get_multilingual_scenarios()
        }
    
    @staticmethod
    def get_scenario_by_id(category: str, scenario_id: int) -> Dict[str, Any]:
        """Get specific scenario by category and ID"""
        scenarios = DemoScenarios.get_all_scenarios()
        if category in scenarios and 0 <= scenario_id < len(scenarios[category]):
            return scenarios[category][scenario_id]
        return None
    
    @staticmethod
    def export_scenarios_json() -> str:
        """Export all scenarios as JSON for external testing"""
        return json.dumps(DemoScenarios.get_all_scenarios(), indent=2, ensure_ascii=False)