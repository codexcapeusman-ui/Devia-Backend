"""
Example client for testing the Unified AI Agent System
Demonstrates the complete workflow with conversational AI
"""

import json
import requests
import time
from typing import Dict, Any

class DeviaAgentClient:
    """Client for interacting with Devia AI Agent System"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.agent_url = f"{base_url}/api/agent"
    
    def process_request(
        self, 
        prompt: str, 
        user_id: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """Send request to unified agent endpoint"""
        
        payload = {
            "prompt": prompt,
            "user_id": user_id,
            "language": language
        }
        
        try:
            response = requests.post(f"{self.agent_url}/process", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """Get conversation status for user"""
        
        payload = {"user_id": user_id}
        
        try:
            response = requests.post(f"{self.agent_url}/conversation/status", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def reset_conversation(self, user_id: str) -> Dict[str, Any]:
        """Reset conversation for user"""
        
        payload = {"user_id": user_id}
        
        try:
            response = requests.post(f"{self.agent_url}/conversation/reset", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        
        try:
            response = requests.get(f"{self.agent_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

def print_response(response: Dict[str, Any], title: str = "Response"):
    """Pretty print response"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print()

def demo_invoice_workflow():
    """Demonstrate invoice creation workflow"""
    print("\n🧾 INVOICE WORKFLOW DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_invoice"
    
    # Reset conversation
    client.reset_conversation(user_id)
    
    # Step 1: Initial prompt with missing data
    print("Step 1: Initial request with incomplete data")
    response = client.process_request(
        prompt="Create an invoice for website development work",
        user_id=user_id
    )
    print_response(response, "Initial Invoice Request")
    
    # Step 2: Provide missing customer information
    if not response.get("success"):
        print("Step 2: Providing customer information")
        response = client.process_request(
            prompt="The customer is John Doe from ABC Corporation, email john@abc.com",
            user_id=user_id
        )
        print_response(response, "After Customer Info")
    
    # Step 3: Provide missing work details
    if not response.get("success"):
        print("Step 3: Providing work details")
        response = client.process_request(
            prompt="The work was 40 hours of website development at €60 per hour",
            user_id=user_id
        )
        print_response(response, "Final Invoice Creation")

def demo_quote_workflow():
    """Demonstrate quote creation workflow"""
    print("\n💰 QUOTE WORKFLOW DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_quote"
    
    # Reset conversation
    client.reset_conversation(user_id)
    
    # Complete quote in one go
    print("Step 1: Complete quote request")
    response = client.process_request(
        prompt="Create a quote for Jane Smith at XYZ Company (jane@xyz.com) for website redesign including 3 pages, logo design, and 6 months hosting estimated at €2500",
        user_id=user_id
    )
    print_response(response, "Quote Creation")

def demo_customer_workflow():
    """Demonstrate customer data extraction workflow"""
    print("\n👥 CUSTOMER WORKFLOW DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_customer"
    
    # Reset conversation
    client.reset_conversation(user_id)
    
    # Add customer data
    print("Step 1: Adding customer information")
    response = client.process_request(
        prompt="Add new customer: Mike Johnson from Tech Solutions Ltd, email mike@techsolutions.com, phone +33 1 23 45 67 89, address 123 Business Street, Paris 75001",
        user_id=user_id
    )
    print_response(response, "Customer Data Extraction")

def demo_job_workflow():
    """Demonstrate job scheduling workflow"""
    print("\n📅 JOB SCHEDULING WORKFLOW DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_job"
    
    # Reset conversation
    client.reset_conversation(user_id)
    
    # Schedule job with incomplete info
    print("Step 1: Initial job scheduling request")
    response = client.process_request(
        prompt="Schedule website maintenance for next Tuesday",
        user_id=user_id
    )
    print_response(response, "Initial Job Request")
    
    # Provide missing details
    if not response.get("success"):
        print("Step 2: Providing missing details")
        response = client.process_request(
            prompt="It's for ABC Corporation, should start at 2 PM and take about 3 hours",
            user_id=user_id
        )
        print_response(response, "Complete Job Scheduling")

def demo_expense_workflow():
    """Demonstrate expense tracking workflow"""
    print("\n💳 EXPENSE WORKFLOW DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_expense"
    
    # Reset conversation
    client.reset_conversation(user_id)
    
    # Track expense
    print("Step 1: Recording expense")
    response = client.process_request(
        prompt="Track expense: Office supplies from Staples on September 20, 2025 for €75.60 including VAT",
        user_id=user_id
    )
    print_response(response, "Expense Tracking")

def demo_conversation_management():
    """Demonstrate conversation state management"""
    print("\n💬 CONVERSATION MANAGEMENT DEMO")
    print("="*60)
    
    client = DeviaAgentClient()
    user_id = "demo_user_conversation"
    
    # Check initial status
    print("Step 1: Initial conversation status")
    status = client.get_conversation_status(user_id)
    print_response(status, "Initial Status")
    
    # Start conversation
    print("Step 2: Starting conversation")
    response = client.process_request(
        prompt="I need to create something but I'm not sure what",
        user_id=user_id
    )
    print_response(response, "Unclear Intent")
    
    # Check status after unclear intent
    print("Step 3: Status after unclear intent")
    status = client.get_conversation_status(user_id)
    print_response(status, "Status After Unclear Intent")
    
    # Reset conversation
    print("Step 4: Resetting conversation")
    reset_result = client.reset_conversation(user_id)
    print_response(reset_result, "Reset Result")
    
    # Check status after reset
    print("Step 5: Status after reset")
    status = client.get_conversation_status(user_id)
    print_response(status, "Status After Reset")

def main():
    """Run all demo workflows"""
    
    print("🤖 DEVIA AI AGENT SYSTEM - UNIFIED ENDPOINT DEMO")
    print("="*70)
    
    # Check health first
    client = DeviaAgentClient()
    health = client.health_check()
    print_response(health, "Health Check")
    
    if not health.get("status") == "healthy":
        print("❌ System not healthy. Please check your setup.")
        return
    
    print("✅ System is healthy! Running demo workflows...")
    
    # Run all demos
    demo_conversation_management()
    demo_invoice_workflow()
    demo_quote_workflow()
    demo_customer_workflow()
    demo_job_workflow()
    demo_expense_workflow()
    
    print("\n🎉 DEMO COMPLETED!")
    print("="*70)
    print("\n📝 What happened:")
    print("1. ✅ Intent Detection: AI identified what user wanted to create")
    print("2. ✅ Data Extraction: AI extracted relevant information from prompts")
    print("3. ✅ Missing Data Detection: AI identified what information was missing")
    print("4. ✅ Conversational Flow: AI asked for missing data when needed")
    print("5. ✅ Response Generation: AI formatted responses to match manual endpoints")
    print("\n🔧 Next Steps:")
    print("- Replace dummy data with actual business logic")
    print("- Integrate with your existing database")
    print("- Customize response formats to match your frontend needs")
    print("- Add authentication and user management")

if __name__ == "__main__":
    main()