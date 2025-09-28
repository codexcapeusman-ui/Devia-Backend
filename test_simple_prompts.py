#!/usr/bin/env python3
"""
Test script for simple prompts functionality
Tests the AI services with simple prompts like "show all my clients"
"""

import asyncio
import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.semantic_kernel_service import SemanticKernelService
from services.unified_agent_service import UnifiedAgentService
from config.settings import Settings

async def test_simple_prompts():
    """Test simple prompts with the AI services"""
    
    print("🚀 Testing Simple Prompts with AI Services")
    print("=" * 50)
    
    # Initialize settings and services
    settings = Settings()
    sk_service = SemanticKernelService(settings)
    unified_service = UnifiedAgentService(sk_service)
    
    # Initialize the semantic kernel service
    print("📡 Initializing Semantic Kernel Service...")
    await sk_service.initialize()
    
    if not sk_service.is_initialized():
        print("❌ Failed to initialize Semantic Kernel Service")
        return
    
    print("✅ Semantic Kernel Service initialized successfully")
    
    # Test user ID (simulate a logged-in user)
    test_user_id = "test_user_123"
    
    # Test prompts
    test_prompts = [
        "show all my clients",
        "list my invoices", 
        "get my jobs",
        "display my expenses",
        "show my quotes",
        "get invoice by id 507f1f77bcf86cd799439011",
        "show client with id 507f1f77bcf86cd799439012",
        "list jobs for tomorrow",
        "show overdue invoices"
    ]
    
    print(f"\n🧪 Testing {len(test_prompts)} simple prompts...")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing: '{prompt}'")
        print("-" * 30)
        
        try:
            # Process the prompt through unified agent service
            response = await unified_service.process_agent_request(
                prompt=prompt,
                user_id=test_user_id,
                language="en"
            )
            
            print(f"✅ Response received:")
            print(f"   Success: {response.get('success', False)}")
            print(f"   Message: {response.get('message', 'No message')}")
            print(f"   Intent: {response.get('intent', 'Unknown')}")
            print(f"   Operation: {response.get('operation', 'Unknown')}")
            
            # Show data preview if available
            data = response.get('data')
            if data:
                if isinstance(data, dict):
                    if 'clients' in data:
                        print(f"   📊 Found {len(data['clients'])} clients")
                    elif 'invoices' in data:
                        print(f"   📊 Found {len(data['invoices'])} invoices")
                    elif 'jobs' in data:
                        print(f"   📊 Found {len(data['jobs'])} jobs")
                    elif 'expenses' in data:
                        print(f"   📊 Found {len(data['expenses'])} expenses")
                    elif 'quotes' in data:
                        print(f"   📊 Found {len(data['quotes'])} quotes")
                    else:
                        print(f"   📊 Data keys: {list(data.keys())}")
                else:
                    print(f"   📊 Data type: {type(data)}")
            else:
                print("   📊 No data returned")
                
        except Exception as e:
            print(f"❌ Error processing prompt: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Testing completed!")
    print("=" * 50)

async def test_intent_detection():
    """Test intent detection for various prompts"""
    
    print("\n🔍 Testing Intent Detection")
    print("=" * 30)
    
    settings = Settings()
    sk_service = SemanticKernelService(settings)
    unified_service = UnifiedAgentService(sk_service)
    
    await sk_service.initialize()
    
    test_prompts = [
        ("show all my clients", "customer", "get"),
        ("create new invoice", "invoice", "create"),
        ("get invoice by id 123", "invoice", "get"),
        ("list my jobs", "job", "get"),
        ("schedule meeting", "job", "create"),
        ("display expenses", "expense", "get"),
        ("show my quotes", "quote", "get")
    ]
    
    for prompt, expected_intent, expected_operation in test_prompts:
        print(f"\nTesting: '{prompt}'")
        try:
            intent, operation, confidence = await unified_service._detect_intent(prompt, "en")
            print(f"  Expected: {expected_intent}/{expected_operation}")
            print(f"  Detected: {intent.value}/{operation.value} (confidence: {confidence:.2f})")
            
            if intent.value == expected_intent and operation.value == expected_operation:
                print("  ✅ Correct detection")
            else:
                print("  ⚠️  Detection mismatch")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Simple Prompts Test Suite")
    print("=" * 50)
    
    # Run the tests
    asyncio.run(test_simple_prompts())
    asyncio.run(test_intent_detection())
