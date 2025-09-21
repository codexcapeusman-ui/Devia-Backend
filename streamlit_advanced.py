"""
Advanced Testing Components for Streamlit UI
Provides detailed testing, debugging, and performance analysis
"""

import streamlit as st
import requests
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class AdvancedTesting:
    """Advanced testing features for the AI Agent system"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.agent_url = f"{base_url}/api/agent"
    
    def render_batch_testing(self):
        """Render batch testing interface"""
        st.subheader("🧪 Batch Testing")
        
        # Predefined test scenarios
        test_scenarios = {
            "Invoice Creation": [
                "Create invoice for John Doe at ABC Corp for website development 40 hours at €50/hour",
                "Invoice for Jane Smith at XYZ Ltd, consulting services €1500 including VAT",
                "Generate invoice for Tech Solutions, 20 hours programming at €75/hour"
            ],
            "Quote Generation": [
                "Quote for website redesign including 3 pages, logo design, and hosting for ABC Company",
                "Generate quote for mobile app development for startup company",
                "Quote for e-commerce website with payment integration and inventory management"
            ],
            "Customer Management": [
                "Add new customer: Mike Johnson from Tech Solutions Ltd, email mike@techsolutions.com, phone +33 1 23 45 67 89",
                "Customer data: Sarah Wilson, sarah@example.com, 555-0123, 123 Business St, Paris",
                "New client: Robert Brown, robert@rbcompany.com, address 456 Tech Avenue, Lyon"
            ],
            "Job Scheduling": [
                "Schedule website maintenance for ABC Corp next Tuesday at 2 PM, should take 3 hours",
                "Book appointment for system upgrade tomorrow at 10 AM, duration 2 hours",
                "Schedule consultation meeting for Friday at 3 PM with new client"
            ],
            "Expense Tracking": [
                "Track expense: Office supplies from Staples €45.80 including VAT on September 20, 2025",
                "Expense: Business lunch €85.50 with VAT, restaurant Le Bistro, September 21, 2025",
                "Record expense: Software license €199 monthly subscription, Microsoft Office"
            ]
        }
        
        # Select test category
        category = st.selectbox("Select Test Category", list(test_scenarios.keys()))
        
        # Show test prompts
        st.write("**Test Prompts:**")
        selected_prompts = []
        for i, prompt in enumerate(test_scenarios[category]):
            if st.checkbox(f"Test {i+1}", key=f"test_{category}_{i}"):
                selected_prompts.append(prompt)
            st.write(f"• {prompt}")
        
        # Batch test settings
        col1, col2 = st.columns(2)
        with col1:
            test_user_id = st.text_input("Test User ID", value="batch_test_user")
        with col2:
            delay_between_tests = st.number_input("Delay between tests (seconds)", min_value=0, max_value=10, value=1)
        
        # Run batch tests
        if st.button("🚀 Run Batch Tests") and selected_prompts:
            self.run_batch_tests(selected_prompts, test_user_id, delay_between_tests)
    
    def run_batch_tests(self, prompts: List[str], user_id: str, delay: int):
        """Execute batch tests and display results"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, prompt in enumerate(prompts):
            status_text.text(f"Testing {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.agent_url}/process",
                    json={"prompt": prompt, "user_id": f"{user_id}_{i}", "language": "en"},
                    timeout=30
                )
                end_time = time.time()
                
                result = {
                    "prompt": prompt,
                    "response": response.json(),
                    "response_time": end_time - start_time,
                    "status_code": response.status_code,
                    "success": response.status_code == 200 and response.json().get("success", False)
                }
            except Exception as e:
                end_time = time.time()
                result = {
                    "prompt": prompt,
                    "response": {"error": str(e)},
                    "response_time": end_time - start_time,
                    "status_code": 500,
                    "success": False
                }
            
            results.append(result)
            progress_bar.progress((i + 1) / len(prompts))
            
            if delay > 0 and i < len(prompts) - 1:
                time.sleep(delay)
        
        status_text.text("Batch testing completed!")
        
        # Display results
        self.display_batch_results(results)
    
    def display_batch_results(self, results: List[Dict[str, Any]]):
        """Display batch test results"""
        st.subheader("📊 Batch Test Results")
        
        # Summary metrics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["response_time"] for r in results) / total_tests
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Success Rate", f"{(successful_tests/total_tests)*100:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        # Detailed results
        for i, result in enumerate(results):
            with st.expander(f"Test {i+1}: {'✅' if result['success'] else '❌'} ({result['response_time']:.2f}s)"):
                st.write("**Prompt:**", result["prompt"])
                st.write("**Response Time:**", f"{result['response_time']:.2f} seconds")
                st.write("**Status Code:**", result["status_code"])
                st.json(result["response"])
    
    def render_performance_monitor(self):
        """Render performance monitoring interface"""
        st.subheader("📈 Performance Monitor")
        
        if "performance_data" not in st.session_state:
            st.session_state.performance_data = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run Performance Test"):
                self.run_performance_test()
        
        with col2:
            if st.button("🗑️ Clear Performance Data"):
                st.session_state.performance_data = []
                st.rerun()
        
        # Display performance data
        if st.session_state.performance_data:
            df = pd.DataFrame(st.session_state.performance_data)
            
            # Response time chart
            fig = px.line(
                df, 
                x="timestamp", 
                y="response_time",
                color="intent",
                title="Response Time Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.write("**Performance Summary:**")
            summary = df.groupby("intent").agg({
                "response_time": ["mean", "min", "max"],
                "success": "mean"
            }).round(3)
            st.dataframe(summary)
    
    def run_performance_test(self):
        """Run a performance test with various prompts"""
        test_prompts = [
            ("invoice", "Create invoice for test customer €100"),
            ("quote", "Generate quote for basic website"),
            ("customer", "Add customer John Test, john@test.com"),
            ("job", "Schedule test job for tomorrow"),
            ("expense", "Track test expense €50")
        ]
        
        for intent, prompt in test_prompts:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.agent_url}/process",
                    json={"prompt": prompt, "user_id": f"perf_test_{intent}", "language": "en"},
                    timeout=15
                )
                end_time = time.time()
                
                st.session_state.performance_data.append({
                    "timestamp": datetime.now(),
                    "intent": intent,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200 and response.json().get("success", False)
                })
            except Exception:
                end_time = time.time()
                st.session_state.performance_data.append({
                    "timestamp": datetime.now(),
                    "intent": intent,
                    "response_time": end_time - start_time,
                    "success": False
                })
        
        st.rerun()
    
    def render_debug_console(self):
        """Render debug console for detailed testing"""
        st.subheader("🐛 Debug Console")
        
        # Manual request builder
        st.write("**Manual Request Builder:**")
        
        col1, col2 = st.columns(2)
        with col1:
            debug_prompt = st.text_area("Prompt", height=100)
            debug_user_id = st.text_input("User ID", value="debug_user")
        
        with col2:
            debug_language = st.selectbox("Language", ["en", "fr"])
            debug_context = st.text_area("Context (JSON)", height=100, placeholder='{"key": "value"}')
        
        if st.button("🔍 Send Debug Request"):
            if debug_prompt:
                # Parse context
                context = None
                if debug_context.strip():
                    try:
                        context = json.loads(debug_context)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in context field")
                        return
                
                # Build payload
                payload = {
                    "prompt": debug_prompt,
                    "user_id": debug_user_id,
                    "language": debug_language
                }
                if context:
                    payload["context"] = context
                
                # Send request with timing
                start_time = time.time()
                try:
                    with st.spinner("Sending debug request..."):
                        response = requests.post(f"{self.agent_url}/process", json=payload, timeout=30)
                    end_time = time.time()
                    
                    # Display results
                    st.write(f"**Response Time:** {end_time - start_time:.3f} seconds")
                    st.write(f"**Status Code:** {response.status_code}")
                    
                    if response.headers.get('content-type', '').startswith('application/json'):
                        result = response.json()
                        st.json(result)
                    else:
                        st.text(response.text)
                
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
    def render_conversation_analyzer(self):
        """Render conversation analysis tools"""
        st.subheader("💬 Conversation Analyzer")
        
        # Get conversation status
        user_id = st.text_input("Analyze User ID", value="demo_user")
        
        if st.button("📊 Analyze Conversation"):
            try:
                response = requests.post(
                    f"{self.agent_url}/conversation/status",
                    json={"user_id": user_id},
                    timeout=10
                )
                
                if response.status_code == 200:
                    status = response.json()
                    if status.get("success"):
                        data = status.get("data", {})
                        
                        # Display conversation info
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Status:**", data.get("status", "Unknown"))
                            st.write("**State:**", data.get("state", "Unknown"))
                            st.write("**Intent:**", data.get("intent", "None"))
                        
                        with col2:
                            confidence = data.get("confidence", 0)
                            st.write("**Confidence:**", f"{confidence:.2f}")
                            st.write("**Has Data:**", data.get("has_data", False))
                            st.write("**Created:**", data.get("created_at", "Unknown"))
                        
                        # Confidence gauge
                        if confidence > 0:
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = confidence * 100,
                                title = {'text': "Intent Confidence %"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 70
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No active conversation found")
                else:
                    st.error(f"API error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def render_advanced_testing_tab():
    """Render the advanced testing tab"""
    testing = AdvancedTesting("http://127.0.0.1:8000")
    
    # Sub-tabs for different testing features
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "🧪 Batch Testing", 
        "📈 Performance", 
        "🐛 Debug Console", 
        "💬 Conversation Analysis"
    ])
    
    with sub_tab1:
        testing.render_batch_testing()
    
    with sub_tab2:
        testing.render_performance_monitor()
    
    with sub_tab3:
        testing.render_debug_console()
    
    with sub_tab4:
        testing.render_conversation_analyzer()