"""
Streamlit UI for Devia AI Agent System
Interactive testing interface for the unified AI agent backend
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import advanced testing features
try:
    from streamlit_advanced import render_advanced_testing_tab
    from demo_scenarios import DemoScenarios
except ImportError:
    def render_advanced_testing_tab():
        st.error("Advanced testing features not available")
    
    class DemoScenarios:
        @staticmethod
        def get_all_scenarios():
            return {}
        st.error("Advanced testing features not available. Please ensure streamlit_advanced.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Devia AI Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stAlert > div {
    padding: 10px 15px;
    border-radius: 8px;
}

.success-message {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.error-message {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.info-message {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.intent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    margin: 2px;
}

.intent-invoice { background-color: #e3f2fd; color: #1976d2; }
.intent-quote { background-color: #f3e5f5; color: #7b1fa2; }
.intent-customer { background-color: #e8f5e8; color: #388e3c; }
.intent-job { background-color: #fff3e0; color: #f57c00; }
.intent-expense { background-color: #ffebee; color: #d32f2f; }
.intent-unknown { background-color: #f5f5f5; color: #616161; }
</style>
""", unsafe_allow_html=True)

class DeviaAgentUI:
    """Main UI class for Devia AI Agent System"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"
        self.agent_url = f"{self.base_url}/api/agent"
        
        # Initialize session state
        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if 'current_user' not in st.session_state:
            st.session_state.current_user = "demo_user"
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'system_status' not in st.session_state:
            st.session_state.system_status = None
    
    def check_backend_health(self) -> Dict[str, Any]:
        """Check if backend is running and healthy"""
        try:
            response = requests.get(f"{self.agent_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Backend server is not running or not accessible"
            }
    
    def process_agent_request(
        self, 
        prompt: str, 
        user_id: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """Send request to unified agent endpoint"""
        try:
            payload = {
                "prompt": prompt,
                "user_id": user_id,
                "language": language
            }
            
            response = requests.post(f"{self.agent_url}/process", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Request failed: {str(e)}"
            }
    
    def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """Get conversation status"""
        try:
            payload = {"user_id": user_id}
            response = requests.post(f"{self.agent_url}/conversation/status", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reset_conversation(self, user_id: str) -> Dict[str, Any]:
        """Reset conversation"""
        try:
            payload = {"user_id": user_id}
            response = requests.post(f"{self.agent_url}/conversation/reset", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def render_header(self):
        """Render application header"""
        st.title("🤖 Devia AI Agent System")
        st.markdown("### Interactive Testing Interface for Unified AI Agents")
        
        # System status indicator
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("🔄 Check System Health", type="secondary"):
                    with st.spinner("Checking system health..."):
                        st.session_state.system_status = self.check_backend_health()
            
            with col2:
                status = st.session_state.system_status
                if status:
                    if status.get("status") == "healthy":
                        st.success("✅ System Healthy")
                    else:
                        st.error("❌ System Error")
            
            with col3:
                st.markdown(f"**User ID:** `{st.session_state.current_user}`")
    
    def render_sidebar(self):
        """Render sidebar with controls and status"""
        st.sidebar.header("🔧 Controls")
        
        # User settings
        st.sidebar.subheader("User Settings")
        new_user = st.sidebar.text_input(
            "User ID",
            value=st.session_state.current_user,
            help="Unique identifier for conversation tracking"
        )
        if new_user != st.session_state.current_user:
            st.session_state.current_user = new_user
            st.rerun()
        
        language = st.sidebar.selectbox(
            "Language",
            ["en", "fr"],
            index=0,
            help="Response language"
        )
        
        # Conversation controls
        st.sidebar.subheader("💬 Conversation")
        
        if st.sidebar.button("🗑️ Reset Conversation", type="secondary"):
            with st.spinner("Resetting conversation..."):
                result = self.reset_conversation(st.session_state.current_user)
                if result.get("success"):
                    st.sidebar.success("Conversation reset!")
                    st.session_state.conversation_history = []
                else:
                    st.sidebar.error(f"Reset failed: {result.get('error', 'Unknown error')}")
        
        if st.sidebar.button("📊 Get Status", type="secondary"):
            with st.spinner("Getting status..."):
                status = self.get_conversation_status(st.session_state.current_user)
                if status.get("success"):
                    st.sidebar.json(status.get("data", {}))
                else:
                    st.sidebar.error(f"Status check failed: {status.get('error', 'Unknown error')}")
        
        # System info
        st.sidebar.subheader("🏥 System Health")
        if st.session_state.system_status:
            status = st.session_state.system_status
            if status.get("status") == "healthy":
                st.sidebar.success("✅ Backend Online")
                if "supported_intents" in status:
                    st.sidebar.write("**Supported Intents:**")
                    for intent in status["supported_intents"]:
                        st.sidebar.write(f"• {intent}")
            else:
                st.sidebar.error("❌ Backend Offline")
                st.sidebar.write(status.get("message", "Unknown error"))
        
        # Example prompts
        st.sidebar.subheader("💡 Example Prompts")
        examples = {
            "Invoice": "Create invoice for John Doe at ABC Corp for website development 40 hours at €50/hour",
            "Quote": "Generate quote for website redesign including 3 pages and hosting for XYZ Company",
            "Customer": "Add new customer: Jane Smith, email jane@example.com, phone +33 1 23 45 67 89",
            "Job": "Schedule website maintenance for tomorrow at 2 PM, duration 3 hours",
            "Expense": "Track expense: Office supplies from Staples €45.80 including VAT on September 20, 2025"
        }
        
        for category, example in examples.items():
            if st.sidebar.button(f"📝 {category}", key=f"example_{category}"):
                st.session_state.example_prompt = example
        
        return language
    
    def format_intent_badge(self, intent: str) -> str:
        """Format intent as a colored badge"""
        if not intent or intent == "unknown":
            return f'<span class="intent-badge intent-unknown">UNKNOWN</span>'
        
        intent_classes = {
            "invoice": "intent-invoice",
            "quote": "intent-quote", 
            "customer": "intent-customer",
            "job": "intent-job",
            "expense": "intent-expense"
        }
        
        css_class = intent_classes.get(intent, "intent-unknown")
        return f'<span class="intent-badge {css_class}">{intent.upper()}</span>'
    
    def render_response(self, response: Dict[str, Any], timestamp: str):
        """Render AI agent response"""
        if response.get("success"):
            # Successful response
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.write("✅ **Operation Completed Successfully**")
            
            intent = response.get("intent", "unknown")
            st.markdown(f"**Intent:** {self.format_intent_badge(intent)}", unsafe_allow_html=True)
            
            if "message" in response:
                st.write(f"**Message:** {response['message']}")
            
            # Show the data
            if "data" in response and response["data"]:
                st.write("**Generated Data:**")
                st.json(response["data"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif response.get("action") == "provide_missing_data":
            # Missing data response
            st.markdown('<div class="info-message">', unsafe_allow_html=True)
            st.write("ℹ️ **Additional Information Needed**")
            st.write(response.get("message", "Missing required information"))
            
            if "missing_fields" in response:
                st.write("**Missing Fields:**")
                for field in response["missing_fields"]:
                    st.write(f"• {field}")
            
            if "current_data" in response and response["current_data"]:
                st.write("**Current Data:**")
                st.json(response["current_data"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif response.get("action") == "clarify_intent":
            # Clarification needed
            st.markdown('<div class="info-message">', unsafe_allow_html=True)
            st.write("❓ **Intent Clarification Needed**")
            st.write(response.get("message", "Please clarify what you'd like me to help with"))
            
            if "suggestions" in response:
                st.write("**Suggestions:**")
                for suggestion in response["suggestions"]:
                    st.write(f"• {suggestion}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Error response
            st.markdown('<div class="error-message">', unsafe_allow_html=True)
            st.write("❌ **Error Occurred**")
            st.write(response.get("message", "Unknown error"))
            
            if "error" in response:
                st.write(f"**Error Details:** {response['error']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_chat_interface(self, language: str):
        """Render main chat interface"""
        st.header("💬 Chat with AI Agent")
        
        # Chat input
        demo_prompt_value = st.session_state.get("demo_prompt", "")
        if demo_prompt_value:
            st.info(f"📋 Demo prompt loaded: {demo_prompt_value[:100]}...")
        
        prompt_input = st.text_area(
            "Enter your request:",
            value=demo_prompt_value or st.session_state.get("example_prompt", ""),
            height=100,
            placeholder="Examples:\n• Create invoice for John Doe...\n• Add customer Jane Smith...\n• Schedule maintenance...",
            help="Describe what you want to create or manage. The AI will guide you through the process."
        )
        
        # Clear prompts after use
        if "example_prompt" in st.session_state:
            del st.session_state.example_prompt
        if "demo_prompt" in st.session_state and prompt_input == demo_prompt_value:
            # Keep demo prompt until user modifies it or sends it
            pass
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            send_button = st.button("🚀 Send Request", type="primary", disabled=not prompt_input.strip())
        
        with col2:
            if st.button("🧪 Test Mode"):
                st.session_state.test_mode = True
        
        with col3:
            if st.button("📋 Clear History"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Process request
        if send_button and prompt_input.strip():
            with st.spinner("🤖 AI Agent is processing your request..."):
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Clear demo prompt after sending
                if "demo_prompt" in st.session_state:
                    del st.session_state.demo_prompt
                
                # Add user message to history
                st.session_state.conversation_history.append({
                    "type": "user",
                    "content": prompt_input,
                    "timestamp": timestamp
                })
                
                # Process with AI agent
                response = self.process_agent_request(
                    prompt_input,
                    st.session_state.current_user,
                    language
                )
                
                # Add AI response to history
                st.session_state.conversation_history.append({
                    "type": "agent",
                    "content": response,
                    "timestamp": timestamp
                })
                
                st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.header("📝 Conversation History")
            
            for i, message in enumerate(reversed(st.session_state.conversation_history)):
                with st.container():
                    if message["type"] == "user":
                        st.markdown(f"**👤 You** ({message['timestamp']})")
                        st.write(message["content"])
                    else:
                        st.markdown(f"**🤖 AI Agent** ({message['timestamp']})")
                        self.render_response(message["content"], message["timestamp"])
                    
                    st.divider()
    
    def render_analytics(self):
        """Render analytics and insights"""
        st.header("📊 Analytics & Insights")
        
        if not st.session_state.conversation_history:
            st.info("No conversation data available yet. Start chatting with the AI agent!")
            return
        
        # Extract data for analytics
        intents = []
        response_times = []
        success_rates = []
        
        for message in st.session_state.conversation_history:
            if message["type"] == "agent":
                content = message["content"]
                if "intent" in content:
                    intents.append(content["intent"])
                success_rates.append(1 if content.get("success") else 0)
        
        if intents:
            col1, col2 = st.columns(2)
            
            with col1:
                # Intent distribution
                intent_df = pd.DataFrame({"Intent": intents})
                intent_counts = intent_df["Intent"].value_counts()
                
                fig = px.pie(
                    values=intent_counts.values,
                    names=intent_counts.index,
                    title="Intent Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Success rate
                success_rate = sum(success_rates) / len(success_rates) * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = success_rate,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Success Rate %"},
                    delta = {'reference': 80},
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
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Conversation summary
        st.subheader("📈 Session Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.conversation_history))
        
        with col2:
            agent_messages = len([m for m in st.session_state.conversation_history if m["type"] == "agent"])
            st.metric("Agent Responses", agent_messages)
        
        with col3:
            if intents:
                unique_intents = len(set(intents))
                st.metric("Unique Intents", unique_intents)
        
        with col4:
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates) * 100
                st.metric("Success Rate", f"{avg_success:.1f}%")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        language = self.render_sidebar()
        
        # Render header
        self.render_header()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Chat Interface", 
            "📊 Analytics", 
            "🧪 Advanced Testing", 
            "📖 Documentation",
            "🎯 Demo Scenarios"
        ])
        
        with tab1:
            self.render_chat_interface(language)
        
        with tab2:
            self.render_analytics()
        
        with tab3:
            render_advanced_testing_tab()
        
        with tab4:
            self.render_documentation()
        
        with tab5:
            self.render_demo_scenarios_tab()
    
    def render_documentation(self):
        """Render documentation and help"""
        st.header("📖 Documentation")
        
        st.markdown("""
        ## 🤖 How to Use the AI Agent System
        
        The Devia AI Agent System uses a unified endpoint that can handle all your business operations through natural language.
        
        ### 🎯 Supported Operations
        
        | Intent | Description | Example Prompt |
        |--------|-------------|----------------|
        | **Invoice** | Create invoices from descriptions | *"Create invoice for John Doe at ABC Corp for 40 hours at €50/hour"* |
        | **Quote** | Generate quotes and estimates | *"Quote for website redesign including 3 pages and hosting"* |
        | **Customer** | Add/manage customer data | *"Add customer Jane Smith, email jane@example.com, phone 555-0123"* |
        | **Job** | Schedule jobs and appointments | *"Schedule maintenance for tomorrow at 2 PM, duration 3 hours"* |
        | **Expense** | Track expenses and receipts | *"Office supplies from Staples €45.80 including VAT"* |
        
        ### 🔄 Workflow
        
        1. **Intent Detection**: AI analyzes your prompt to understand what you want
        2. **Data Extraction**: AI extracts relevant information from your prompt  
        3. **Missing Data Check**: AI identifies what information is still needed
        4. **Conversational Flow**: AI asks for missing data when needed
        5. **Result Generation**: AI creates the final result in the correct format
        
        ### 💡 Tips for Better Results
        
        - **Be specific**: Include as much detail as possible in your initial prompt
        - **Use natural language**: Write as you would speak to a human assistant
        - **Provide context**: Mention dates, amounts, names, and other relevant details
        - **Follow up**: If the AI asks for more information, provide it in your next message
        
        ### 🌍 Language Support
        
        The system supports both English (en) and French (fr). You can switch languages in the sidebar.
        
        ### 🔧 Technical Details
        
        - **Backend**: FastAPI with Semantic Kernel and OpenAI GPT models
        - **Endpoint**: `POST /api/agent/process`
        - **Conversation State**: Maintained per user ID
        - **Response Format**: Structured JSON matching your existing manual endpoints
        """)
    
    def render_demo_scenarios_tab(self):
        """Render demo scenarios and examples"""
        st.header("🎯 Demo Scenarios")
        
        st.markdown("""
        ### 📝 Predefined Testing Scenarios
        
        Use these scenarios to test different aspects of the AI agent system. Each scenario includes:
        - **Expected Intent**: What the AI should detect
        - **Expected Outcome**: Success, missing data, or error
        - **Missing Fields**: What information should be requested
        - **Notes**: Additional context about the test
        """)
        
        # Get all scenarios
        all_scenarios = DemoScenarios.get_all_scenarios()
        
        if not all_scenarios:
            st.warning("Demo scenarios not available. Please ensure demo_scenarios.py is properly imported.")
            return
        
        # Category selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            category = st.selectbox(
                "📁 Select Category",
                options=list(all_scenarios.keys()),
                index=0
            )
        
        with col2:
            scenarios = all_scenarios.get(category, [])
            if scenarios:
                scenario_titles = [f"{i+1}. {s['title']}" for i, s in enumerate(scenarios)]
                selected_idx = st.selectbox(
                    "📋 Select Scenario",
                    options=range(len(scenario_titles)),
                    format_func=lambda x: scenario_titles[x]
                )
                selected_scenario = scenarios[selected_idx]
            else:
                st.warning("No scenarios available for this category")
                return
        
        # Display scenario details
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📝 {selected_scenario['title']}")
            st.markdown(f"**Description**: {selected_scenario['description']}")
            
            # Display prompt
            st.markdown("**Test Prompt**:")
            st.code(selected_scenario['prompt'], language="text")
            
            # Copy prompt button
            if st.button("📋 Copy Prompt to Chat", key=f"copy_{category}_{selected_idx}"):
                # Store in session state for use in chat tab
                if 'demo_prompt' not in st.session_state:
                    st.session_state.demo_prompt = ""
                st.session_state.demo_prompt = selected_scenario['prompt']
                st.success("✅ Prompt copied! Go to Chat Interface tab to test.")
        
        with col2:
            st.markdown("**Expected Results**:")
            
            if 'expected_intent' in selected_scenario:
                intent_color = {
                    'invoice': '#FF6B6B',
                    'quote': '#4ECDC4', 
                    'customer': '#45B7D1',
                    'job': '#96CEB4',
                    'expense': '#FECA57',
                    'unknown': '#9B59B6'
                }.get(selected_scenario['expected_intent'], '#95A5A6')
                
                st.markdown(f"""
                <div style="background-color: {intent_color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>Intent:</strong> {selected_scenario['expected_intent']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'expected_outcome' in selected_scenario:
                outcome_color = {
                    'success': '#27AE60',
                    'missing_data': '#F39C12', 
                    'error': '#E74C3C',
                    'clarify_intent': '#8E44AD'
                }.get(selected_scenario['expected_outcome'], '#95A5A6')
                
                st.markdown(f"""
                <div style="background-color: {outcome_color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>Outcome:</strong> {selected_scenario['expected_outcome']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'missing_fields' in selected_scenario and selected_scenario['missing_fields']:
                st.markdown("**Missing Fields**:")
                for field in selected_scenario['missing_fields']:
                    st.markdown(f"• `{field}`")
            
            if 'notes' in selected_scenario:
                st.markdown(f"**Notes**: {selected_scenario['notes']}")
        
        # Special handling for conversation flows
        if 'steps' in selected_scenario:
            st.markdown("---")
            st.subheader("🔄 Conversation Flow")
            
            for i, step in enumerate(selected_scenario['steps']):
                with st.expander(f"Step {i+1}: {step['prompt'][:50]}..."):
                    st.markdown(f"**User Input**: {step['prompt']}")
                    st.markdown(f"**Expected Response**: {step['expected']}")
                    
                    if st.button(f"📋 Copy Step {i+1}", key=f"copy_step_{i}"):
                        st.session_state.demo_prompt = step['prompt']
                        st.success(f"✅ Step {i+1} copied to chat!")
        
        # Quick test all scenarios
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Test This Scenario"):
                # Auto-run the scenario (placeholder for future implementation)
                st.info("Auto-testing feature coming soon!")
        
        with col2:
            if st.button("📊 Run Category Tests"):
                st.info("Batch testing feature available in Advanced Testing tab!")
        
        with col3:
            if st.button("📥 Export Scenarios"):
                scenarios_json = DemoScenarios.export_scenarios_json()
                st.download_button(
                    label="💾 Download JSON",
                    data=scenarios_json,
                    file_name=f"devia_scenarios_{category.lower().replace(' ', '_')}.json",
                    mime="application/json"
                )
        
        # Statistics
        st.markdown("---")
        st.subheader("📈 Scenario Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_scenarios = sum(len(scenarios) for scenarios in all_scenarios.values())
        with col1:
            st.metric("Total Scenarios", total_scenarios)
        
        with col2:
            st.metric("Categories", len(all_scenarios))
        
        with col3:
            current_category_count = len(all_scenarios.get(category, []))
            st.metric("Current Category", current_category_count)
        
        with col4:
            # Count scenarios by intent
            intent_counts = {}
            for cat_scenarios in all_scenarios.values():
                for scenario in cat_scenarios:
                    intent = scenario.get('expected_intent', 'unknown')
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            most_common_intent = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else "none"
            st.metric("Most Tested Intent", most_common_intent)

# Main application
def main():
    """Main function to run the Streamlit app"""
    app = DeviaAgentUI()
    app.run()

if __name__ == "__main__":
    main()