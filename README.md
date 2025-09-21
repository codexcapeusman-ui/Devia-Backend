# Devia AI Agent System

## Overview

The Devia AI Agent System provides a unified, intelligent interface for all business operations. Instead of multiple specialized endpoints, users interact with a single AI agent through natural language that can:

- 🧾 **Create Invoices**: Generate professional invoices from descriptions
- 💰 **Generate Quotes**: Create project quotes and estimates  
- 👥 **Manage Customers**: Add and organize customer information
- 📅 **Schedule Jobs**: Book appointments and work sessions
- 💳 **Track Expenses**: Record and categorize business expenses

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Environment

Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start Backend

```bash
uvicorn main:app --reload --port 8000
```

### 4. Launch Testing UI

```bash
# Using batch script (Windows)
start_ui.bat

# Or manually
streamlit run streamlit_ui.py --server.port 8501
```

### 5. Open Browser

Navigate to: `http://localhost:8501`

## 🎯 How It Works

### Unified Conversation Flow

```
User: "Create invoice for ABC Corp, website development 40 hours at €60/hour"
AI: "I need the customer's email to complete the invoice."
User: "Email is contact@abc.com"  
AI: "Invoice created! ID: INV-001, Total: €2,400"
```

### Intelligent Intent Detection

The AI automatically detects what you want to do:
- **Invoice keywords**: "invoice", "bill", "charge"
- **Quote keywords**: "quote", "estimate", "proposal"  
- **Customer keywords**: "add customer", "new client"
- **Job keywords**: "schedule", "appointment", "book"
- **Expense keywords**: "expense", "cost", "receipt"

### Conversation State Management

- Remembers context within conversations
- Asks for missing information intelligently
- Maintains data until complete
- Provides helpful guidance

## 📱 Using the Testing Interface

### 💬 Chat Interface
- Natural language conversation with AI agent
- Real-time response and guidance
- Conversation history tracking
- Multi-user support

### 📊 Analytics Dashboard
- Intent detection statistics
- Response time monitoring  
- Success/failure metrics
- User activity tracking

### 🧪 Advanced Testing
- **Batch Testing**: Run predefined scenarios automatically
- **Performance Monitor**: Track response times and bottlenecks
- **Debug Console**: Manual request builder with detailed analysis
- **Conversation Analyzer**: Visualize conversation flows

### 🎯 Demo Scenarios
- **150+ Test Scenarios**: Covering all intents and edge cases
- **Conversation Flows**: Multi-step interaction examples
- **Expected Outcomes**: Know what to expect from each test
- **One-Click Testing**: Copy scenarios directly to chat

### 📖 Documentation
- Complete API reference
- Usage examples and best practices
- Troubleshooting guides
- Integration instructions

## 🔧 API Usage

### Unified Endpoint

```bash
POST http://localhost:8000/api/agent/process
```

**Request:**
```json
{
  "prompt": "Create invoice for John Doe at ABC Corp for website development",
  "user_id": "user123",
  "conversation_id": "conv456"
}
```

**Response:**
```json
{
  "conversation_id": "conv_12345",
  "intent": "invoice", 
  "confidence": 0.95,
  "status": "missing_data",
  "response": "I need the customer's email and total amount to complete the invoice.",
  "missing_fields": ["customer_email", "total_amount"],
  "context": {
    "customer_name": "John Doe",
    "company": "ABC Corp",
    "service": "website development"
  }
}
```

### Conversation Management

```bash
# Get conversation status
GET /api/agent/conversation/{conversation_id}/status

# Reset conversation
DELETE /api/agent/conversation/{conversation_id}/reset
```

## 🎪 Example Prompts

### Complete Requests (Immediate Success)
```
"Create invoice for John Doe at ABC Corp, email john@abc.com, for website development: 40 hours at €60/hour, total €2400"

"Generate quote for TechStart (info@techstart.com) for mobile app development, estimated €15,000"

"Add customer: Sarah Wilson, email sarah@tech.com, phone +33123456789, address 123 Business St, Paris"
```

### Partial Requests (AI Will Ask for More)
```
"I need to create an invoice"
"Generate a quote for website redesign" 
"Add new customer John Smith"
"Schedule appointment tomorrow"
"Record business expense for lunch"
```

### Conversational Follow-ups
```
User: "Create invoice for ABC Company"
AI: "I need the customer email, services provided, and total amount."
User: "Email is contact@abc.com, website maintenance €500"
AI: "Invoice created successfully! ID: INV-001"
```

## 🧪 Testing Scenarios

The system includes 150+ predefined test scenarios covering:

### **Invoice Scenarios**
- Complete invoice with all details
- Missing customer information
- Multi-item invoices
- VAT calculations

### **Quote Scenarios** 
- Website development quotes
- E-commerce solutions
- Mobile app projects
- Long-term contracts

### **Customer Scenarios**
- Complete customer profiles
- Business vs individual customers
- Missing contact information
- Company details

### **Job Scenarios**
- Maintenance appointments
- Consultation meetings
- After-hours scheduling
- Recurring appointments

### **Expense Scenarios**
- Office supplies with VAT
- Business meals
- Software licenses
- Travel expenses

### **Edge Cases**
- Ambiguous requests
- Invalid data
- Mixed intents
- Very long prompts

### **Multilingual Tests**
- French language support
- Cross-language consistency
- International formatting

## 🔍 Advanced Features

### **Intent Detection**
- Confidence scoring (0.0-1.0)
- Fallback to clarification when uncertain
- Context-aware intent recognition
- Multi-language support

### **Data Extraction**
- Smart field parsing from natural language
- Validation and error handling
- Contextual data completion
- Format standardization

### **Conversation Management**
- Per-user conversation state
- Context retention across messages
- Missing data tracking
- Intelligent follow-up questions

### **Response Formatting**
- Matches existing manual endpoint formats
- Structured JSON responses
- Error handling with specific guidance
- Success confirmations with relevant data

## 🛠️ Development

### Project Structure
```
Devia-Backend/
├── main.py                     # FastAPI application
├── api/routes.py              # Unified API endpoint
├── services/
│   ├── unified_agent_service.py  # Core AI agent logic
│   └── semantic_kernel_service.py # AI processing
├── streamlit_ui.py            # Main testing interface
├── streamlit_advanced.py      # Advanced testing features
├── demo_scenarios.py          # Test scenarios
├── demo_client.py            # CLI testing client
├── start_ui.bat              # Windows startup script
└── requirements.txt          # Dependencies
```

### Adding New Intents

1. Add to `Intent` enum in `unified_agent_service.py`
2. Define required fields in `REQUIRED_FIELDS`
3. Add keywords to `_detect_intent()` method
4. Implement extraction in `_extract_data()` method
5. Add test scenarios to `demo_scenarios.py`

### Configuration

**Intent Detection Confidence Thresholds:**
- High: ≥ 0.8 (proceed immediately)
- Medium: 0.6-0.79 (proceed with validation)
- Low: < 0.6 (ask for clarification)

**Conversation Settings:**
- Session timeout: 30 minutes
- Max exchanges: 50 per conversation
- Context retention: Per user basis

## 🐛 Troubleshooting

### Common Issues

**Low Confidence Scores**
- Add more specific keywords to intent detection
- Provide clearer prompts with context
- Check for typos or unclear language

**Missing Data Not Detected**
- Verify field validation logic
- Check required fields mapping
- Test with debug console

**Context Not Maintained**
- Ensure conversation_id consistency
- Check session state management
- Verify user_id format

**Slow Response Times**
- Monitor OpenAI API limits
- Check network connectivity
- Use Performance Monitor in testing UI

### Debug Tools

- **Debug Console**: Manual request testing
- **Performance Monitor**: Response time analysis
- **Conversation Analyzer**: State visualization
- **Batch Testing**: Automated validation

## 🔐 Security & Performance

### Security
- Input validation and sanitization
- Rate limiting implementation
- Secure conversation data storage
- API key protection

### Performance
- Response caching where appropriate
- Optimized OpenAI prompts
- Efficient conversation state management
- Response compression

## 📈 Monitoring

The system provides comprehensive monitoring:
- Real-time response time tracking
- Intent detection accuracy metrics
- Conversation success rates
- User activity analytics
- Error rate monitoring

## 🌍 Language Support

Currently supports:
- **English (en)**: Full feature support
- **French (fr)**: Complete localization

Additional languages can be added by:
1. Adding translations to `i18n/locales/`
2. Testing with multilingual scenarios
3. Updating intent detection keywords

## 📝 License

This project is part of the Devia business management system.

---

For detailed API documentation, see `AI_AGENT_DOCUMENTATION.md`

For comprehensive testing scenarios, see `demo_scenarios.py`

For advanced features, explore the Streamlit testing interface!