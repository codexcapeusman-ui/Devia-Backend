# Devia AI Agent System Documentation

## Overview

The Devia AI Agent System is a unified intelligent agent that replaces multiple specialized endpoints with a single, conversational interface. It uses natural language processing to detect user intent, extract required data, and manage conversational flows to gather missing information.

## Architecture

### Core Components

1. **Unified Agent Service** (`services/unified_agent_service.py`)
   - Intent detection with confidence scoring
   - Data extraction for each domain
   - Conversation state management
   - Response formatting

2. **Unified API Endpoint** (`api/routes.py`)
   - Single `/api/agent/process` endpoint
   - Conversation management routes
   - Error handling and validation

3. **Streamlit Testing Interface** (`streamlit_ui.py` + `streamlit_advanced.py`)
   - Interactive chat interface
   - Analytics and monitoring
   - Advanced testing features
   - Batch testing capabilities

## Supported Intents

### 1. Invoice Creation (`invoice`)
**Purpose**: Create invoices for completed work or services

**Required Fields**:
- `customer_name`: Client name
- `customer_email`: Client email address
- `items`: List of services/products
- `total_amount`: Invoice total

**Optional Fields**:
- `due_date`: Payment due date
- `tax_rate`: VAT/Tax percentage
- `notes`: Additional notes

**Example Prompts**:
- "Create invoice for John Doe at ABC Corp, email john@abc.com, for website development 40 hours at €60/hour"
- "Invoice ABC Company for website maintenance €500 due in 30 days"

### 2. Quote Generation (`quote`)
**Purpose**: Generate quotes for potential projects

**Required Fields**:
- `customer_name`: Prospect name
- `customer_email`: Prospect email
- `services`: Proposed services
- `estimated_total`: Quote amount

**Optional Fields**:
- `valid_until`: Quote expiration date
- `terms`: Terms and conditions
- `notes`: Additional information

**Example Prompts**:
- "Generate quote for Jane Smith at XYZ Company for website redesign €3500"
- "Quote for mobile app development, customer sarah@tech.com, estimated €15000"

### 3. Customer Management (`customer`)
**Purpose**: Add or manage customer information

**Required Fields**:
- `name`: Customer name
- `email`: Customer email
- `phone`: Phone number
- `address`: Customer address

**Optional Fields**:
- `company`: Company name
- `notes`: Additional notes

**Example Prompts**:
- "Add customer Mike Johnson, email mike@tech.com, phone +33123456789"
- "New client: ABC Corp, contact john@abc.com, located in Paris"

### 4. Job Scheduling (`job`)
**Purpose**: Schedule appointments and work sessions

**Required Fields**:
- `customer_name`: Client name
- `date`: Appointment date
- `time`: Appointment time
- `duration`: Expected duration

**Optional Fields**:
- `description`: Job description
- `location`: Meeting location
- `notes`: Additional notes

**Example Prompts**:
- "Schedule maintenance for ABC Corp next Tuesday 2 PM, 3 hours"
- "Book consultation with John Doe tomorrow 10 AM"

### 5. Expense Tracking (`expense`)
**Purpose**: Record business expenses

**Required Fields**:
- `amount`: Expense amount
- `date`: Expense date
- `category`: Expense category
- `description`: Expense description

**Optional Fields**:
- `vat_rate`: VAT percentage
- `receipt_number`: Receipt reference
- `notes`: Additional notes

**Example Prompts**:
- "Record expense: Office supplies €75.60 including VAT on September 20"
- "Business lunch €125.50 with client yesterday"

## API Endpoints

### Process Agent Request
```
POST /api/agent/process
```

**Request Body**:
```json
{
  "prompt": "Create invoice for John Doe...",
  "user_id": "user123",
  "conversation_id": "conv456" // optional
}
```

**Response Format**:
```json
{
  "conversation_id": "conv_12345",
  "intent": "invoice",
  "confidence": 0.95,
  "status": "success" | "missing_data" | "error",
  "response": "Invoice created successfully",
  "data": {
    "invoice_id": "INV-001",
    "total": 2400
  },
  "missing_fields": [],
  "context": {
    "current_data": {...},
    "next_step": "..."
  }
}
```

### Conversation Management
```
GET /api/agent/conversation/{conversation_id}/status
DELETE /api/agent/conversation/{conversation_id}/reset
```

## Conversation Flow

### 1. Intent Detection
- Analyzes user prompt for intent keywords and context
- Assigns confidence score (0.0-1.0)
- Falls back to clarification if confidence < 0.6

### 2. Data Extraction
- Extracts relevant information based on detected intent
- Validates extracted data format and completeness
- Stores partial data in conversation state

### 3. Missing Data Check
- Compares extracted data against required fields
- Identifies missing or invalid information
- Generates specific questions for missing data

### 4. Response Generation
- If complete: Processes request and returns result
- If incomplete: Asks for missing information
- Maintains conversation context for follow-up

### 5. Context Management
- Stores conversation state per user/conversation
- Maintains extracted data across interactions
- Supports context switching and clarification

## Testing Interface

### Main Chat Interface
1. **User Selection**: Choose or create test users
2. **Chat Window**: Interactive conversation with agent
3. **Response Display**: Formatted agent responses with data
4. **Conversation History**: Track chat sessions

### Analytics Dashboard
- Intent distribution charts
- Response time metrics
- Success/failure rates
- Conversation flow analysis

### Advanced Testing Features

#### Batch Testing
- Predefined test scenarios for all intents
- Automated testing with expected outcomes
- Success/failure reporting
- Performance metrics

#### Performance Monitor
- Real-time response time tracking
- Historical performance data
- Bottleneck identification
- System health metrics

#### Debug Console
- Manual request builder
- JSON context viewer
- Detailed response analysis
- Conversation state inspection

## Usage Examples

### Complete Workflows

#### Invoice Creation Flow
```
User: "I need to create an invoice"
Agent: "I'll help you create an invoice. Could you provide the customer name, email, services provided, and total amount?"

User: "It's for ABC Company"
Agent: "Great! I have the customer name as ABC Company. I still need their email address, the services provided, and the total amount."

User: "Email is contact@abc.com, website development 25 hours at €65/hour"
Agent: "Perfect! I have:
- Customer: ABC Company
- Email: contact@abc.com  
- Service: Website development (25 hours × €65/hour = €1,625)

Invoice created successfully! Invoice ID: INV-001, Total: €1,625"
```

#### Quote Generation Flow
```
User: "Generate quote for website redesign for TechStart, estimated €3500"
Agent: "I need the customer's email address to complete the quote."

User: "Email is info@techstart.com"
Agent: "Quote generated successfully! Quote ID: QUO-001 for TechStart (info@techstart.com), Website redesign: €3,500"
```

### Error Handling
```
User: "Create invoice for invalid-email for €-100"
Agent: "I found some issues with the invoice details:
- Email format appears invalid
- Amount cannot be negative
Please provide a valid email address and positive amount."
```

## Configuration

### Intent Detection Thresholds
- High confidence: ≥ 0.8 (proceed with extraction)
- Medium confidence: 0.6-0.79 (proceed with caution)
- Low confidence: < 0.6 (ask for clarification)

### Conversation Settings
- Session timeout: 30 minutes
- Maximum conversation length: 50 exchanges
- Context retention: Per user basis

### Response Formatting
- Success responses include relevant data
- Error responses include specific guidance
- Missing data responses ask targeted questions

## Development

### Running the System

1. **Start Backend**:
   ```bash
   cd Devia-Backend
   uvicorn main:app --reload --port 8000
   ```

2. **Start UI** (using batch script):
   ```bash
   start_ui.bat
   ```

3. **Manual UI Start**:
   ```bash
   streamlit run streamlit_ui.py --server.port 8501
   ```

### Testing

1. **Interactive Testing**: Use Streamlit UI for manual testing
2. **Batch Testing**: Use Advanced Testing tab for automated scenarios
3. **API Testing**: Use demo_client.py for direct API testing
4. **Unit Testing**: Run pytest on individual components

### Adding New Intents

1. Add intent to `Intent` enum in `unified_agent_service.py`
2. Define required fields in `REQUIRED_FIELDS` mapping
3. Add intent detection keywords in `_detect_intent` method
4. Implement data extraction logic in `_extract_data` method
5. Add response formatting in appropriate method
6. Create test scenarios in `demo_scenarios.py`

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**: Add more specific keywords for intent detection
2. **Missing Data Not Detected**: Check field validation logic
3. **Context Not Maintained**: Verify conversation_id consistency
4. **Slow Response Times**: Check OpenAI API limits and optimize prompts

### Debug Steps

1. Use Debug Console in Advanced Testing
2. Check conversation state in response context
3. Verify intent detection confidence scores
4. Analyze response timing in Performance Monitor

## Security Considerations

- Validate all user input
- Sanitize extracted data
- Implement rate limiting
- Secure conversation data storage
- Audit conversation logs

## Performance Optimization

- Cache common intent patterns
- Optimize OpenAI prompts
- Implement response compression
- Use conversation state efficiently
- Monitor API usage limits

---

*This documentation covers the complete Devia AI Agent System. For specific implementation details, refer to the source code and inline comments.*