import streamlit as st
import json
import os
import hashlib
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load your product data
PRODUCT_DATA = {
    "products": [
        {
            "name": "OptimAIze Buddy",
            "description": "A multilingual, AI-powered assistant for navigating complex services using verified data sources.",
            "business_context": {
                "target_market": "Enterprise customer service, Government services, Healthcare portals",
                "revenue_model": "SaaS subscription, Pay-per-query, Enterprise licensing",
                "kpis": ["User satisfaction score", "Support ticket reduction", "Multilingual adoption rate", "Average resolution time"],
                "competitive_advantage": "24/7 multilingual support, Verified data sources, Form automation",
                "growth_metrics": "Monthly active users, Query success rate, Cost per resolution",
                "business_impact": "Reduces support costs by 40-60%, Improves customer satisfaction by 30%",
                "challenges": "Integration complexity, Data verification overhead, Language model accuracy"
            },
            "features": [
                "Answers queries across multiple domains",
                "24/7 multilingual access",
                "Form filling & booking inside chat",
                "Reduces support burden"
            ]
        },
        {
            "name": "OptimAIze Automation",
            "description": "AI-based document screening solution for automating application and compliance workflows.",
            "business_context": {
                "target_market": "Banking, Insurance, Legal, Government compliance",
                "revenue_model": "Transaction-based pricing, Enterprise contracts, API calls",
                "kpis": ["Processing time reduction", "Error rate reduction", "Compliance accuracy", "Manual review reduction"],
                "competitive_advantage": "Real-time error detection, Content quality analysis, Multi-format support",
                "growth_metrics": "Documents processed per month, Accuracy improvement, Customer retention rate",
                "business_impact": "Increases processing speed by 70%, Reduces compliance errors by 85%",
                "challenges": "Document variability, Regulatory changes, Integration with legacy systems"
            },
            "features": [
                "Validates file types & completeness",
                "Content quality analysis",
                "Flags errors in real-time"
            ]
        },
        {
            "name": "OptimAIze Assist",
            "description": "GenAI-powered tool trained on manuals and SOPs to assist field operators & engineers in real-time.",
            "business_context": {
                "target_market": "Manufacturing, Utilities, Oil & Gas, Telecommunications",
                "revenue_model": "Per-user subscription, Equipment-based licensing, Service contracts",
                "kpis": ["First-time fix rate", "Mean time to repair", "Knowledge utilization", "Escalation rate reduction"],
                "competitive_advantage": "Trained on proprietary manuals, Real-time troubleshooting, Escalation automation",
                "growth_metrics": "Active technicians, Solved incidents per day, Manual usage reduction",
                "business_impact": "Reduces equipment downtime by 35%, Improves first-time fix rate by 50%",
                "challenges": "Knowledge base maintenance, Field connectivity, Technician adoption"
            },
            "features": [
                "Answers from manuals & logs",
                "Summarizes troubleshooting",
                "Escalation automation"
            ]
        },
        {
            "name": "OptimAIze Grader",
            "description": "An academic assistant that learns from human feedback and automates rubric-based grading.",
            "business_context": {
                "target_market": "Educational institutions, Online learning platforms, Corporate training",
                "revenue_model": "Per-student pricing, Institutional licensing, Pay-per-assessment",
                "kpis": ["Grading time reduction", "Grading consistency", "Feedback quality", "Instructor satisfaction"],
                "competitive_advantage": "Learning from feedback, Rubric-based scoring, Standardization",
                "growth_metrics": "Number of assessments, Institutions using, Student satisfaction",
                "business_impact": "Reduces grading time by 80%, Improves grading consistency by 95%",
                "challenges": "Rubric complexity, Subjectivity handling, Institutional adoption"
            },
            "features": [
                "Rubric-based scoring",
                "Learns over time",
                "Standardizes evaluation"
            ]
        },
        {
            "name": "OptimAIze PID Reader",
            "description": "Engineering drawing intelligence system that analyzes P&ID drawings to detect, track, and map pipeline paths, instruments, and equipment.",
            "business_context": {
                "target_market": "Engineering firms, Construction companies, Oil & Gas, Chemical plants",
                "revenue_model": "Per-drawing analysis, Project-based pricing, Enterprise licensing",
                "kpis": ["Drawing analysis time", "Error detection rate", "Compliance accuracy", "Project risk reduction"],
                "competitive_advantage": "AutoCAD DXF/CAD support, Instrument identification, Pipeline mapping",
                "growth_metrics": "Drawings processed, Error prevention rate, Project acceleration",
                "business_impact": "Reduces design review time by 65%, Prevents construction errors by 90%",
                "challenges": "Drawing format variations, Legacy drawing quality, Industry standards compliance"
            },
            "features": [
                "Identifies pipeline start/end points",
                "Recognizes instruments and equipment",
                "Maps pipeline connections",
                "Detects design errors",
                "Classifies components by type"
            ]
        },
        {
            "name": "OptimAIze Price Predictor",
            "description": "Market intelligence system that predicts vehicle/product prices using statistical and LLM approaches based on comprehensive datasets.",
            "business_context": {
                "target_market": "Automotive dealers, E-commerce platforms, Insurance companies, Financial institutions",
                "revenue_model": "Per-prediction API, Subscription plans, Enterprise analytics",
                "kpis": ["Prediction accuracy", "Market coverage", "Response time", "Customer adoption"],
                "competitive_advantage": "Multi-feature analysis, Real-time market data, Statistical + LLM hybrid",
                "growth_metrics": "Predictions per month, Market segments covered, Accuracy improvement",
                "business_impact": "Improves pricing accuracy by 25%, Reduces market research time by 75%",
                "challenges": "Data quality variability, Market volatility, Feature importance weighting"
            },
            "features": [
                "Uses features: price, brand, model, mileage, transmission, CO2 emissions, emission class, fuel type, warranty",
                "Statistical and LLM-based predictions",
                "Market trend analysis",
                "Competitive pricing insights",
                "Real-time market data integration"
            ],
            "prediction_approach": "Combines statistical regression models with LLM-based market intelligence for hybrid predictions"
        }
    ]
}

def setup_openai():
    """Set up OpenAI API from environment variables using dotenv"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # Try to get from Streamlit secrets (for deployment)
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            st.sidebar.error("""
            🔑 **OpenAI API Key not found!**
            
            Please set your OpenAI API key in one of these ways:
            
            ### **Method 1: Environment Variable (Recommended for Local)**
            1. Create a `.env` file in your project root
            2. Add this line to `.env`:
            ```
            OPENAI_API_KEY=your-api-key-here
            ```
            3. The app will automatically load it
            
            ### **Method 2: Streamlit Secrets (for Cloud Deployment)**
            1. Create `.streamlit/secrets.toml`
            2. Add this line:
            ```toml
            OPENAI_API_KEY = "your-api-key-here"
            ```
            
            ### **Method 3: System Environment Variable**
            ```bash
            # On Mac/Linux:
            export OPENAI_API_KEY="your-api-key-here"
            
            # On Windows (Command Prompt):
            set OPENAI_API_KEY=your-api-key-here
            
            # On Windows (PowerShell):
            $env:OPENAI_API_KEY="your-api-key-here"
            ```
            """)
            
            # Fallback to sidebar input for testing
            temp_key = st.sidebar.text_input("Enter OpenAI API Key (temporary):", type="password")
            if temp_key:
                api_key = temp_key
                st.sidebar.success("Using temporary API key (not saved)")
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            
            # Test the connection with a simple request
            try:
                test_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.sidebar.success("✅ OpenAI API connected successfully")
                return client
            except Exception as test_error:
                st.sidebar.warning("⚠️ API key loaded but connection test failed")
                # Still return client, might work with actual requests
                return client
                
        except Exception as e:
            st.sidebar.error(f"❌ API Connection Failed: {str(e)}")
            return None
    return None

def get_system_prompt():
    """Create system prompt for business-focused chatbot"""
    return """You are a Business Analysis Assistant for OptimAIze products. Your role is to provide:
1. Non-technical business insights about our products
2. Comparative analysis between products
3. Target market recommendations
4. Revenue model explanations
5. Business impact analysis
6. KPI explanations and tracking suggestions
7. Competitive advantage positioning
8. Growth strategy recommendations

Rules:
- ALWAYS provide non-technical, business-focused explanations
- Use simple, clear language understandable by business executives
- Focus on ROI, business value, and strategic positioning
- When comparing products, highlight which is best for specific business needs
- Provide actionable business recommendations
- Never use technical jargon without business context
- If asked about technical details, redirect to business implications

Available Products Data:
{products_data}

Example Questions You Can Answer:
1. "Which product is best for reducing operational costs?"
2. "Compare all revenue models"
3. "Which product is best for healthcare industry?"
4. "What are the main business challenges?"
5. "How do KPIs differ across products?"
6. "Which product reduces costs the most?"
7. "Compare target markets for all products"
8. "What's the growth potential for each product?"

Remember: You are talking to business professionals, not engineers. Focus on business outcomes, financial impact, and strategic value.""".format(products_data=json.dumps(PRODUCT_DATA, indent=2))

def get_business_analysis(client, user_message, chat_history):
    """Get business analysis from OpenAI"""
    try:
        messages = [
            {"role": "system", "content": get_system_prompt()},
        ]
        
        # Add chat history
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change to "gpt-3.5-turbo" if needed
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in business analysis: {str(e)}"

def display_product_overview():
    """Display product cards with business information"""
    st.subheader("📊 OptimAIze Product Portfolio")
    
    cols = st.columns(2)
    for idx, product in enumerate(PRODUCT_DATA["products"]):
        with cols[idx % 2]:
            with st.expander(f"### {product['name']}", expanded=False):
                st.write(f"**Description:** {product['description']}")
                
                st.write("**🎯 Target Markets:**")
                st.info(product['business_context']['target_market'])
                
                st.write("**💰 Revenue Models:**")
                st.success(product['business_context']['revenue_model'])
                
                st.write("**📈 Business Impact:**")
                st.success(product['business_context']['business_impact'])
                
                st.write("**⚡ Key Features:**")
                for feature in product['features']:
                    st.write(f"• {feature}")
                
                st.write("**⚠️ Key Challenges:**")
                st.warning(product['business_context']['challenges'])

def display_business_metrics():
    """Display business metrics dashboard"""
    st.subheader("📈 Business Performance Dashboard")
    
    # Create metrics for each product
    for product in PRODUCT_DATA["products"]:
        with st.expander(f"{product['name']} - Key Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Target Markets", 
                         value=len(product['business_context']['target_market'].split(", ")))
            
            with col2:
                st.metric("💰 Revenue Streams", 
                         value=len(product['business_context']['revenue_model'].split(", ")))
            
            with col3:
                st.metric("📊 KPIs Tracked", 
                         value=len(product['business_context']['kpis']))
            
            # Display KPIs
            st.write("**📋 Key Performance Indicators:**")
            for kpi in product['business_context']['kpis']:
                st.write(f"• {kpi}")
            
            # Display Growth Metrics
            st.write("**📈 Growth Metrics:**")
            st.info(product['business_context']['growth_metrics'])
            
            # Display Competitive Advantage
            st.write("**🏆 Competitive Advantage:**")
            st.success(product['business_context']['competitive_advantage'])

def display_quick_comparison():
    """Display quick comparison table"""
    st.subheader("⚖️ Product Comparison")
    
    comparison_data = []
    for product in PRODUCT_DATA["products"]:
        comparison_data.append({
            "Product": product["name"],
            "Target Market": product["business_context"]["target_market"].split(", ")[0],
            "Key Impact": product["business_context"]["business_impact"].split(",")[0],
            "Revenue Model": product["business_context"]["revenue_model"].split(", ")[0],
            "Cost Reduction": extract_percentage(product["business_context"]["business_impact"])
        })
    
    st.table(comparison_data)

def extract_percentage(impact_text):
    """Extract percentage from business impact text"""
    import re
    percentages = re.findall(r'(\d+%)', impact_text)
    return percentages[0] if percentages else "N/A"

def create_unique_key(text):
    """Create a unique key from text using hash"""
    return hashlib.md5(text.encode()).hexdigest()[:8]

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="OptimAIze Business Analysis Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Display environment info
    env_status = "✅ .env file loaded" if os.path.exists('.env') else "⚠️ No .env file found"
    st.sidebar.info(env_status)
    
    # Initialize OpenAI client
    client = setup_openai()
    
    # Main title
    st.title("🤖 OptimAIze Business Analysis Chatbot")
    st.markdown("""
    Get business insights, comparative analysis, and strategic recommendations for OptimAIze products.
    All responses are **non-technical** and focused on **business value**.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Business Analysis Assistant. I can help you analyze OptimAIze products from a business perspective. Ask me about product comparisons, revenue models, target markets, or business impact!"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Quick Actions in Sidebar
    st.sidebar.subheader("🚀 Quick Actions")
    
    if st.sidebar.button("🔄 Clear Chat History", key="clear_chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared! How can I help you with business analysis today?"}
        ]
        st.rerun()
    
    # Suggested questions with unique keys
    st.sidebar.subheader("💡 Suggested Business Questions")
    suggested_questions = [
        "Which product has the highest ROI?",
        "Compare all revenue models",
        "Which product is best for healthcare industry?",
        "What are the main business challenges?",
        "How do KPIs differ across products?",
        "Which product reduces costs the most?",
        "Compare target markets for all products",
        "What's the growth potential for each product?"
    ]
    
    for question in suggested_questions:
        # Create a unique key for each button using the question text
        unique_key = f"btn_{create_unique_key(question)}"
        if st.sidebar.button(f"❓ {question}", key=unique_key):
            if client:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing business data..."):
                        response = get_business_analysis(client, question, st.session_state.messages)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.sidebar.error("Please configure OpenAI API key first")
    
    # Main content area
    if client:
        # Product Overview Section
        display_product_overview()
        
        # Quick Comparison
        display_quick_comparison()
        
        # Business Metrics Section
        display_business_metrics()
        
        # Chat input
        if prompt := st.chat_input("Ask about business insights, comparisons, or recommendations..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing business implications..."):
                    response = get_business_analysis(client, prompt, st.session_state.messages)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show product info even without API
        st.warning("⚠️ **OpenAI API key is required for the chatbot functionality**")
        st.info("Please configure your API key in the sidebar to enable business analysis and chat features.")
        
        # Still display static information
        display_product_overview()
        display_business_metrics()

if __name__ == "__main__":
    main()