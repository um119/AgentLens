"""
AgentLens - Streamlit Web Application
AI-Powered LLM Discovery Assistant for Agentic AI Workflows
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import config
from agent_core import search_llms_for_workflow
import ollama_utils

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables"""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = []
    if "local_models" not in st.session_state:
        st.session_state.local_models = []


def display_header():
    """Display the application header"""
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown(f"*{config.TAGLINE}*")
    st.markdown("---")


def display_sidebar():
    """Display sidebar with system information"""
    with st.sidebar:
        st.header("System Status")

        # Web Search Status
        st.subheader("🌐 Web Search")
        try:
            from web_search import WebSearch
            test_search = WebSearch()
            results = test_search.search_llms("test", max_results=1)
            st.success("✅ Web Search Active")
        except Exception as e:
            st.warning(f"⚠️ Web Search Limited")

        st.markdown("---")

        # Ollama model status
        model = config.OLLAMA_MODEL
        st.subheader("🤖 AI Model")
        st.success(f"Model: {model}")

        # Ollama service status
        if ollama_utils.is_ollama_available():
            st.success("✅ Ollama Running")
            try:
                models = ollama_utils.OllamaManager().list_models()
                st.session_state.local_models = models
                st.markdown(f"**{len(models)} local models**")
            except:
                st.markdown("Unable to list models")
        else:
            st.error("❌ Ollama Not Running")
            st.markdown("""
            **To run:**
            1. Install [Ollama](https://ollama.ai/)
            2. Run `ollama serve`
            3. `ollama pull qwen3.5:cloud`
            """)

        st.markdown("---")

        # Search history
        st.subheader("History")
        if st.session_state.search_history:
            for i, query in enumerate(reversed(st.session_state.search_history[-5:])):
                st.markdown(f"- {query[:40]}...")
        else:
            st.markdown("*No searches yet*")

        st.markdown("---")

        # How it works
        st.subheader("How It Works")
        st.markdown("""
        1. Describe your workflow
        2. Click **Search LLMs**
        3. Web searches latest LLM info
        4. AI recommends best models
        5. Compare & choose
        """)


def display_query_input() -> str:
    """Display query input and return the query"""
    st.subheader("🔍 Describe Your Workflow")

    # Example queries
    example = st.selectbox(
        "Try an example:",
        ["", *config.EXAMPLE_QUERIES],
        label_visibility="collapsed"
    )

    # Text input
    query = st.text_area(
        "Describe your agentic AI workflow:",
        value=example,
        placeholder="I'm building a marketing automation agent that creates campaigns, targets audiences, and generates reports.",
        height=100,
        label_visibility="collapsed"
    )

    # Search button
    col1, col2 = st.columns([1, 5])
    with col1:
        search_clicked = st.button("🔎 Search LLMs", type="primary", use_container_width=True)

    return query if search_clicked else ""


def display_recommendations(results: List[Dict[str, Any]]):
    """Display LLM recommendations in card format"""
    st.subheader("📊 Recommended LLMs")

    if not results:
        st.info("No recommendations found. Try a different query.")
        return

    for i, llm in enumerate(results):
        with st.expander(f"**{i+1}. {llm.get('name', 'Unknown')}**", expanded=True):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Description:**")
                st.write(llm.get("description", "No description"))

            with col2:
                st.markdown("**Key Details:**")

                params = llm.get("parameters", "Unknown")
                st.markdown(f"- **Parameters:** {params}")

                features = llm.get("key_features", "Not specified")
                st.markdown(f"- **Key Features:** {features}")

                tool_support = llm.get("tool_calling_support", "Unknown")
                st.markdown(f"- **Tool Calling:** {tool_support}")


def display_local_models(workflow_query: str):
    """Display local Ollama models section"""
    st.subheader("💻 Local Models (Ollama)")

    manager = ollama_utils.OllamaManager()

    if not manager.is_ollama_running():
        st.info("Ollama is not running. Install and start Ollama to see local models.")
        return

    models = manager.list_models()

    if not models:
        st.info("No local models found. Pull models with `ollama pull <model>`")
        return

    # Get suitable models for the workflow
    suitable = []
    if workflow_query:
        suitable = manager.get_suitable_models(workflow_query)

    # Display models
    cols = st.columns(min(len(models), 3))
    for i, model in enumerate(models):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"**{model['name']}**")
                st.markdown(f"Size: {manager.format_model_size(model['size'])}")

                # Show suitability if we have a query
                if suitable:
                    suit = next((s for s in suitable if s['name'] == model['name']), None)
                    if suit:
                        suitability = suit.get('suitability', 'Unknown')
                        if suitability == 'Excellent':
                            st.success(f"✓ {suitability}")
                        elif suitability == 'Good' or suitability.startswith('Good'):
                            st.info(f"✓ {suitability}")
                        else:
                            st.markdown(f"▫ {suitability}")

    st.markdown("---")

    # Model testing section
    with st.expander("🧪 Test a Local Model"):
        test_model = st.selectbox("Select model:", [m['name'] for m in models])
        test_prompt = st.text_input("Test prompt:", "What is the capital of France?")
        if st.button("Run Test"):
            with st.spinner("Running model..."):
                response = manager.test_model(test_model, test_prompt)
                if response:
                    st.markdown("**Response:**")
                    st.write(response)
                else:
                    st.error("Model test failed")


def display_comparison_table(results: List[Dict[str, Any]]):
    """Display comparison table for recommended LLMs"""
    st.subheader("📋 Comparison Table")

    if not results:
        return

    # Prepare data for table
    table_data = []
    for llm in results:
        table_data.append({
            "Name": llm.get("name", "Unknown"),
            "Provider": "Cloud",  # All cloud models from search
            "Parameters": llm.get("parameters", "Unknown"),
            "Tool Support": "Yes" if "Yes" in str(llm.get("tool_calling_support", "")) else "Check Details",
            "Cost Tier": "API (paid)",
            "Suitability": "High",  # Default since these are recommendations
        })

    # Also add local models
    if st.session_state.local_models:
        manager = ollama_utils.OllamaManager()
        for model in st.session_state.local_models:
            table_data.append({
                "Name": model['name'],
                "Provider": "Local (Ollama)",
                "Parameters": manager._extract_parameters(model['name']),
                "Tool Support": "Via fine-tuning",
                "Cost Tier": "Free",
                "Suitability": "Medium",
            })

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def main():
    """Main application entry point"""
    initialize_session_state()

    display_header()
    display_sidebar()

    # Query input
    query = display_query_input()

    if query:
        # Add to search history
        st.session_state.search_history.append(query)

        # Search for LLMs
        with st.spinner("Searching for best LLMs..."):
            results = search_llms_for_workflow(query)
            st.session_state.current_results = results

        st.markdown("---")

        # Display results
        display_recommendations(results)

        # Display comparison table
        display_comparison_table(results)

        # Display local models
        st.markdown("---")
        display_local_models(query)

    else:
        # Show placeholder content when no query
        st.markdown("""
        ### Getting Started
        Describe your agentic AI workflow above to get personalized LLM recommendations.

        The system will search the web for the latest information and recommend
        models based on:
        - Tool/Function calling support
        - Reasoning capabilities
        - Context window size
        - Cost effectiveness

        ### Example Use Cases
        - Marketing automation agents
        - Customer support workflows
        - Coding/development assistants
        - Research and analysis agents
        """)


if __name__ == "__main__":
    main()