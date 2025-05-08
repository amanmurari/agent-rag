import streamlit as st
from agent import MultiAgentSystem

st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("🧠 RAG-Powered Knowledge Assistant")

@st.cache_resource
def get_agent():
    return MultiAgentSystem()

agent = get_agent()

query = st.text_input("Ask a question:", placeholder="Try asking for definitions, calculations, or product details...")

if query:
    with st.spinner("Processing..."):
        result = agent.route_query(query)
        
        st.subheader("Decision Process")
        branch_used = result[0]
        st.write(f"Used {branch_used.upper()} branch")
        
        st.subheader("Answer")
        if branch_used == "rag":
            answer = result[1]
            context = result[2]
            st.write(answer)
            
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(context):
                    st.markdown(f"**Source {i+1}:** {doc.page_content}")
        else:
            answer = result[1]
            st.write(answer)