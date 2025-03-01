import os
import tempfile
from datetime import datetime
from typing import List
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import io
import time

import streamlit as st
import google.generativeai as genai
import bs4
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools

# Voice recording settings
SAMPLE_RATE = 44100
CHANNELS = 1
RECORDING_DURATION = 5  # seconds

def record_audio():
    """Record audio from microphone and return the audio data."""
    try:
        # Record audio
        recording = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16
        )
        sd.wait()
        return recording
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def save_audio_to_wav(audio_data):
    """Convert audio data to WAV format."""
    if audio_data is None:
        return None
    
    try:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buffer
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio file to text using speech recognition."""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Error with speech recognition service: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

class GeminiEmbedder(Embeddings):
    def __init__(self, model_name="models/text-embedding-004"):
        genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']


# Constants
COLLECTION_NAME = "gemini-thinking-agent-agno"


# Streamlit App Initialization
st.title("🤔 Agentic RAG with Gemini Thinking and Agno")

# Session State Initialization
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'url_list' not in st.session_state:
    st.session_state.url_list = []
if 'url_input' not in st.session_state:
    st.session_state.url_input = ""


# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key)
qdrant_url = st.sidebar.text_input("Qdrant URL", 
                                 placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                 value=st.session_state.qdrant_url)


# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Update session state
st.session_state.google_api_key = google_api_key
st.session_state.qdrant_api_key = qdrant_api_key
st.session_state.qdrant_url = qdrant_url

# Add in the sidebar configuration section, after the existing API inputs
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", 
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found"
    )
    st.session_state.exa_api_key = exa_api_key
    
    # Optional domain filtering
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", 
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Add this to the sidebar configuration section
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Lower values will return more documents but might be less relevant. Higher values are more strict."
)


# Utility Functions
def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


# Document Processing Functions
def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


# Vector Store Management
def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,  # Gemini embedding-004 dimension
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder()
        )
        
        # Add documents
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None


# Add this after the GeminiEmbedder class
def get_query_rewriter_agent() -> Agent:
    """Initialize a query rewriting agent."""
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions="""You are an expert at reformulating questions to be more precise and detailed. 
        Your task is to:
        1. Analyze the user's question
        2. Rewrite it to be more specific and search-friendly
        3. Expand any acronyms or technical terms
        4. Return ONLY the rewritten query without any additional text or explanations
        
        Example 1:
        User: "What does it say about FDA?"
        Output: "What are the key regulations, guidelines, and compliance requirements established by the U.S. Food and Drug Administration (FDA) for pharmaceutical products?"

        Example 2:
        User: "Tell me about 21 CFR"
        Output: "Explain the requirements and compliance standards outlined in Title 21 of the Code of Federal Regulations (CFR) for the manufacturing and marketing of food, drugs, and medical devices in the United States"
     
     """,
        show_tool_calls=False,
        markdown=True,
    )


def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-exp-1206"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query and prioritize FDA-related and FMI-related content
        2. Compile and summarize the most relevant information 
        3. Include sources in your response
    
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""You are an expert Intelligent Agent specializing in providing accurate, fact-based answers regarding FDA regulations and compliance. 

When answering questions, adhere strictly to the information provided by the user, either through uploaded documents or web sources they share. 

Your responsibilities are:

1. **When given context from uploaded documents**:
   - Focus exclusively on the information from the provided documents.
   - Be precise in your responses, citing specific details from the document.
   
2. **When given web search results**:
   - Clearly indicate that the information is derived from the web search.
   - Synthesize the information logically and ensure accuracy based on the provided web content.
   
3. **FDA Expertise**:
   - Provide answers based strictly on FDA documentation, guidelines, and regulations. Do not make assumptions, interpretations, or provide any information outside of FDA-compliant documentation.
   - Only cite facts that are clearly outlined in official FDA documents or trusted resources associated with the FDA.

4. **No Hallucination**:
   - Avoid generating or inferring any information that is not explicitly provided in the documents or web results.
   - Ensure that your responses are grounded in verifiable and factual data, especially when it comes to sensitive FDA-related matters.

5. **Accuracy and Clarity**:
   - Your responses must maintain the highest level of accuracy and clarity, especially when dealing with FDA compliance and regulatory matters. Any information not supported by documentation should not be included.

By adhering to these principles, you will ensure that your answers are precise, reliable, and in compliance with FDA standards. Always prioritize the integrity of the information provided by the user over general knowledge or external sources.

        """,
        show_tool_calls=True,
        markdown=True,
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """
    Check if documents in vector store are relevant to the query.
    
    Args:
        query: The search query
        vector_store: The vector store to search in
        threshold: Similarity threshold
        
    Returns:
        tuple[bool, List]: (has_relevant_docs, relevant_docs)
    """
    if not vector_store:
        return False, []
        
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs


# Main Application Flow
if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    genai.configure(api_key=st.session_state.google_api_key)
    
    qdrant_client = init_qdrant()
    
    # File/URL Upload Section
    st.sidebar.header("📁 Data Upload")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # URL Management Section
    st.sidebar.subheader("🌐 URL Management")
    
    # Initialize process_urls variable
    process_urls = False
    
    # Create a form for URL input
    with st.sidebar.form(key="url_form"):
        url_input = st.text_input("Enter URL", key="url_input")
        submit_url = st.form_submit_button("Add URL to List")
    
    # Process URL submission
    if submit_url and url_input:
        if url_input not in st.session_state.url_list and url_input not in st.session_state.processed_documents:
            st.session_state.url_list.append(url_input)
            st.rerun()
        else:
            st.sidebar.warning("URL already in list or processed!")

    # Process URLs button outside the form
    if st.session_state.url_list:
        process_urls = st.sidebar.button("Process All URLs")
    else:
        st.sidebar.info("Add URLs to process")

    # Display URLs to be processed
    if st.session_state.url_list:
        st.sidebar.markdown("**URLs to Process:**")
        for idx, url in enumerate(st.session_state.url_list):
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            with col1:
                st.text(f"{idx + 1}. {url}")
            with col2:
                if st.button("🗑️", key=f"remove_pending_{url}"):
                    st.session_state.url_list.remove(url)
                    st.rerun()

    # Process URLs when button is clicked
    if process_urls and st.session_state.url_list:
        with st.spinner('Processing URLs...'):
            for url in st.session_state.url_list[:]:  # Create a copy to iterate
                if url not in st.session_state.processed_documents:
                    texts = process_web(url)
                    if texts and qdrant_client:
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(texts)
                        else:
                            st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                        st.session_state.processed_documents.append(url)
                        st.success(f"✅ Added URL: {url}")
                        st.session_state.url_list.remove(url)
            if not st.session_state.url_list:
                st.success("✅ All URLs processed successfully!")
                st.rerun()

    # Process documents
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name not in st.session_state.processed_documents:
                with st.spinner(f'Processing PDF: {file_name}...'):
                    texts = process_pdf(uploaded_file)
                    if texts and qdrant_client:
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(texts)
                        else:
                            st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                        st.session_state.processed_documents.append(file_name)
                        st.success(f"✅ Added PDF: {file_name}")

    # Display processed sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        
        # Add a clear documents button
        if st.sidebar.button("🗑️ Clear All Documents"):
            if qdrant_client:
                try:
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    st.session_state.vector_store = None
                    st.session_state.processed_documents = []
                    st.success("✅ All documents cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing documents: {str(e)}")
        
        # Group documents by type
        pdfs = [doc for doc in st.session_state.processed_documents if doc.endswith('.pdf')]
        urls = [doc for doc in st.session_state.processed_documents if not doc.endswith('.pdf')]
        
        if pdfs:
            st.sidebar.subheader("📄 PDF Documents")
            for pdf in pdfs:
                col1, col2 = st.sidebar.columns([0.8, 0.2])
                with col1:
                    st.text(pdf)
                with col2:
                    if st.button("🗑️", key=f"delete_{pdf}"):
                        st.session_state.processed_documents.remove(pdf)
                        st.rerun()
        
        if urls:
            st.sidebar.subheader("🌐 Web Sources")
            for url in urls:
                col1, col2 = st.sidebar.columns([0.8, 0.2])
                with col1:
                    st.text(url)
                with col2:
                    if st.button("🗑️", key=f"delete_{url}"):
                        st.session_state.processed_documents.remove(url)
                        st.rerun()

    # Chat Interface
    # Create main chat container
    chat_container = st.container()
    
    # Voice recording modal and feedback
    voice_modal = st.empty()
    
    # Create three columns for chat input, voice input, and search toggle
    chat_col, voice_col, toggle_col = st.columns([0.8, 0.1, 0.1])

    with chat_col:
        prompt = st.chat_input("Ask about your documents...")

    with voice_col:
        voice_button = st.button("🎤", help=f"Click to record voice input (Duration: {RECORDING_DURATION} seconds)")
        
        if voice_button:
            # Use the full width for recording feedback
            with voice_modal.container():
                st.markdown("### 🎙️ Recording in Progress")
                st.info(f"Please speak now. Recording will stop automatically after {RECORDING_DURATION} seconds.")
                
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    progress_bar = st.progress(0)
                with col2:
                    countdown = st.empty()
                
                # Record audio with visual feedback
                for i in range(RECORDING_DURATION):
                    if i == 0:  # Only record once at the start
                        audio_data = record_audio()
                    progress = (i + 1) / RECORDING_DURATION
                    progress_bar.progress(progress)
                    countdown.markdown(f"**Time left:** {RECORDING_DURATION - i} seconds")
                    time.sleep(1)
                
                if audio_data is not None:
                    # Convert audio to WAV format
                    wav_buffer = save_audio_to_wav(audio_data)
                    if wav_buffer:
                        # Reset buffer position
                        wav_buffer.seek(0)
                        # Transcribe audio to text
                        with st.spinner("Transcribing your message..."):
                            transcribed_text = transcribe_audio(wav_buffer)
                            if transcribed_text:
                                prompt = transcribed_text
                                st.success("✅ Successfully transcribed!")
                                st.info(f"Your message: {transcribed_text}")
                            else:
                                st.error("❌ Could not transcribe the audio. Please try again.")
                    else:
                        st.error("❌ Error processing audio. Please try again.")
                else:
                    st.error("❌ Error recording audio. Please check your microphone and try again.")
            
            # Clear the modal after processing
            time.sleep(2)
            voice_modal.empty()

    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

    if prompt:
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Step 1: Rewrite the query for better retrieval
        with st.spinner("🤔 Reformulating query..."):
            try:
                query_rewriter = get_query_rewriter_agent()
                rewritten_query = query_rewriter.run(prompt).content
                
                with st.expander("🔄 See rewritten query"):
                    st.write(f"Original: {prompt}")
                    st.write(f"Rewritten: {rewritten_query}")
            except Exception as e:
                st.error(f"❌ Error rewriting query: {str(e)}")
                rewritten_query = prompt

        # Step 2: Choose search strategy based on force_web_search toggle
        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # Try document search first
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5, 
                    "score_threshold": st.session_state.similarity_threshold
                }
            )
            docs = retriever.invoke(rewritten_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found in database, falling back to web search...")

        # Step 3: Use web search if:
        # 1. Web search is forced ON via toggle, or
        # 2. No relevant documents found AND web search is enabled in settings
        if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🔍 Searching the web..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else:
                            st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        # Step 4: Generate response using the RAG agent
        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                
                if context:
                    full_prompt = f"""Context: {context}

Original Question: {prompt}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                    st.info("ℹ️ No relevant information found in documents or web search.")

                response = rag_agent.run(full_prompt)
                
                # Add assistant response to history
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response.content)
                    
                    # Show sources if available
                    if not st.session_state.force_web_search and 'docs' in locals() and docs:
                        with st.expander("🔍 See document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                st.write(f"{source_icon} Source {i} from {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

else:
    st.warning("⚠️ Please enter your Google API Key to continue")