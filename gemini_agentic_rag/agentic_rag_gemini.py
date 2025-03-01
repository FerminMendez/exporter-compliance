import os
import tempfile
from datetime import datetime
from typing import List
import speech_recognition as sr
import time
import numpy as np
import json
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# Voice recording settings
RECORDING_DURATION = 10  # seconds

def detect_silence(audio_data, threshold):
    """Detect if the audio segment is silence."""
    return np.mean(np.abs(audio_data)) < threshold

def record_and_transcribe():
    """Record audio from microphone and transcribe it directly."""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Adjusting for ambient noise... Please wait.")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Create placeholders for visual feedback
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.info("🎙️ Ready! Start speaking when you see the timer...")
            time.sleep(1)  # Give user a moment to prepare
            
            # Start recording with visual feedback
            start_time = time.time()
            status_placeholder.info("🎙️ Recording... Speak now!")
            
            try:
                # Show progress bar during recording
                while (time.time() - start_time) < RECORDING_DURATION:
                    elapsed = time.time() - start_time
                    remaining = RECORDING_DURATION - elapsed
                    progress = int((elapsed / RECORDING_DURATION) * 100)
                    progress_placeholder.progress(progress)
                    status_placeholder.info(f"🎙️ Recording... {remaining:.1f} seconds remaining")
                    time.sleep(0.1)
                
                # Record audio
                audio = recognizer.listen(source, timeout=RECORDING_DURATION, phrase_time_limit=RECORDING_DURATION)
                status_placeholder.success("✅ Recording complete! Processing your speech...")
                
                # Transcribe audio using Google Speech Recognition
                text = recognizer.recognize_google(audio)
                return text
                
            except sr.WaitTimeoutError:
                status_placeholder.warning("No speech detected.")
                return None
                
    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from speech recognition service: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
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

# Load initial URLs from environment
try:
    INITIAL_URLS = json.loads(os.getenv('INITIAL_URLS', '[]'))
except json.JSONDecodeError:
    INITIAL_URLS = [
        "https://www.fda.gov/media/163132/download",
        "https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods",
        "https://www.fmi.org/food-safety/traceability-rule"
    ]


# Streamlit App Initialization
st.title("🤔 FDA Agent. RAG with Gemini")

# Session State Initialization with environment variables
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.getenv('GOOGLE_API_KEY', '')
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = os.getenv('QDRANT_API_KEY', '')
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = os.getenv('QDRANT_URL', '')
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = os.getenv('EXA_API_KEY', '')
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = os.getenv('USE_WEB_SEARCH', 'false').lower() == 'true'
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
if 'url_list' not in st.session_state:
    st.session_state.url_list = INITIAL_URLS.copy()
if 'url_input' not in st.session_state:
    st.session_state.url_input = ""
if 'initial_urls_processed' not in st.session_state:
    st.session_state.initial_urls_processed = False


# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
google_api_key = st.sidebar.text_input(
    "Google API Key", 
    type="password",
    value=st.session_state.google_api_key,
    help="Enter your Google API Key or set it in the .env file"
)
qdrant_api_key = st.sidebar.text_input(
    "Qdrant API Key",
    type="password",
    value=st.session_state.qdrant_api_key,
    help="Enter your Qdrant API Key or set it in the .env file"
)
qdrant_url = st.sidebar.text_input(
    "Qdrant URL", 
    placeholder="https://your-cluster.cloud.qdrant.io:6333",
    value=st.session_state.qdrant_url,
    help="Enter your Qdrant URL or set it in the .env file"
)


# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Update session state
st.session_state.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY', '')
st.session_state.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY', '')
st.session_state.qdrant_url = qdrant_url or os.getenv('QDRANT_URL', '')

# Add in the sidebar configuration section, after the existing API inputs
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox(
    "Enable Web Search Fallback",
    value=st.session_state.use_web_search,
    help="Enable web search fallback or set it in the .env file"
)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", 
        type="password",
        value=st.session_state.exa_api_key,
        help="Enter your Exa AI API Key or set it in the .env file"
    )
    st.session_state.exa_api_key = exa_api_key or os.getenv('EXA_API_KEY', '')
    
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
    value=float(os.getenv('SIMILARITY_THRESHOLD', '0.7')),
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
        instructions="""You are an expert at reformulating questions about regulatory compliance and traceability requirements. 
        Your task is to:
        1. Analyze the user's question in the context of FDA compliance and traceability requirements
        2. Rewrite it to be more specific and search-friendly, focusing on regulatory aspects
        3. Expand any acronyms, technical terms, or regulatory references
        4. Consider the compliance context and any relevant regulatory frameworks
        5. Return ONLY the rewritten query without any additional text or explanations
        
        Example 1:
        User: "What does it say about FDA?"
        Output: "What are the key regulations, guidelines, and compliance requirements established by the U.S. Food and Drug Administration (FDA) for food traceability and supply chain documentation?"

        Example 2:
        User: "Tell me about 21 CFR"
        Output: "Explain the specific requirements and compliance standards outlined in Title 21 of the Code of Federal Regulations (CFR) regarding food traceability, record-keeping requirements, and supply chain documentation in the United States"
     
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
        instructions="""You are a compliance and regulatory web search expert. Your task is to:
        1. Search the web for relevant information about FDA compliance and traceability requirements
        2. Prioritize official FDA documentation, regulatory guidelines, and authoritative industry sources
        3. Focus on finding the most recent and applicable regulatory information
        4. Compile and summarize the relevant information with emphasis on compliance requirements
        5. Include citations and references to specific regulations or guidance documents
        6. Note effective dates, compliance deadlines, and any upcoming regulatory changes
        7. Include sources in your response, particularly official regulatory sources
        8. Highlight any critical compliance requirements or potential risks
    
        Always maintain a compliance-focused perspective and ensure information is current and authoritative.
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""You are an expert Compliance Assistant specializing in FDA regulations and traceability requirements. 
Your primary focus is helping users understand and implement compliance requirements accurately.

When answering questions, adhere to these principles:

1. **Source Hierarchy and References**:
   - ALWAYS prioritize information from user-provided documents and URLs
   - When citing information, use this priority order:
     1. User-uploaded PDFs and documents
     2. User-provided URLs that have been processed
     3. Official FDA/regulatory websites from verified web searches
   - For each reference:
     * Include specific document names or URLs from the user's uploaded content
     * Quote relevant sections with page numbers or section identifiers
     * If referencing external sources, verify they are from the processed URL list

2. **Document-Based Responses**:
   - Focus exclusively on information from provided documents and official sources
   - Cite specific sections using clear identifiers (e.g., "From [document name], Section 3.2:")
   - When quoting, include document source and location (e.g., "As stated in [user-provided PDF], page 5:")
   
3. **Web Search Integration**:
   - Only cite web sources that have been successfully processed and added to the knowledge base
   - When using web search results, explicitly state: "According to [processed URL]:"
   - If information would require an unverified external source, instead suggest: "You may want to upload additional documentation about [topic] for me to provide accurate information."

4. **Compliance Expertise**:
   - Focus on practical implementation of compliance requirements
   - Break down complex regulatory requirements into actionable steps
   - Highlight critical compliance points and common pitfalls
   - Emphasize record-keeping and documentation requirements
   
5. **Risk Management**:
   - Identify potential compliance risks in your responses
   - Suggest preventive measures and best practices
   - Note any exceptions or special circumstances
   
6. **Clarity and Accuracy**:
   - Use precise regulatory language when appropriate
   - Define technical terms and acronyms
   - Structure responses to clearly separate requirements from recommendations
   
7. **Follow-up Questions**:
   - Always end your response with 1-2 relevant follow-up questions
   - Focus follow-ups on:
     * Clarifying compliance requirements
     * Implementation details
     * Specific aspects of the user's situation
     * Risk assessment and mitigation
     * Documentation and record-keeping needs
     * Suggesting additional relevant documentation to upload if needed

8. **Documentation**:
   - Emphasize the importance of proper documentation
   - Reference specific forms, templates, or record-keeping requirements
   - Note any required retention periods for documents

9. **Information Gaps**:
   - If a question requires information not available in provided documents:
     * Clearly state that the information is not in current documents
     * Suggest specific types of documents that would help answer the question
     * Offer to analyze additional documents if provided
   - Never make assumptions about requirements not explicitly covered in available documents

Remember to:
- Only cite sources that are verifiably in the user's uploaded content or processed URLs
- If information would require external sources, ask the user to provide relevant documentation
- Clearly distinguish between information from provided documents vs. general knowledge
- Include relevant follow-up questions in every response
- When suggesting additional documents, be specific about what types would be helpful

Example follow-up questions:
- "Would you like me to provide specific citations from [document name] regarding these requirements?"
- "Should we review the documentation requirements outlined in [specific uploaded document]?"
- "Would it be helpful to upload additional documentation about [specific aspect] to provide more detailed guidance?"
- "Which specific aspect of [mentioned requirement from document] would you like me to elaborate on?"
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
    
    # Process initial URLs if not done yet
    if not st.session_state.initial_urls_processed and qdrant_client:
        with st.spinner('Processing initial URLs...'):
            for url in INITIAL_URLS:
                if url not in st.session_state.processed_documents:
                    texts = process_web(url)
                    if texts and qdrant_client:
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(texts)
                        else:
                            st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                        st.session_state.processed_documents.append(url)
                        if url in st.session_state.url_list:
                            st.session_state.url_list.remove(url)
            st.session_state.initial_urls_processed = True
            st.success("✅ Initial URLs processed successfully!")
            
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
        voice_button = st.button("🎤", help=f"Click to record voice input. You will have {RECORDING_DURATION} seconds to speak.")
        
        if voice_button:
            # Use the full width for recording feedback
            with voice_modal.container():
                st.markdown("### 🎙️ Voice Input")
                # Record and transcribe
                transcribed_text = record_and_transcribe()
                
                if transcribed_text:
                    prompt = transcribed_text
                    st.success("✅ Successfully transcribed!")
                    st.info(f"Your message: {transcribed_text}")
            
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