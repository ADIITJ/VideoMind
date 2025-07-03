import streamlit as st
import os
import psutil
from video_processing import VideoProcessor
from knowledge_graph import TemporalKnowledgeGraph
from query_processor import QueryProcessor

# Initialize components
video_processor = VideoProcessor()
knowledge_graph = TemporalKnowledgeGraph()
query_processor = QueryProcessor(knowledge_graph)

# Streamlit UI
st.title("Agent-Based Temporal Memory Graph")
st.write("Upload a video and query events or perform counterfactual reasoning.")

# Video upload
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
if uploaded_file:
    # Save video temporarily
    video_path = os.path.join("static", uploaded_file.name)
    os.makedirs("static", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Video uploaded successfully!")

    # Process video
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            try:
                descriptions = video_processor.process_video(video_path)
                for desc in descriptions:
                    # Simplified event extraction
                    subject = "person"  # Placeholder; enhance with NLP if needed
                    action = "interacts_with"
                    object_ = ", ".join(desc["objects"])
                    knowledge_graph.add_event(desc["timestamp"], subject, action, object_, desc["description"])
                st.success("Video processed and events stored in memory graph.")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

# Query input
query = st.text_input("Enter your query (e.g., 'What did the person do with the mug?' or 'What if the person didn’t take the mug?')")
if query:
    with st.spinner("Processing query..."):
        try:
            result = query_processor.process_query(query)
            st.write("**Result**:")
            st.write(result)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# System metrics
st.write("**System Metrics**")
cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().percent
st.write(f"CPU Usage: {cpu_usage}%")
st.write(f"RAM Usage: {ram_usage}%")