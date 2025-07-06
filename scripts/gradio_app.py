import os
import sys
import numpy as np
from PIL import Image
import torch
import gradio as gr
from open_clip import create_model_and_transforms
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from src import extract, detect_cluster, caption, summarization, utils
from src.graph_context import build_temporal_graph, gather_context
from src.query_parser import parse_query, ground_event
from src.llm_reasoning import summarize_time_llm, summarize_object_llm
import ollama

# Add project root to sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Extended counterfactual phrases with basic NLP enhancement
CF_PHRASES = [
    "what if", "suppose", "imagine", "if instead", "had happened", "if only", "if", "if we",
    "if I had", "if we had", "instead of", "rather than", "alternatively", "in a different scenario",
    "hypothetically", "let's say", "assuming that", "could have", "would have", "were to", "had been"
]

def is_counterfactual(query):
    """Determine if a query is counterfactual using phrases and basic verb tense checks."""
    ql = query.lower()
    has_cf_phrase = any(phrase in ql for phrase in CF_PHRASES)
    # Basic check for past tense or conditional verbs (e.g., "had", "would")
    has_conditional = any(word in ql for word in ["had", "would", "could", "were"])
    return has_cf_phrase or has_conditional

def find_best_object(query, obj_summ):
    """Find the best matching object in the query using substring matching."""
    best_obj, best_len = None, 0
    q_lower = query.lower()
    for obj in obj_summ.keys():
        candidate = obj.lower()
        if candidate in q_lower and len(candidate) > best_len:
            best_obj, best_len = obj, len(candidate)
    return best_obj

def get_nearest_time_summary(idx, paths, time_summ):
    """Get the nearest time summary based on frame index."""
    best_key, best_diff = None, float('inf')
    for key in time_summ.keys():
        if key in paths:
            pseudo_idx = paths.index(key)
            diff = abs(pseudo_idx - idx)
            if diff < best_diff:
                best_diff, best_key = diff, key
    return time_summ.get(best_key, "")

def get_whole_video_summary(caps):
    """Generate a summary of the entire video."""
    return summarize_time_llm(list(caps.values()))

def process_video(video_path):
    """Process the uploaded video and return state and status."""
    if not video_path or not os.path.isfile(video_path):
        return None, "Error: Invalid video file."
    
    try:
        os.makedirs('data/frames', exist_ok=True)
        # Clear previous frames
        for f in os.listdir('data/frames'):
            os.remove(os.path.join('data/frames', f))
        
        # Extract frames
        n = extract.extract_frames(video_path, 'data/frames', fps=1)
        if n == 0:
            return None, "Error: No frames extracted from video."
        
        frames = sorted(os.listdir('data/frames'))
        paths = [os.path.join('data/frames', f) for f in frames]
        
        # Load CLIP model
        model_clip, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model_clip = model_clip.to('cpu')
        embs = []
        for p in paths:
            img = preprocess(Image.open(p)).unsqueeze(0).to('cpu')
            with torch.no_grad():
                e = model_clip.encode_image(img).cpu().numpy().flatten()
            embs.append(e)
        embs = np.stack(embs)
        
        # Cluster embeddings
        clusters = detect_cluster.hierarchical_cluster(embs, paths)
        
        # Caption frames
        caps = {p: caption.caption_frame(p) for p in paths}
        
        # Summarize
        time_summ = summarization.summarize_by_time(caps)
        obj_summ = summarization.summarize_by_object(caps)
        
        # Build temporal graph
        nodes = [(i, caps[p]) for i, p in enumerate(paths)]
        G = build_temporal_graph(nodes)
        
        state = {
            'paths': paths,
            'caps': caps,
            'time_summ': time_summ,
            'obj_summ': obj_summ,
            'G': G
        }
        return state, f"Video processed successfully. {n} frames analyzed."
    except Exception as e:
        return None, f"Error during processing: {str(e)}"

def ask_question(question, state):
    """Answer a user's question and return the answer with a frame reference."""
    if not state:
        return "Please process a video first.", None

    paths = state['paths']
    caps = state['caps']
    time_summ = state['time_summ']
    obj_summ = state['obj_summ']
    G = state['G']

    if not question.strip():
        return "Please enter a valid question.", None

    try:
        # Parse query
        text, tsec = parse_query(question)
        if tsec is not None:
            idx = min(range(len(paths)), key=lambda i: abs(i - tsec))
        else:
            eps = [{'caption': caps[p]} for p in paths]
            idx = ground_event(text, eps)
        
        # Gather context
        time_summary = get_nearest_time_summary(idx, paths, time_summ)
        best_obj = find_best_object(question, obj_summ)
        object_summary = obj_summ.get(best_obj, "") if best_obj else ""
        ctx = gather_context(G, idx, before=5, after=5)
        ctx_captions = [cap for i, _, cap in ctx]
        
        # Determine question type and compose LLM prompt
        if is_counterfactual(question):
            whole_summary = get_whole_video_summary(caps)
            all_obj_summaries = "\n".join([f"{obj}: {summ}" for obj, summ in obj_summ.items()])
            context_until_change = [cap for i, _, cap in ctx if i <= idx]
            llm_context = (
                "Whole video summary:\n" + whole_summary + "\n\n"
                "Object summaries:\n" + all_obj_summaries + "\n\n"
                "Context up to change:\n" + "\n".join(context_until_change)
            )
            system_prompt = (
                "You are an expert video reasoning assistant. Answer this counterfactual question "
                "based strictly on the provided whole video summary, object summaries, and context "
                "up to the point of change. Do not invent new objects or events. Be concise and logical."
            )
            user_prompt = f"{llm_context}\n\nCounterfactual Question:\n{question}\n\nAnswer:"
            timestamp = f"(based on scenario at {idx} seconds)"
        else:
            llm_context = f"Time summary: {time_summary}\n"
            if object_summary:
                llm_context += f"Object summary ({best_obj}): {object_summary}\n"
            llm_context += "Context captions:\n" + "\n".join(ctx_captions)
            system_prompt = (
                "You are an expert video reasoning assistant. Answer this question based strictly "
                "on the provided time summary, object summary (if any), and context captions. "
                "Do not invent new events. Keep your answer concise and factual."
            )
            user_prompt = f"{llm_context}\n\nQuestion:\n{question}\n\nAnswer:"
            timestamp = f"(referring to frame at {idx} seconds)"
        
        # Query LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        resp = ollama.chat(model="mistral:latest", messages=messages)
        answer = resp.get('message', {}).get('content', '').strip()

        # Only return the frame path (not a tuple with answer) to Gradio's image/video component
        return answer + f"\n\n{timestamp}", paths[idx]
    except Exception as e:
        return f"Error answering question: {str(e)}", None

def send_message(message, state, chat_history):
    """Handle user message and update chat history."""
    answer, frame_path = ask_question(message, state)
    # Only append the answer (string) to the chat history, not a tuple
    chat_history.append((message, answer))
    return chat_history, ""

# Build the Gradio interface
with gr.Blocks(title="Video Question Answering") as app:
    gr.Markdown(
        """
        # VideoMind: Video Question Answering App
        Upload a video, click "Process Video", then ask questions about what happened or explore "what if" scenarios.
        - Normal questions (e.g., "What happened at 5 seconds?") show the relevant frame.
        - Counterfactual questions (e.g., "What if the car turned left?") reason hypothetically.
        """
    )
    
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        with gr.Column():
            process_button = gr.Button("Process Video")
            status_label = gr.Label("Video not processed")
    
    chatbot = gr.Chatbot(label="Conversation")
    message_input = gr.Textbox(
        label="Your Question",
        placeholder="E.g., 'What happened at 10 seconds?' or 'What if the dog ran away?'"
    )
    send_button = gr.Button("Send")
    
    state = gr.State()
    
    process_button.click(
        fn=process_video,
        inputs=video_input,
        outputs=[state, status_label]
    )
    
    send_button.click(
        fn=send_message,
        inputs=[message_input, state, chatbot],
        outputs=[chatbot, message_input]
    )
    
    video_input.change(
        fn=lambda: (None, "Video not processed"),
        outputs=[state, status_label]
    )

app.launch()