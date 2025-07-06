import os
import sys
# ensure project root is on PYTHONPATH
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

print("Starting Streamlit app...")

import streamlit as st
import numpy as np
import faiss
from PIL import Image
import torch
from src import extract, detect_cluster, caption, summarization, utils
from src.graph_context import build_temporal_graph, gather_context
from src.query_parser import parse_query, ground_event
from src.llm_reasoning import ask_with_chunks
from open_clip import create_model_and_transforms

st.set_page_config(layout='wide', page_title='VideoMind')
st.title('🎥 VideoMind — Video QA & Counterfactual')

# Upload
video = st.file_uploader('Upload Video File', type=['mp4', 'mov'])
if video:
    print("Video uploaded.")
    os.makedirs('data/frames', exist_ok=True)

    # Save uploaded file with correct extension
    video_ext = os.path.splitext(video.name)[-1]
    vp = f'data/input{video_ext}'
    st.success(f'Uploaded {video.name}')

    with open(vp, 'wb') as f:
        f.write(video.read())
    print(f"Saved video to {vp}")

    # Extract frames
    print("Extracting frames...")
    n = extract.extract_frames(vp, 'data/frames', fps=1)
    print(f"Extracted {n} frames.")
    st.success(f'Extracted {n} frames at 1 FPS.')

    # Prepare frame paths
    frames = sorted(os.listdir('data/frames'))
    print(f"Frame files: {frames}")
    paths = [os.path.join('data/frames', f) for f in frames]

    # Compute CLIP embeddings
    print("Loading CLIP model...")
    model_clip, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model_clip = model_clip.to('cpu')
    embs = []
    print("Computing CLIP embeddings...")
    for i, p in enumerate(paths):
        print(f"Processing frame {i}: {p}")
        img = preprocess(Image.open(p)).unsqueeze(0).to('cpu')
        with torch.no_grad():
            e = model_clip.encode_image(img).cpu().numpy().flatten()
        embs.append(e)
    embs = np.stack(embs)
    print("CLIP embeddings computed.")

    # Hierarchical clustering
    print("Clustering embeddings...")
    clusters = detect_cluster.hierarchical_cluster(embs, paths)
    print(f"Clusters: {clusters.keys()}")

    # Caption all frames
    print("Captioning frames...")
    caps = {}
    for i, p in enumerate(paths):
        print(f"Captioning frame {i}: {p}")
        caps[p] = caption.caption_frame(p)
    print("All frames captioned.")

    # Summaries
    print("Summarizing by time...")
    time_summ = summarization.summarize_by_time(caps)
    print("Summarizing by object...")
    obj_summ = summarization.summarize_by_object(caps)

    # Build graph
    print("Building temporal graph...")
    nodes = [(i, caps[p]) for i,p in enumerate(paths)]
    G = build_temporal_graph(nodes)
    print("Temporal graph built.")

    # Sidebar summaries
    st.sidebar.header('Summaries')
    st.sidebar.subheader('By Time Window')
    for t, text in time_summ.items():
        st.sidebar.write(f'{t}: {text}')
    st.sidebar.subheader('By Object')
    for obj, text in obj_summ.items():
        st.sidebar.write(f'{obj}: {text}')

    # Query input
    q = st.text_input('Ask about this video (timestamp or event)')
    if q:
        print(f"Received query: {q}")
        text, tsec = parse_query(q)
        print(f"Parsed query: text={text}, tsec={tsec}")
        if tsec is not None:
            # nearest frame index by time (assuming 1 FPS)
            idx = min(range(len(paths)), key=lambda i: abs(i - tsec))
            print(f"Using timestamp, selected frame idx: {idx}")
        else:
            # ground via semantic similarity
            eps = [{'caption': caps[p]} for p in paths]
            idx = ground_event(text, eps)
            print(f"Using semantic grounding, selected frame idx: {idx}")

        # gather context window
        print("Gathering context window...")
        ctx = gather_context(G, idx, before=5, after=5)
        print(f"Context window: {ctx}")
        # split into chunks
        half = len(ctx) // 2 or len(ctx)
        chunk1 = "\n".join(f"{i}: {cap}" for i,_,cap in ctx[:half])
        chunk2 = "\n".join(f"{i}: {cap}" for i,_,cap in ctx[half:])
        print("Sending chunks to LLM...")
        answer = ask_with_chunks([chunk1, chunk2], text)
        print("LLM answer received.")

        st.subheader('💬 Answer')
        st.write(answer)
        st.image(utils.show_image(paths[idx]), width=400, caption=f'Frame {idx}')
