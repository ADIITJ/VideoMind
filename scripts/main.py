import os
import sys
# ensure project root is on PYTHONPATH
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

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

st.set_page_config(layout='wide', page_title='VidMemoryNet')
st.title('🎥 VidMemoryNet — Video QA & Counterfactual')

# Upload
video = st.file_uploader('Upload MP4 video', type=['mp4'])
if video:
    os.makedirs('data/frames', exist_ok=True)
    vp = 'data/input.mp4'
    with open(vp, 'wb') as f:
        f.write(video.read())
    # Extract frames
    n = extract.extract_frames(vp, 'data/frames', fps=1)
    st.success(f'Extracted {n} frames at 1 FPS.')

    # Prepare frame paths
    frames = sorted(os.listdir('data/frames'))
    paths = [os.path.join('data/frames', f) for f in frames]

    # Compute CLIP embeddings
    model_clip, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model_clip = model_clip.to('mps')
    embs = []
    for p in paths:
        img = preprocess(Image.open(p)).unsqueeze(0).to('mps')
        with torch.no_grad():
            e = model_clip.encode_image(img).cpu().numpy().flatten()
        embs.append(e)
    embs = np.stack(embs)

    # Hierarchical clustering
    clusters = detect_cluster.hierarchical_cluster(embs, paths)

    # Caption all frames
    caps = {p: caption.caption_frame(p) for p in paths}

    # Summaries
    time_summ = summarization.summarize_by_time(caps)
    obj_summ = summarization.summarize_by_object(caps)

    # Build graph
    nodes = [(i, caps[p]) for i,p in enumerate(paths)]
    G = build_temporal_graph(nodes)

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
        text, tsec = parse_query(q)
        if tsec is not None:
            # nearest frame index by time (assuming 1 FPS)
            idx = min(range(len(paths)), key=lambda i: abs(i - tsec))
        else:
            # ground via semantic similarity
            eps = [{'caption': caps[p]} for p in paths]
            idx = ground_event(text, eps)

        # gather context window
        ctx = gather_context(G, idx, before=5, after=5)
        # split into chunks
        half = len(ctx) // 2 or len(ctx)
        chunk1 = "\n".join(f"{i}: {cap}" for i,_,cap in ctx[:half])
        chunk2 = "\n".join(f"{i}: {cap}" for i,_,cap in ctx[half:])
        answer = ask_with_chunks([chunk1, chunk2], text)

        st.subheader('💬 Answer')
        st.write(answer)
        st.image(utils.show_image(paths[idx]), width=400, caption=f'Frame {idx}')
