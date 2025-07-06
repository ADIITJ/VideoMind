import os
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import numpy as np
from PIL import Image
import torch
from src import extract, detect_cluster, caption, summarization, utils
from src.graph_context import build_temporal_graph, gather_context
from src.query_parser import parse_query, ground_event
from src.llm_reasoning import summarize_time_llm, summarize_object_llm
from open_clip import create_model_and_transforms

def find_best_object(query, obj_summ):
    # Detect the best matching object from the query using substring matching (can be improved with embeddings)
    best_obj = None
    best_len = 0
    q_lower = query.lower()
    for obj in obj_summ.keys():
        candidate = obj.lower()
        if candidate in q_lower and len(candidate) > best_len:
            best_obj = obj
            best_len = len(candidate)
    return best_obj

def get_nearest_time_summary(idx, paths, time_summ):
    best_key = None
    best_diff = float('inf')
    for key in time_summ.keys():
        if key in paths:
            pseudo_idx = paths.index(key)
            diff = abs(pseudo_idx - idx)
            if diff < best_diff:
                best_diff = diff
                best_key = key
    return time_summ.get(best_key, "")

def is_counterfactual(query):
    cf_phrases = ["what if", "suppose", "imagine", "if instead", "had happened", "if only", "if", "if we"]
    ql = query.lower()
    return any(phrase in ql for phrase in cf_phrases)

def get_whole_video_summary(caps):
    from src.llm_reasoning import summarize_time_llm
    return summarize_time_llm(list(caps.values()))

def main():
    video_path = '/Users/ashishdate/Downloads/trial.mov'
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        return

    os.makedirs('data/frames', exist_ok=True)
    video_ext = os.path.splitext(video_path)[-1]
    vp = f'data/input{video_ext}'
    if video_path != vp:
        print(f"Copying video to {vp}")
        with open(video_path, 'rb') as src, open(vp, 'wb') as dst:
            dst.write(src.read())
    else:
        print(f"Using video at {vp}")

    print("Extracting frames...")
    n = extract.extract_frames(vp, 'data/frames', fps=1)
    print(f"Extracted {n} frames at 1 FPS.")

    frames = sorted(os.listdir('data/frames'))
    paths = [os.path.join('data/frames', f) for f in frames]

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

    print("Clustering embeddings...")
    clusters = detect_cluster.hierarchical_cluster(embs, paths)
    print(f"Clusters: {list(clusters.keys())}")

    print("Captioning frames...")
    caps = {}
    for i, p in enumerate(paths):
        print(f"Captioning frame {i}: {p}")
        caps[p] = caption.caption_frame(p)
    print("All frames captioned.")

    print("Summarizing by time...")
    time_summ = summarization.summarize_by_time(caps)
    print("Summarizing by object...")
    obj_summ = summarization.summarize_by_object(caps)

    print("Building temporal graph...")
    nodes = [(i, caps[p]) for i, p in enumerate(paths)]
    G = build_temporal_graph(nodes)
    print("Temporal graph built.")

    print("\nSummaries by Time Window:")
    for t, text in time_summ.items():
        print(f'{t}: {text}')
    print("\nSummaries by Object:")
    for obj, text in obj_summ.items():
        print(f'{obj}: {text}')

    print("\nReady for questions! (Type 'exit' to quit)")
    while True:
        q = input("\nAsk about this video (timestamp or event, or counterfactual): ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not q:
            continue
        text, tsec = parse_query(q)
        print(f"Parsed query: text={text}, tsec={tsec}")
        if tsec is not None:
            idx = min(range(len(paths)), key=lambda i: abs(i - tsec))
            print(f"Using provided timestamp, selected frame idx: {idx}")
        else:
            eps = [{'caption': caps[p]} for p in paths]
            idx = ground_event(text, eps)
            print(f"Using semantic grounding, selected frame idx: {idx}")

        # Determine time summary by finding nearest key from time_summ
        time_summary = get_nearest_time_summary(idx, paths, time_summ)
        print(f"Selected time summary: {time_summary}")

        # Detect object mentioned in the query and get object summary if available
        best_obj = find_best_object(q, obj_summ)
        if best_obj:
            object_summary = obj_summ[best_obj]
            print(f"Detected object in query: {best_obj} with summary: {object_summary}")
        else:
            object_summary = ""
            print("No object-specific summary detected.")

        print("Gathering context window...")
        ctx = gather_context(G, idx, before=5, after=5)
        print(f"Context window: {ctx}")
        ctx_captions = [cap for i, _, cap in ctx]

        # Compose LLM context
        if is_counterfactual(q):
            print("Counterfactual query detected.")
            # Get summary of the whole video
            whole_summary = get_whole_video_summary(caps)
            # Compose object summaries for all objects
            all_obj_summaries = "\n".join([f"{obj}: {summ}" for obj, summ in obj_summ.items()])
            # Context up to the point of change
            context_until_change = [cap for i, _, cap in ctx if i <= idx]
            llm_context = (
                "Whole video summary:\n" + whole_summary + "\n\n"
                "Object summaries:\n" + all_obj_summaries + "\n\n"
                "Context captions up to the point of change:\n" + "\n".join(context_until_change)
            )
            system_prompt = (
                "You are an expert video reasoning assistant. "
                "You are to answer a counterfactual question about the video. "
                "Base your answer strictly on the provided whole video summary, object summaries, and the context up to the point of change. "
                "Do not invent new objects or events. "
                "Limit your answer to what is plausible given the background and objects present. "
                "Be concise and logical."
            )
            user_prompt = (
                f"{llm_context}\n\nCounterfactual Question:\n{q}\n\nCounterfactual Answer:"
            )
        else:
            # Normal factual QA
            llm_context = "Time summary: " + time_summary + "\n"
            if object_summary:
                llm_context += f"Object summary ({best_obj}): " + object_summary + "\n"
            llm_context += "Context captions:\n" + "\n".join(ctx_captions)
            system_prompt = (
                "You are an expert video reasoning assistant. "
                "Using the provided time summary, object summary (if any), and context captions, "
                "answer the user's question strictly based on this information. "
                "Do not invent new events. "
                "Keep your answer concise and factual."
            )
            user_prompt = (
                f"{llm_context}\n\nQuestion:\n{q}\n\nAnswer:"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        import ollama
        resp = ollama.chat(model="mistral:latest", messages=messages)
        answer = resp.get('message', {}).get('content', '').strip()

        print("\n💬 Answer:")
        print(answer)
        print(f"(Frame {idx}: {paths[idx]})")

if __name__ == "__main__":
    main()
