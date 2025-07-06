# VideoMind: Dynamic Episodic Memory Engine for Video QA & Counterfactual Reasoning

## 🚀 Overview

VideoMind is a novel system for deep video understanding, enabling users to ask both factual and counterfactual questions about any video. It combines state-of-the-art vision models, hierarchical clustering, temporal graph construction, and LLM-powered reasoning to deliver rich, context-aware answers.

---

## 🧩 Key Features

- **Automatic frame extraction and BLIP captioning**
- **CLIP-based hierarchical clustering in time**
- **YOLOv8-based object detection and object-centric clustering**
- **Temporal graph construction for context reasoning**
- **LLM-powered (Ollama/Mistral) summarization and QA**
- **Counterfactual reasoning grounded in video context**
- **Interactive Gradio UI for seamless user experience**

---

## 🛠️ How It Works

1. **Frame Extraction & Captioning**
   - Video is split into frames at a chosen FPS.
   - Each frame is captioned using BLIP.

2. **CLIP Embedding & Clustering**
   - Frames are embedded using CLIP.
   - Hierarchical clustering is performed in the time domain.
   - YOLOv8 detects objects, enabling object-based clustering.

3. **Temporal Graph Construction**
   - Both time and object clusters are represented as nodes in a temporal graph.
   - Edges encode temporal relationships.

4. **Summarization**
   - Each time and object cluster is summarized using an LLM (Ollama/Mistral).

5. **Interactive QA & Counterfactual Reasoning**
   - Users ask questions via the Gradio UI.
   - The system detects relevant time/object context, gathers summaries, and formulates a prompt for the LLM.
   - For counterfactuals, the LLM is restricted to only reason about the background, objects, and events up to the point of change.

---

## 🖥️ Usage

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
```

You also need to install [Ollama](https://ollama.com/) and pull the Mistral model:

```bash
ollama pull mistral:latest
```

### 2. **Run the Gradio App**

```bash
python scripts/gradio_app.py
```

- Open the provided local URL in your browser.
- Upload a video (`.mp4`, `.mov`, `.avi`, `.mkv`).
- Click **Process Video**.
- Ask questions about the video or try counterfactuals (e.g., "What if the car turned left?").

### 3. **Command-Line Interface (Optional)**

You can also use the CLI version:

```bash
python scripts/cli_main.py
```

---

## 🧠 Novelty & Architecture

- **Hierarchical Temporal Clustering:** Recursive time-based clustering enables multi-scale temporal reasoning.
- **Object-centric Episodic Memory:** Object-based clustering and summaries support object-centric queries and counterfactuals.
- **Temporal Graph:** Enables efficient context window gathering for any frame or cluster.
- **LLM-Guided Summarization & QA:** All answers are grounded in extracted context, minimizing hallucination.
- **Counterfactual Reasoning:** The system can answer "what if" questions by leveraging only the available context and objects.


---

Authored by Atharva Date

