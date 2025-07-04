import re
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")

# Parse query
def parse_query(text: str) -> tuple:
    match = TIME_PATTERN.search(text)
    if match:
        mm, ss = map(int, match.groups())
        return text, mm*60 + ss
    return text, None

# Ground event
def ground_event(text: str, episodes: list) -> int:
    q_emb = model.encode(text, convert_to_tensor=True)
    caps = [ep['caption'] for ep in episodes]
    emb_caps = model.encode(caps, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, emb_caps)[0].cpu().numpy()
    return int(sims.argmax())