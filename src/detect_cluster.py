from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Scene boundary detection
def detect_scenes(video_path: str, threshold: float = 27.0):
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    vm.set_downscale_factor()
    vm.start()
    sm.detect_scenes(vm)
    scenes = sm.get_scene_list()
    return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]

# Hierarchical clustering of embeddings
def hierarchical_cluster(embeddings: np.ndarray, frame_paths: list, distance_threshold: float = 0.5):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        affinity='euclidean',
        linkage='ward'
    )
    labels = clustering.fit_predict(embeddings)
    clusters = {}
    for lab, fp, emb in zip(labels, frame_paths, embeddings):
        clusters.setdefault(lab, []).append((fp, emb))
    return clusters