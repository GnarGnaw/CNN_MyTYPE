import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MatchEngine:
    def __init__(self):
        # We track the 'embeddings' (attributes + landmarks) of people you like
        self.liked_embeddings = []
        self.all_embeddings = {}  # {image_id: vector}

    def add_to_database(self, image_id, attr_probs, landmarks):
        # Flatten landmarks and combine with attributes to create a 'Face Vector'
        vector = np.concatenate([attr_probs, landmarks.flatten()])
        self.all_embeddings[image_id] = vector

    def record_like(self, image_id):
        if image_id in self.all_embeddings:
            self.liked_embeddings.append(self.all_embeddings[image_id])

    def find_best_match(self):
        if not self.liked_embeddings:
            return None

        # Create 'My Type' average vector
        target_vector = np.mean(self.liked_embeddings, axis=0).reshape(1, -1)

        best_score = -1
        best_match = None

        for img_id, vector in self.all_embeddings.items():
            # Skip if you already liked them
            if any(np.array_equal(vector, liked) for liked in self.liked_embeddings):
                continue

            score = cosine_similarity(target_vector, vector.reshape(1, -1))[0][0]
            if score > best_score:
                best_score = score
                best_match = img_id

        return best_match, best_score