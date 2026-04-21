import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io


class Recommender:
    def __init__(self, attr_path, landmark_path):
        self.attr_df = self._load_celeba_style_file(attr_path)
        self.land_df = self._load_celeba_style_file(landmark_path)

        if self.attr_df is None:
            self.filenames = []
            return

        self.filenames = self.attr_df.iloc[:, 0].values
        self.land_df = self.land_df[self.land_df.iloc[:, 0].isin(self.filenames)]

        attr_values = self.attr_df.iloc[:, 1:].values
        land_values = self.land_df.iloc[:, 1:].values / 250.0

        self.features = np.hstack([attr_values, land_values])
        self.attr_names = self.attr_df.columns[1:].tolist()

        self.user_profile = np.zeros(self.features.shape[1])
        self.liked_count = 0
        self.viewed_indices = set()

    def _load_celeba_style_file(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            headers = lines[1].strip().split()
            data_lines = [l.strip() for l in lines[2:] if '.jpg' in l.lower()]
            if not data_lines: return None
            return pd.read_csv(io.StringIO('\n'.join(data_lines)), sep=r'\s+', names=['image_id'] + headers,
                               engine='python')
        except:
            return None

    def get_next(self):
        n_files = len(self.filenames)
        if n_files == 0: return "no_data", -1

        # Logic to strictly avoid duplicates
        available_indices = list(set(range(n_files)) - self.viewed_indices)
        if not available_indices:
            self.viewed_indices.clear()  # Reset if we run out of photos
            available_indices = list(range(n_files))

        if self.liked_count < 3:
            idx = np.random.choice(available_indices)
        else:
            similarities = cosine_similarity([self.user_profile], self.features)[0]
            # Mask out viewed indices
            for v_idx in self.viewed_indices:
                similarities[v_idx] = -2.0
            idx = int(np.argmax(similarities))

        self.viewed_indices.add(idx)
        return str(self.filenames[idx]), int(idx)

    def find_best_match(self):
        """Finds the best match using full vector cosine similarity."""
        if self.liked_count == 0:
            return self.get_next()

        # Calculate cosine similarity across all features (Attributes + Landmarks)
        # user_profile is a vector of floats, features is a matrix of 1/-1 and floats
        similarities = cosine_similarity([self.user_profile], self.features)[0]

        idx = int(np.argmax(similarities))
        best_score = similarities[idx]

        # Extract attribute data for debug
        n_attr = len(self.attr_names)
        match_attrs = self.features[idx][:n_attr]
        active_traits = [self.attr_names[i] for i, val in enumerate(match_attrs) if val == 1]

        return str(self.filenames[idx]), idx, active_traits, best_score

    def update_profile(self, idx, liked):
        if idx == -1: return
        alpha = 0.4 if liked else 0.2
        target = self.features[idx]
        if liked:
            self.user_profile = self.user_profile + alpha * (target - self.user_profile)
            self.liked_count += 1
        else:
            self.user_profile = self.user_profile - alpha * (target - self.user_profile)