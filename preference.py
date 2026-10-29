"""Preference learning and adaptive pair sampling; pure numpy, testable."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.zeros_like(v)


@dataclass
class PreferenceSession:
    embeddings: np.ndarray
    decay: float = 0.85
    alpha: float = 0.45
    beta: float = 0.45
    stretch_weight: float = 1.6
    frontier_weight: float = 1.4
    seed: int = 42
    user_vector: np.ndarray = field(init=False)
    seen: set = field(default_factory=set)
    count: int = 0
    history: list = field(default_factory=list)
    last_pair: Optional[tuple] = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or len(self.embeddings) < 2:
            raise ValueError('At least two image vectors are required')
        self.embeddings = np.stack([normalize(v) for v in self.embeddings])
        self.user_vector = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        self.rng = np.random.default_rng(self.seed)

    def select_pair(self):
        n = len(self.embeddings)
        pool = np.array([i for i in range(n) if i not in self.seen], dtype=int)
        if len(pool) < 2:
            self.seen.clear()
            pool = np.arange(n)
        if self.count == 0 or np.linalg.norm(self.user_vector) < 1e-9:
            first = int(self.rng.choice(pool))
            candidates = pool[pool != first]
            # Cold-start diversity: choose an image relatively different from first.
            second = int(candidates[np.argmin(self.embeddings[candidates] @ self.embeddings[first])])
        else:
            sim = self.embeddings[pool] @ self.user_vector
            # Cycle through 4:6:2 safe/stretch/frontier, using ranking regions.
            mode = ['safe'] * 4 + ['stretch'] * 6 + ['frontier'] * 2
            category = mode[self.count % len(mode)]
            order = np.argsort(sim)[::-1]
            if category == 'safe':
                segment = order[:max(2, len(order) // 3)]
            elif category == 'stretch':
                lo, hi = len(order) // 4, max(len(order) // 4 + 2, (3 * len(order)) // 4)
                segment = order[lo:hi]
            else:
                segment = order[max(0, (2 * len(order)) // 3):]
            first = int(self.rng.choice(pool[segment]))
            other = pool[pool != first]
            # Second image is near first on user preference axis, but visually different.
            axis_delta = np.abs((self.embeddings[other] - self.embeddings[first]) @ self.user_vector)
            diversity = 1 - self.embeddings[other] @ self.embeddings[first]
            novelty = np.array([0 if i not in self.seen else 0.4 for i in other])
            utility = -axis_delta + 0.25 * diversity - novelty
            top = np.argsort(utility)[-min(8, len(utility)):]
            second = int(self.rng.choice(other[top]))
        self.seen.update([first, second])
        self.last_pair = (first, second)
        return self.last_pair

    def vote(self, choice: str):
        if self.last_pair is None:
            raise ValueError('Call select_pair first')
        # Strong / slight preference for left or right; ties are skipped.
        strength = {'left_strong': 1.0, 'left_slight': 0.5, 'right_slight': -0.5, 'right_strong': -1.0, 'skip': 0.0}[choice]
        a, b = self.last_pair
        if strength:
            direction = self.embeddings[a] - self.embeddings[b]
            # Normalize after each update, as in the original preference-vector formulation.
            self.user_vector = normalize(self.decay * self.user_vector + self.alpha * strength * direction).astype('float32')
        self.history.append({'round': self.count + 1, 'left': a, 'right': b, 'choice': choice})
        self.count += 1
        self.last_pair = None

    def ranking(self, count=12):
        if np.linalg.norm(self.user_vector) < 1e-9:
            return np.arange(min(count, len(self.embeddings))), np.zeros(min(count, len(self.embeddings)))
        scores = self.embeddings @ self.user_vector
        indices = np.argsort(-scores)[:count]
        return indices, scores[indices]
