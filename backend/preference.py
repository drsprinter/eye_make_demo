"""Two-stage preference study: 20 four-choice pairs, then 10 five-star ratings."""
import numpy as np

PAIR_ROUNDS = 20
RATING_ROUNDS = 10


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-8 else np.zeros_like(x)


class PreferenceSession:
    def __init__(self, embeddings, seed=None):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) < PAIR_ROUNDS * 2 + RATING_ROUNDS or not np.isfinite(x).all():
            raise ValueError('At least 50 valid image embeddings required')
        self.x = np.array([unit(v) for v in x], dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.phase = 'pair'
        self.pair_count = 0
        self.rating_count = 0
        self.pair = None
        self.candidate = None
        self.seen = set()
        self.rated = set()
        self.history = []
        self.positive = np.zeros(x.shape[1], dtype=np.float32)
        self.negative = np.zeros(x.shape[1], dtype=np.float32)
        self.long_vector = np.zeros(x.shape[1], dtype=np.float32)
        self.short_raw = np.zeros(x.shape[1], dtype=np.float32)
        self.short_vector = np.zeros(x.shape[1], dtype=np.float32)
        self.select_pair()

    def select_pair(self):
        if self.phase != 'pair':
            return None
        available = np.array([i for i in range(len(self.x)) if i not in self.seen], dtype=int)
        if len(available) < 2:
            raise ValueError('Not enough unshown images for 20 comparisons')
        # Diverse, without replacement. Bias second toward semantic contrast.
        a = int(self.rng.choice(available))
        remaining = available[available != a]
        distance = 1 - self.x[remaining] @ self.x[a]
        top = np.argsort(distance)[-min(15, len(remaining)):]
        b = int(self.rng.choice(remaining[top]))
        self.pair = (a, b)
        self.seen.update((a, b))
        return self.pair

    def vote(self, choice):
        if self.phase != 'pair' or self.pair is None:
            raise ValueError('Pair phase not active')
        if choice not in ('left', 'right', 'both_like', 'both_dislike'):
            raise ValueError('Invalid comparison choice')
        a, b = self.pair
        # All choices provide directional evidence; both-like and both-dislike
        # respectively add positive and negative evidence for BOTH exemplars.
        if choice == 'left':
            self.positive += self.x[a]
            self.negative += self.x[b]
        elif choice == 'right':
            self.positive += self.x[b]
            self.negative += self.x[a]
        elif choice == 'both_like':
            self.positive += 0.5 * (self.x[a] + self.x[b])
        else:
            self.negative += 0.5 * (self.x[a] + self.x[b])
        self.pair_count += 1
        self.history.append({'phase': 'pair', 'round': self.pair_count, 'left': a, 'right': b, 'choice': choice, 'image': '', 'rating': ''})
        self.pair = None
        # Run the entire study using the progressively updated estimate.
        self.long_vector = unit(self.positive - self.negative)
        if self.pair_count == PAIR_ROUNDS:
            self.phase = 'rating'
            self.select_candidate()
        else:
            self.select_pair()

    def select_candidate(self):
        if self.phase != 'rating':
            return None
        remaining = np.array([i for i in range(len(self.x)) if i not in self.rated and i not in self.seen], dtype=int)
        if not len(remaining):
            raise ValueError('No unrated images remain')
        # Sample from highly aligned long-term preferences; slight diversity in
        # rank prevents showing only visually near-identical images.
        sim = self.x[remaining] @ self.long_vector
        order = np.argsort(-sim)
        shortlist = remaining[order[:min(40, len(order))]]
        # Scan high-ranked candidates with controlled exploration.
        idx = min(len(shortlist) - 1, (self.rating_count * 3) % max(len(shortlist), 1))
        self.candidate = int(shortlist[idx])
        return self.candidate

    def rate(self, rating):
        if self.phase != 'rating' or self.candidate is None:
            raise ValueError('Rating phase not active')
        if not isinstance(rating, int) or rating not in (1, 2, 3, 4, 5):
            raise ValueError('Rating must be an integer from 1 to 5')
        i = self.candidate
        # 3 is neutral. Negative ratings move the session vector away.
        self.short_raw += (rating - 3) / 2 * self.x[i]
        self.short_vector = unit(self.short_raw)
        self.rated.add(i)
        self.rating_count += 1
        self.history.append({'phase': 'rating', 'round': self.rating_count, 'left': '', 'right': '', 'choice': '', 'image': i, 'rating': rating})
        self.candidate = None
        if self.rating_count == RATING_ROUNDS:
            self.phase = 'done'
        else:
            self.select_candidate()

    def ranking(self, limit=10):
        # Long-term portrait is frozen after round 20; short-term refines it.
        if np.linalg.norm(self.short_vector) > 1e-8:
            profile = unit(0.7 * self.long_vector + 0.3 * self.short_vector)
        else:
            profile = self.long_vector
        scores = self.x @ profile
        # Never recommend explicitly rejected rated images (ratings 1-2).
        rejected = {int(h['image']) for h in self.history if h['phase'] == 'rating' and h['rating'] <= 2}
        eligible = np.array([i for i in range(len(self.x)) if i not in rejected], dtype=int)
        order = eligible[np.argsort(-scores[eligible])][:limit]
        return order, scores[order]
