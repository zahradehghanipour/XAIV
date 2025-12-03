import copy
import os
import pickle
import random
from collections import Counter, defaultdict, namedtuple
from enum import Enum

from dl_verifier.utils.debug_utils import _DEBUGENABLED, debug, info, warn

# TODO needs refactor, not using this in RL.py just yet.

StatePair = namedtuple("StatePair", ["s0", "s1"])
EMetadata = namedtuple("EMetadata", ["who", "eid"])


class DecisionMaker(Enum):
    RND = 0
    EXP = 1
    DQN = 2


class ReplayBuffer:
    def __init__(self):
        """
        Stores and retrieves past experiences for training.
        """
        self.buffer = []
        self.episode = []  # The current episode
        self.nepisodes = 0
        self.nsafeepisodes = 0
        self.safeepisodes = []
        self.nsteps = 0
        self.whodoneit = defaultdict(list)

    def _whodonewhat(self):
        offset = self.size
        actors = []  # List of actors for current episode
        for idx, (meta, *_rest) in enumerate(self.episode):
            self.whodoneit[meta.who].append(offset + idx)
            actors.append(meta.who)
        return Counter(actors)

    def _normalize(self, episode, epsilon=1e-8, method="sum"):
        """
        Normalizes rewards in a single episode.
        """
        debug(f"Normalise rewards with method '{method}'")

        rewards = [r for *_rest, r, _ in episode]

        if method == "sum":
            total = sum(rewards) + epsilon
            normalized = [r / total for r in rewards]

        elif method == "minmax":
            rmin = min(rewards)
            # `r = state.s1.glb - state.s0.glb`, and `glb` is less or equal to `0`;
            # so `rmax = 0`
            rmax = 0
            scale = (rmax - rmin) + epsilon
            normalized = [(r - rmin) / scale for r in rewards]

        elif method == "zscore":
            # Compute mean and std. deviation
            ravg = sum(rewards) / len(rewards)
            rdev = (
                sum((r - ravg) ** 2 for r in rewards) / len(rewards)
            ) ** 0.5 + epsilon
            normalized = [(r - ravg) / rdev for r in rewards]

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

        return [
            (metadata, state, action, normal, terminal, depth)
            for (normal, (metadata, state, action, _, terminal, depth)) in zip(
                normalized, episode
            )
        ]

    def _computereward(self, s0, s1):
        """
        A simple reward function for now, based on global lower bound(s)
        """
        # Safety check

        glb0 = s0["glb"].view(-1)  # curr glb
        glb1 = s1["glb"].view(-1)  # next glb

        if glb0.shape != glb1.shape:
            raise ValueError(
                f"Global lower bounds do not match ({glb0.shape} vs {glb1.shape})"
            )

        return (glb1 - glb0).min().item()

    def step(self, s0, s1, action, who, depth):
        # debug(f"Storing step #{(self.nsteps + 1):03d}")
        self.episode.append(
            (
                EMetadata(who, self.nepisodes),
                StatePair(s0, s1),
                action,
                self._computereward(s0, s1),
                0,
                depth,
            )
        )
        self.nsteps += 1

    def episodeComplete(self, safe, bonus=10, penalty=-1, norm=None):
        # Index episode so we know who done which action
        # in the current episode.
        counts = self._whodonewhat()

        info(
            f"[ZRL] Episode #{(self.nepisodes + 1):03d} completed ({safe}) in {len(self.episode)} steps\n"
            f"[ZRL]\tEXP: {counts[DecisionMaker.EXP]:02d}\n"
            f"[ZRL]\tDQN: {counts[DecisionMaker.DQN]:02d}\n"
            f"[ZRL]\tRND: {counts[DecisionMaker.RND]:02d}"
        )

        # Update last entry in episode to indicate terminal state
        metadata, state, action, reward, _, depth = self.episode[-1]
        reward += bonus if safe else penalty
        self.episode[-1] = (metadata, state, action, reward, 1, depth)  # 1 = terminal

        if norm is not None:
            self.episode = self._normalize(self.episode, method=norm)

        self.buffer.extend(self.episode)

        if safe:
            # Keep track of the safe episode ids
            self.safeepisodes.append(self.nepisodes)

        # Clear episode and inc. counters
        self.episode = []
        self.nepisodes += 1
        return

    def sample(self, batchsize=None, onlysuccessful=False, who: DecisionMaker = None):
        assert self.size > 0, "Empty buffer"

        # Take a stable snapshot of current size
        size = len(self.buffer)

        # 1) Build initial candidate indices
        if who is not None:
            sampleids = self.whodoneit.get(who, [])
            # Guard against stale indices
            sampleids = [i for i in sampleids if 0 <= i < size]
            if not sampleids:
                raise ValueError(f"No actions by decision-maker {who}")
        else:
            # Full range (already valid)
            # (Print can stay if you want)
            # print(f"[ZRL] size: {size}")
            sampleids = list(range(size))

        # Bail out early if nothing to sample
        if not sampleids:
            raise ValueError("No valid samples available in buffer.")

        # 2) If only successful, keep steps whose episode id is in safeepisodes
        if onlysuccessful:
            safe_set = set(self.safeepisodes)
            sampleids = [
                idx
                for idx in sampleids
                if 0 <= idx < size and self.buffer[idx][0].eid in safe_set
            ]
            if not sampleids:
                raise ValueError("No samples from successful episodes left.")

        # 3) Choose ids
        if batchsize is None or batchsize >= len(sampleids):
            selectedids = sampleids
            if batchsize is not None and batchsize > len(sampleids):
                warn(f"Only {len(selectedids)}/{batchsize} returned in batch")
        else:
            selectedids = random.sample(sampleids, batchsize)

        # 4) Shuffle in-place to randomize order even when requesting all
        random.shuffle(selectedids)

        # 5) Gather samples
        samples = [self.buffer[idx] for idx in selectedids]
        metadata, state, action, reward, terminal, depth = zip(*samples)

        if _DEBUGENABLED:
            stats = {
                "who": Counter(m.who for m in metadata),
                "eid": Counter(m.eid for m in metadata),
            }
            s = "\n"
            s += "[ZRL]Training batch:\n"
            s += f"[ZRL]\t{len(samples)} samples from {len(stats['eid'])} episodes:\n"
            for dm in DecisionMaker:
                s += f"[ZRL]\t\t{dm.name}: {stats['who'][dm]:4d}\n"
            debug(s)

        return state, action, reward, terminal, depth

    def step_sample(
        self, batchsize=None, onlysuccessful=True, who: DecisionMaker = None
    ):
        """
        Sample a batch of individual steps from the buffer.
        Each step is a tuple: (metadata, state, action, reward, terminal, depth)
        """
        assert len(self.buffer) > 0, "Empty buffer"

        # Gather all indices
        if who is not None:
            sampleids = self.whodoneit.get(who, [])
            sampleids = [i for i in sampleids if i < len(self.buffer)]
            if not sampleids:
                raise ValueError(f"No valid steps from decision-maker {who}")
        else:
            sampleids = list(range(len(self.buffer)))

        if batchsize is None or batchsize >= len(sampleids):
            selectedids = sampleids
            if batchsize is not None and batchsize > len(sampleids):
                warn(f"[ZRL] Only {len(selectedids)}/{batchsize} returned in batch")
        else:
            selectedids = random.sample(sampleids, batchsize)

        random.shuffle(selectedids)

        # Extract samples
        samples = [self.buffer[idx] for idx in selectedids]

        # Unpack each sample tuple
        metadata, state, action, reward, terminal, depth = zip(*samples)

        if _DEBUGENABLED:
            # Summarize batch stats
            stats = {"who": Counter(m.who for m in metadata)}
            s = "\n"
            s += "Training batch:\n"
            s += f"\t{len(samples)} samples\n"
            for who in DecisionMaker:
                s += f"\t\t{who.name}: {stats['who'][who]:4d}\n"
            debug(s)

        return state, action, reward, terminal, depth

    def __len__(self):
        return len(self.buffer)

    @property
    def size(self):
        return len(self.buffer)

    def clear(self):
        # Clear episode
        self.episode = []
        # Clear buffer
        self.buffer = []

    def info(self):
        """
        Prints out statistics.
        """
        s = "\n"
        s += "[ZRL]Replay buffer:\n"

        s += f"[ZRL]\t{self.size} steps:\n"
        for who in DecisionMaker:
            lst = self.whodoneit.get(who, [])
            s += f"[ZRL]\t\t{who.name}: {len(lst):4d} steps\n"

        s += f"[ZRL]\t{len(self.safeepisodes)}/{self.nepisodes} safe episodes\n"

        info(f"{s}")

    def store(self, path="buffer.pkl", prefix=None):
        # Is there an episode in progress?!
        if len(self.episode) > 0:
            raise Exception("Cannot store state while an episode is in progress")

        if prefix:
            directory, filename = os.path.split(path)
            filename = f"{prefix}-{filename}"
            path = os.path.join(directory, filename)

        with open(path, "wb") as fstream:
            pickle.dump(self, fstream)

    @classmethod
    def restore(cls, path="buffer.pkl"):
        if not os.path.exists(path):
            warn(f"File {path} not found. Starting with empty buffer.")
            return cls()

        info(f"[ZRL] Restoring state from {path}")

        with open(path, "rb") as f:
            buffer = pickle.load(f)

        if not isinstance(buffer, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        buffer.info()

        return buffer

    def safe_buffer(self, path: str = "safe_buffer.pkl"):
        """
        Create and save a new ReplayBuffer that contains ONLY steps from safe episodes.

        Args:
            path: File path to save the safe-only buffer (pickle).

        Returns:
            ReplayBuffer: the newly created safe-only buffer (also saved to `path`).

        Raises:
            ValueError: when main buffer is empty or no safe episodes recorded.
        """
        # Guard rails
        if len(self.buffer) == 0:
            raise ValueError("Cannot create safe buffer: main buffer is empty.")
        if len(self.safeepisodes) == 0:
            raise ValueError("Cannot create safe buffer: no safe episodes recorded.")

        safe_eids = set(self.safeepisodes)

        # Build a new buffer with only safe-episode entries
        safe_buf = ReplayBuffer()
        safe_buf.buffer = []
        safe_buf.episode = []  # ensure no in-progress episode
        safe_buf.whodoneit = defaultdict(list)

        for _, (meta, state, action, reward, terminal, depth) in enumerate(self.buffer):
            if meta.eid in safe_eids:
                new_idx = len(safe_buf.buffer)
                safe_buf.buffer.append((meta, state, action, reward, terminal, depth))
                safe_buf.whodoneit[meta.who].append(new_idx)

        # Recompute counters/metadata for the subset
        subset_eids = sorted({meta.eid for (meta, *_rest) in safe_buf.buffer})
        safe_buf.nepisodes = len(subset_eids)
        safe_buf.nsteps = len(safe_buf.buffer)
        safe_buf.safeepisodes = subset_eids[:]  # all episodes in this subset are safe
        safe_buf.nsafeepisodes = len(subset_eids)

        # (Optional) quick sanity checks
        if safe_buf.nsteps == 0:
            raise ValueError(
                "Safe buffer ended up empty (no steps from safe episodes)."
            )

        # Persist to disk using the class' own store logic
        safe_buf.store(path)

        info(
            f"[ZRL] Saved safe-only buffer: {len(safe_buf.buffer)} steps "
            f"from {len(safe_buf.safeepisodes)} safe episodes -> {path}"
        )

        return safe_buf

    def split_by_steps(
        self, exp, buffer, train_ratio=0.8, seed=None, n_steps=None, save=False
    ):
        """
        Split a ReplayBuffer into train/test by steps with deterministic shuffling.

        Args:
            buffer: ReplayBuffer
            train_ratio (float):    Fraction of selected steps to put into the train
                                    split (0., 1.).
            seed (int|None):        If given, use this seed for a reproducible shuffle
                                    of step indices.
            n_steps (int|None):     If given, use only the first `n_steps` steps *after*
                                    shuffling. If None and `ask=True`, you'll be
                                    prompted. If None and `ask=False`, uses all steps.
            save (bool):            If True, saves two new buffers next to `path`, with
                                    filenames including seed and n_steps.

        Returns:
            (train_buffer, test_buffer):    Deep-copied buffers with `.buffer` replaced
                                            by the split step lists.
        """
        if not hasattr(buffer, "buffer"):
            raise AttributeError(
                "[ZRL]Loaded object does not have a `.buffer` attribute."
            )

        steps = list(buffer.buffer)
        total = len(steps)
        print(f"[ZRL][Split] Loaded buffer with {total} steps.")

        # -- clamp n_steps
        if n_steps is None:
            n_steps = total
        if n_steps == -1:
            n_steps = total
        if n_steps <= 0:
            raise ValueError("[ZRL]n_steps must be a positive integer.")
        if n_steps > total:
            print(
                f"[ZRL][Split] Requested n_steps={n_steps} exceeds total={total}; using all {total}."
            )
            n_steps = total

        # -- deterministic shuffle via local RNG; shuffle indices, not the list
        indices = list(range(total))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(indices)

        # -- select subset
        selected_idx = indices[:n_steps]
        selected_steps = [steps[i] for i in selected_idx]

        # -- split
        if not (0.0 <= train_ratio <= 1.0):
            raise ValueError("[ZRL]train_ratio must be between 0.0 and 1.0")

        split_index = int(train_ratio * n_steps)
        train_steps = selected_steps[:split_index]
        test_steps = selected_steps[split_index:]

        # -- deepcopy meta and replace .buffer
        train_buffer = copy.deepcopy(buffer)
        test_buffer = copy.deepcopy(buffer)
        train_buffer.buffer = train_steps
        test_buffer.buffer = test_steps

        print(f"[ZRL][Split] Seed: {seed}")
        print(
            f"[ZRL][Split] Using {n_steps} steps → Train: {len(train_steps)}, Test: {len(test_steps)} (train_ratio={train_ratio:.2f})"
        )

        # -- optional save
        if save:
            # out_train = f"{exp}_train.pkl"
            out_test = f"{exp}_test.pkl"
            # with open(out_train, "wb") as f:
            #     pickle.dump(train_buffer, f)
            with open(out_test, "wb") as f:
                pickle.dump(test_buffer, f)
            # print(f"[ZRL][Split] Train buffer saved to: {out_train}")
            print(f"[ZRL][Split] Test buffer saved to: {out_test}")

        return train_buffer, test_buffer
