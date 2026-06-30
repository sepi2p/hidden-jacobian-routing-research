import numpy as np
import torch
from .base_attack import BaseAttack
import torch.nn.functional as F


class MyAttack(BaseAttack):
    """
    GA-based black-box attack, implemented in image space.

    - Population are adversarial images x_adv in [0,1].
    - L_inf constraint enforced w.r.t. the clean image x0.
    - use_mcg:
        * True  -> use C-Glow/MCG to seed the initial population
                   with MANY different latent samples (multi-z).
        * False -> pure GA, population is random in the epsilon-ball.
    - Fitness (defined here, using probs from loss_func):
        * Untargeted:   margin = p(true)   - max_{k≠true}   p(k)
        * Targeted:     margin = max_{k≠t} p(k) - p(t)
        * score = -margin
          Success iff margin <= 0.0
    """

    def __init__(self, dataset_name, max_query, targeted, class_num,
                 linf=0.05, use_mcg=True):
        super().__init__(dataset_name, max_query, targeted, class_num, linf)

        # GA hyperparams
        self.pop_size = 15           # population size
        self.num_elite = 10          # extra elites beyond the best
        self.tourn_k = 5             # tournament size (1 = random parent)
        self.crossover_prob = 0.9    # crossover probability
        self.mutation_prob = 0.02    # per-individual patch-application probability
        self.use_mcg = use_mcg

        # how strongly we perturb latent codes to sample multiple z's
        self.latent_sigma = 1.0

        # fraction of population used as candidate parents for "most different" pairing
        self.top_parent_frac = 0.5   # top 50% by fitness

        # thresh used outside via logging / MCG gating (still 0 by default)
        self.bad_mcg_thresh = 0e-2   # used in attack() with 1 - p(true/target)

        # -------------------------
        # Patch configuration
        # -------------------------
        # We mimic Square Attack: 1 square per mutation, side as fraction of image.
        if dataset_name in ["imagenet", "openimage"]:
            # ImageNet-like resolution (224x224)
            self.mutation_num_patches = 0
            self.crossover_num_patches = 5
            self.patch_min_frac = 0.1   # 5% of side
            self.patch_max_frac = 0.8   # 25% of side
        else:
            # CIFAR etc. (lower res, can afford relatively larger patches)
            self.mutation_num_patches = 1
            self.crossover_num_patches = 1
            self.patch_min_frac = 0.15   # 15% of side
            self.patch_max_frac = 0.40   # 40% of side

    # --------------------
    # Latent helpers
    # --------------------
    def _sample_latent_like(self, latent):
        """
        Given a latent structure (tensor or list/tuple of tensors),
        return a new latent sample: latent + N(0, sigma^2 I).
        """
        if latent is None:
            return None

        sigma = self.latent_sigma

        if isinstance(latent, torch.Tensor):
            return latent + sigma * torch.randn_like(latent)

        elif isinstance(latent, (list, tuple)):
            new_list = []
            for z in latent:
                if isinstance(z, torch.Tensor):
                    new_list.append(z + sigma * torch.randn_like(z))
                else:
                    new_list.append(z)
            return type(latent)(new_list)

        return latent

    # --------------------
    # Seeding population
    # --------------------
    def _seed_population(self, x0, init, generate_function, latent):
        """
        Seed a population of adversarial images in [0,1] with L_inf <= eps from x0.

        x0: (1,C,H,W)
        returns: pop (P,C,H,W)

        Strategy:
        - If use_mcg and generate_function/latent available:
            use MULTIPLE latent samples:
              for i in 0..P-1:
                  latent_i = latent (for i=0) or latent + noise (for i>0)
                  delta_i  = generate_function(x0, latent_i)
                  adv_i    = clamp(x0 + delta_i, [0,1] and L_inf <= eps)
        - If that fails or use_mcg=False: fall back to random L_inf deltas.
        """
        device = x0.device
        x0 = x0.detach()

        pop = []

        # --- MCG-based multi-z seeding, if available ---
        can_use_mcg = self.use_mcg and (generate_function is not None) and (latent is not None)

        if can_use_mcg:
            try:
                for i in range(self.pop_size):
                    if i == 0 and init is not None:
                        # if we have an external init (e.g. from square), use that as first
                        adv_i = torch.clamp(x0 + init.view_as(x0), 0.0, 1.0)
                    else:
                        if i == 0:
                            latent_i = latent
                        else:
                            latent_i = self._sample_latent_like(latent)

                        with torch.no_grad():
                            delta_i = generate_function(x0, latent_i).view_as(x0)

                        delta_i = torch.clamp(delta_i, -self.linf, self.linf)
                        adv_i = torch.clamp(x0 + delta_i, 0.0, 1.0)

                    x0_img = x0.squeeze(0)
                    adv_i_img = adv_i.squeeze(0)
                    adv_i_img = torch.max(
                        torch.min(adv_i_img, x0_img + self.linf),
                        x0_img - self.linf
                    )
                    adv_i_img = torch.clamp(adv_i_img, 0.0, 1.0)
                    pop.append(adv_i_img)
            except Exception:
                pop = []

        # --- Fallback: pure random seeds in L_inf ball ---
        if len(pop) == 0:
            for _ in range(self.pop_size):
                delta = (2 * torch.rand_like(x0) - 1.0) * self.linf
                delta = torch.clamp(delta, -self.linf, self.linf)
                adv = torch.clamp(x0 + delta, 0.0, 1.0)

                x0_img = x0.squeeze(0)
                adv_img = adv.squeeze(0)
                adv_img = torch.max(
                    torch.min(adv_img, x0_img + self.linf),
                    x0_img - self.linf
                )
                adv_img = torch.clamp(adv_img, 0.0, 1.0)
                pop.append(adv_img)

        return torch.stack(pop, dim=0).to(device)

    # --------------------
    # Fitness & selection
    # --------------------
    def _fitness(self, pop, y_int, loss_func):
        """
        pop: (P,C,H,W)
        y_int: int
            - untargeted: true label
            - targeted:   target label (your driver overwrites labels)
        loss_func: black-box loss function from base_attack.margin_loss_interface

        returns:
          scores: np.array[P]          (score = -margin)
          success_flags: np.array[P]   (True iff margin <= 0)
          logits_list: list of tensors (probs per individual)
        """
        scores = []
        successes = []
        logits_list = []

        for img in pop:
            x_adv = img.unsqueeze(0)  # (1,C,H,W)
            out = loss_func(x_adv, y_int, targeted=self.targeted)

            # out["logits"] from your margin_loss_interface are already softmax probabilities.
            probs = out["logits"][0]  # (class_num,)
            logits_list.append(probs.detach())

            # Compute a clean, unified margin from probabilities
            if not self.targeted:
                # UNTARGETED: margin = p(true) - max_{k≠true} p(k)
                label = y_int
                p_label = probs[label]
                tmp = probs.clone()
                tmp[label] = -1.0  # probs in [0,1], so -1 removes it from max
                p_max_other = torch.max(tmp)
                margin_val = p_label - p_max_other
            else:
                # TARGETED: margin = max_{k≠target} p(k) - p(target)
                target = y_int
                p_target = probs[target]
                tmp = probs.clone()
                tmp[target] = -1.0
                p_max_other = torch.max(tmp)
                margin_val = p_max_other - p_target

            margin = float(margin_val.item())

            success_flag = (margin <= 0.0)
            score = -margin  # GA maximizes -margin

            scores.append(score)
            successes.append(success_flag)

        scores = np.array(scores, dtype=np.float32)
        successes = np.array(successes, dtype=bool)
        return scores, successes, logits_list

    def _tournament_idx(self, scores):
        """
        Tournament selection index.
        """
        n = len(scores)
        cand = np.random.choice(n, size=min(self.tourn_k, n), replace=False)
        return int(cand[np.argmax(scores[cand])])

    def _crossover(self, a, b):
        """
        Square-patch crossover in image space.

        a, b: (C,H,W)
        Child = a with one or more square patches copied from b.
        """
        child = a.clone()
        C, H, W = child.shape

        num_patches = self.crossover_num_patches
        min_frac = self.patch_min_frac
        max_frac = self.patch_max_frac
        side_max = min(H, W)

        for _ in range(num_patches):
            # Sample square side length
            side = int(np.random.uniform(min_frac, max_frac) * side_max)
            side = max(1, min(side, side_max))

            # Random top-left position
            top = np.random.randint(0, H - side + 1)
            left = np.random.randint(0, W - side + 1)

            # Copy region from parent b into child
            child[:, top:top + side, left:left + side] = \
                b[:, top:top + side, left:left + side]

        return child

    def _mutate(self, img, x0):
        """
        Square-patch mutation (Square-Attack-like):

        - 1 (or few) square patches per individual (depending on config)
        - patch side ∝ image size
        - random noise in [-linf, linf] in that square
        """
        out = img.clone()
        C, H, W = out.shape

        num_patches = self.mutation_num_patches
        min_frac = self.patch_min_frac
        max_frac = self.patch_max_frac
        side_max = min(H, W)

        for _ in range(num_patches):
            if np.random.rand() >= self.mutation_prob:
                continue

            # Sample square side
            side = int(np.random.uniform(min_frac, max_frac) * side_max)
            side = max(1, min(side, side_max))

            # Random location
            top = np.random.randint(0, H - side + 1)
            left = np.random.randint(0, W - side + 1)

            # Noise in [-linf, linf]
            noise = (2.0 * torch.rand((C, side, side), device=out.device) - 1.0) * self.linf

            out[:, top:top + side, left:left + side] = torch.clamp(
                out[:, top:top + side, left:left + side] + noise,
                0.0,
                1.0,
            )

        # Project entire image back into the L_inf ball around x0
        x0_img = x0.squeeze(0)
        out = torch.max(torch.min(out, x0_img + self.linf), x0_img - self.linf)
        out = torch.clamp(out, 0.0, 1.0)

        return out

    def _pair_most_different(self, pop, indices):
        """
        Given population tensor pop (P,C,H,W) and a list/array of indices,
        greedily form pairs of indices that are maximally different in L2.

        Each index is used at most once.
        Returns: list of (i, j) tuples (indices into pop).
        """
        if len(indices) < 2:
            return []

        device = pop.device

        flats = {}
        for idx in indices:
            v = pop[int(idx)].view(-1).to(device)
            flats[int(idx)] = v

        used = set()
        pairs = []

        dists = {}
        idx_list = [int(i) for i in indices]
        for i_idx in range(len(idx_list)):
            for j_idx in range(i_idx + 1, len(idx_list)):
                i = idx_list[i_idx]
                j = idx_list[j_idx]
                d = torch.norm(flats[i] - flats[j], p=2).item()
                dists[(i, j)] = d

        while len(used) < len(idx_list):
            best_pair = None
            best_dist = -1.0
            for (i, j), d in dists.items():
                if i in used or j in used:
                    continue
                if d > best_dist:
                    best_dist = d
                    best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            used.add(i)
            used.add(j)
            pairs.append((i, j))

        return pairs

    # --------------------
    # Main attack
    # --------------------
    def attack(self, loss_func, x, y, init=None, buffer=None,
               generate_function=None, latent=None, **kwargs):
        """
        GA main loop for one image.

        :param loss_func: callable (x_adv, label, targeted)-> dict with margin, logits, loss
        :param x: benign image, tensor shape (1, C, H, W)
        :param y: label (int or tensor)
                  - untargeted: true class
                  - targeted:   target class (your driver overwrites labels)
        :param init: initial perturbation (from MCG/Square) or None
        :param buffer: AttackListBuffer (unused here)
        :param generate_function: C-Glow generate_function, or None (used to seed with multiple z samples)
        :param latent: latent code for current image, or None (used to sample multiple z)
        """
        device = x.device
        x0 = x.detach().to(device)
        y_int = int(y) if not torch.is_tensor(y) else int(y.item())

        # Seed population with multi-z MCG samples if possible
        pop = self._seed_population(x0, init, generate_function, latent)

        best_img = None
        best_score = -1e9
        best_logits = None
        query_cnt = 0

        max_gens = max(self.max_query // self.pop_size, 1)

        for g in range(max_gens):
            if query_cnt >= self.max_query:
                break

            scores, succ_flags, logits_list = self._fitness(pop, y_int, loss_func)
            query_cnt += len(pop)

            # --- Initial population logging + MCG gating ---
            if g == 0:
                # logits_list entries are already probabilities
                prob_scores = []
                for probs in logits_list:
                    # For untargeted: 1 - p(true)
                    # For targeted:   1 - p(target)
                    prob_scores.append(1.0 - probs[y_int].item())
                prob_scores = np.array(prob_scores, dtype=np.float32)

                order_prob = np.argsort(-prob_scores)
                print("\n=== Initial Population Fitness (Sorted) ===")
                for rank, idx in enumerate(order_prob, start=1):
                    idx = int(idx)
                    score_print = prob_scores[idx]
                    mcg_tag = " [MCG seed]" if idx == 0 else ""
                    print(
                        f"Rank {rank:2d} | Idx {idx:2d} | "
                        f"Score {score_print:.4f}{mcg_tag}"
                    )
                print("==========================================")

                best_init_prob_score = float(np.max(prob_scores))
                if (not self.targeted) and best_init_prob_score < self.bad_mcg_thresh:
                    # For untargeted: if MCG is clearly useless (1 - p(true) tiny), discard and reseed randomly
                    print(
                        f"[INFO] Initial best score {best_init_prob_score:.4e} "
                        f"< bad_mcg_thresh={self.bad_mcg_thresh:.2e} "
                        f"→ discarding MCG and reseeding population randomly."
                    )
                    pop = self._seed_population(x0, init=None,
                                                generate_function=None,
                                                latent=None)
                    scores, succ_flags, logits_list = self._fitness(pop, y_int, loss_func)
                    query_cnt += len(pop)

                    prob_scores = []
                    for probs in logits_list:
                        prob_scores.append(1.0 - probs[y_int].item())
                    prob_scores = np.array(prob_scores, dtype=np.float32)

                    order_prob = np.argsort(-prob_scores)
                    print("\n=== Initial Population Fitness (Reseeded, Random) ===")
                    for rank, idx in enumerate(order_prob, start=1):
                        idx = int(idx)
                        score_print = prob_scores[idx]
                        print(
                            f"Rank {rank:2d} | Idx {idx:2d} | Score {score_print:.4f}"
                        )
                    print("======================================================")

            # rank by GA score (-margin)
            order = np.argsort(-scores)
            best_idx = int(order[0])

            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_img = pop[best_idx].detach().clone()
                best_logits = logits_list[best_idx].detach()

            if succ_flags.any():
                win = int(np.where(succ_flags)[0][0])
                adv = pop[win].detach().clone()
                logits_win = logits_list[win].detach()
                print(
                    f"[DEBUG] Finished with success. "
                    f"best_init_score={scores[best_idx]:.4f}, "
                    f"best_final_score={best_score:.4f}"
                )
                return {
                    "success": True,
                    "query_cnt": min(query_cnt, self.max_query),
                    "adv": adv.unsqueeze(0),
                    "logits_best": logits_win.unsqueeze(0),
                }

            # --- Next generation construction ---

            new_pop = []

            # 1) Always carry over the best individual
            new_pop.append(pop[best_idx].detach().clone())

            # 2) Additional elites
            elite_count = min(self.num_elite + 1, len(order))
            elite_indices = [int(i) for i in order[1:elite_count]]
            for idx in elite_indices:
                if len(new_pop) >= self.pop_size:
                    break
                new_pop.append(pop[idx].detach().clone())

            # 3) Pair most different individuals among top_parent_frac
            top_k = max(2, int(self.pop_size * self.top_parent_frac))
            top_k = min(top_k, len(order))
            top_indices = order[:top_k]
            pairs = self._pair_most_different(pop, top_indices)

            for (i, j) in pairs:
                if len(new_pop) >= self.pop_size:
                    break
                p1 = pop[int(i)]
                p2 = pop[int(j)]

                if np.random.rand() < self.crossover_prob:
                    child1 = self._crossover(p1, p2)
                else:
                    child1 = p1.clone()
                child1 = self._mutate(child1, x0)
                new_pop.append(child1)

                if len(new_pop) >= self.pop_size:
                    break

                if np.random.rand() < self.crossover_prob:
                    child2 = self._crossover(p2, p1)
                else:
                    child2 = p2.clone()
                child2 = self._mutate(child2, x0)
                new_pop.append(child2)

                if len(new_pop) >= self.pop_size:
                    break

            # 4) Fill remainder with tournament-based GA
            while len(new_pop) < self.pop_size:
                a_idx = self._tournament_idx(scores)
                b_idx = self._tournament_idx(scores)
                a = pop[a_idx]
                b = pop[b_idx]

                if np.random.rand() < self.crossover_prob:
                    child = self._crossover(a, b)
                else:
                    child = a.clone()

                child = self._mutate(child, x0)
                new_pop.append(child)

            pop = torch.stack(new_pop, dim=0).to(device)

        # budget exhausted, no success
        if best_img is None:
            best_img = pop[0].detach().clone()
            best_logits = logits_list[0].detach() if len(logits_list) > 0 else None

        print(
            f"[DEBUG] Finished with FAILURE. "
            f"best_init_score={scores[order[0]]:.4f}, "
            f"best_final_score={best_score:.4f}"
        )
        return {
            "success": False,
            "query_cnt": min(query_cnt, self.max_query),
            "adv": best_img.unsqueeze(0),
            "logits_best": best_logits.unsqueeze(0) if best_logits is not None else None,
        }
