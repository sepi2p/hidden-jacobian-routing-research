import numpy as np
import torch
import torch.nn.functional as F

from .base_attack import BaseAttack
from .square import SquareAttack


def score_from_logits(logits: torch.Tensor, label_idx: int) -> float:
    """
    score = 1 - p_true(label_idx)
    logits: [B, C] softmax probabilities
    label_idx: int
    """
    p_true = logits[:, label_idx].item()
    return 1.0 - float(p_true)


def get_label_idx(label: int, args) -> int:
    """Return scalar label index according to targeted/untargeted."""
    if getattr(args, "targeted", False):
        return args.target_label
    return int(label)


class SurrogatePGDCandidateGenerator:
    """
    Multi-start PGD on surrogate with multiple loss functions to generate
    strong and diverse candidate perturbations under L_inf.

    We still *rank* candidates by a unified score: score = 1 - p_true_surrogate,
    but individual restarts can use different loss modes to explore.

    Typical usage:
        gen = SurrogatePGDCandidateGenerator(linf=0.05, num_restarts=32, ...)
        deltas_topk, scores_topk = gen.generate(
            x, y, delta_init, surrogate, args, top_k=20
        )
    """

    def __init__(
        self,
        linf: float,
        num_restarts: int = 32,
        pgd_steps: int = 40,
        step_size: float = 0.01,
        loss_modes=None,
        device: str = "cuda",
        diversity_cosine_threshold: float = 0.2,
    ):
        """
        linf: L_inf budget
        num_restarts: how many PGD restarts
        pgd_steps: steps per restart
        step_size: PGD step size
        loss_modes: list of loss names to cycle through,
                    e.g. ["ce", "margin", "cw"]
        diversity_cosine_threshold: minimum cosine distance between
                                    selected candidates (for diversity)
        """
        self.linf = linf
        self.num_restarts = num_restarts
        self.pgd_steps = pgd_steps
        self.step_size = step_size
        self.device = device
        self.diversity_cosine_threshold = diversity_cosine_threshold

        if loss_modes is None:
            loss_modes = ["ce", "margin", "cw"]
        self.loss_modes = loss_modes

    def _surrogate_loss(
        self,
        logits: torch.Tensor,
        loss_label: int,
        loss_mode: str,
        targeted: bool,
    ) -> torch.Tensor:
        """
        Compute a scalar loss on surrogate logits for one image [1,C].

        logits: [1, C]
        loss_label:
            - untargeted: true label
            - targeted:   target label
        loss_mode: "ce", "margin", "cw", etc.
        """
        # logits: [1, C]
        if loss_mode == "ce":
            # CE on loss_label
            log_probs = F.log_softmax(logits, dim=1)
            loss = -log_probs[0, loss_label]
            return loss

        # probabilities
        probs = F.softmax(logits, dim=1)  # [1, C]
        p_true_like = probs[0, loss_label]

        if loss_mode == "margin":
            # margin in prob-space
            probs_clone = probs.clone()
            probs_clone[0, loss_label] = -1.0
            p_other_max = probs_clone.max(dim=1)[0]
            if not targeted:
                # untargeted: p_other_max - p_true (we want large)
                margin = p_other_max - p_true_like
            else:
                # targeted: p_true(target) - p_other_max (we want large)
                margin = p_true_like - p_other_max
            loss = -margin  # maximize margin => minimize negative
            return loss

        if loss_mode == "cw":
            # CW-style margin on logits (not probs)
            logit_true_like = logits[0, loss_label]
            logits_clone = logits.clone()
            logits_clone[0, loss_label] = -1e9
            logit_other_max = logits_clone.max(dim=1)[0]

            kappa = 0.0
            if not targeted:
                # untargeted CW: max(logit_other - logit_true, -kappa)
                margin = logit_other_max - logit_true_like
            else:
                # targeted CW: max(logit_true - logit_other, -kappa)
                margin = logit_true_like - logit_other_max
            loss = -torch.clamp(margin, min=-kappa)
            return loss

        # default fallback: same as CE
        log_probs = F.log_softmax(logits, dim=1)
        loss = -log_probs[0, loss_label]
        return loss

    @torch.no_grad()
    def _score_surrogate(
        self,
        surrogate: torch.nn.Module,
        x: torch.Tensor,
        delta: torch.Tensor,
        label_idx: int,
    ) -> float:
        """
        score = 1 - p_true_surrogate for one candidate delta.
        x: [1,C,H,W], delta: [1,C,H,W]
        """
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        logits = F.softmax(surrogate(x_adv), dim=1)
        p_true = logits[:, label_idx].item()
        return 1.0 - float(p_true)

    def generate(
        self,
        x: torch.Tensor,
        y: int,
        delta_init: torch.Tensor,
        surrogate: torch.nn.Module,
        args,
        top_k: int = 20,
    ):
        """
        Run multi-start PGD on surrogate with multiple loss modes.

        Returns:
            deltas_topk: [K, C, H, W] tensor of top-K diverse deltas
        scores_topk: [K] tensor of corresponding surrogate scores (1 - p_true_sur)
        """
        checkpoints = self.generate_checkpoints(
            x=x,
            y=y,
            delta_init=delta_init,
            surrogate=surrogate,
            args=args,
            top_k=top_k,
            checkpoint_steps=[self.pgd_steps],
        )
        deltas_topk, scores_topk = checkpoints[int(self.pgd_steps)]
        print(f"[SurPGD] Generated {self.num_restarts} candidates, "
              f"selected {deltas_topk.shape[0]} diverse top-K.")
        return deltas_topk, scores_topk

    def _select_diverse_topk(self, all_deltas, all_scores, top_k):
        # sort by score descending
        sort_idx = torch.argsort(all_scores, descending=True)
        all_deltas = all_deltas[sort_idx]
        all_scores = all_scores[sort_idx]

        # diversity-aware top-k selection (greedy by cosine distance)
        selected_deltas = []
        selected_scores = []

        flat = all_deltas.reshape(all_deltas.size(0), -1)
        flat_norm = F.normalize(flat, dim=1)

        for i in range(all_deltas.size(0)):
            if len(selected_deltas) >= top_k:
                break

            d_i = flat_norm[i]
            if len(selected_deltas) == 0:
                selected_deltas.append(all_deltas[i])
                selected_scores.append(all_scores[i])
                continue

            sims = []
            for d_sel in selected_deltas:
                d_sel_flat = d_sel.reshape(-1)
                d_sel_flat = F.normalize(d_sel_flat.unsqueeze(0), dim=1)[0]
                sims.append(torch.dot(d_i, d_sel_flat).item())
            max_sim = max(sims)

            # require cosine similarity < 1 - diversity_threshold
            if max_sim < (1.0 - self.diversity_cosine_threshold):
                selected_deltas.append(all_deltas[i])
                selected_scores.append(all_scores[i])

        if len(selected_deltas) == 0:
            # fallback: at least best one
            selected_deltas.append(all_deltas[0])
            selected_scores.append(all_scores[0])

        deltas_topk = torch.stack(selected_deltas, dim=0)   # [K,C,H,W]
        scores_topk = torch.stack(selected_scores, dim=0)   # [K]
        return deltas_topk, scores_topk

    def generate_checkpoints(
        self,
        x: torch.Tensor,
        y: int,
        delta_init: torch.Tensor,
        surrogate: torch.nn.Module,
        args,
        top_k: int = 20,
        checkpoint_steps=None,
    ):
        """
        Run one multi-start PGD trajectory and independently select top-K
        candidates at requested checkpoint steps.

        This is for ablations: step-5 and step-10 candidates are selected from
        the full restart pool at those exact steps, not from the final step's
        selected candidates.

        Returns:
            dict[int, tuple[deltas_topk, scores_topk]]
        """
        device = x.device
        targeted = getattr(args, "targeted", False)
        label_idx = get_label_idx(y, args)
        loss_label = args.target_label if targeted else int(y)

        x = x.to(device)
        delta_init = delta_init.to(device)
        surrogate.to(device)
        surrogate.eval()

        _, C, H, W = x.shape

        if checkpoint_steps is None:
            checkpoint_steps = [self.pgd_steps]
        checkpoint_steps = sorted({int(s) for s in checkpoint_steps if int(s) > 0})
        if not checkpoint_steps:
            checkpoint_steps = [self.pgd_steps]
        max_step = max(checkpoint_steps)
        if max_step > self.pgd_steps:
            raise ValueError(f"checkpoint step {max_step} exceeds pgd_steps={self.pgd_steps}")

        checkpoint_deltas = {step: [] for step in checkpoint_steps}
        checkpoint_scores = {step: [] for step in checkpoint_steps}

        # print(f"[SurPGD] Generating candidates on surrogate with "
        #       f"num_restarts={self.num_restarts}, pgd_steps={self.pgd_steps}, "
        #       f"loss_modes={self.loss_modes}")

        for r in range(self.num_restarts):
            loss_mode = self.loss_modes[r % len(self.loss_modes)]

            # init delta: start from init + smaller noise
            delta = delta_init.clone()
            noise = torch.empty_like(delta).uniform_(-self.linf, self.linf)
            delta = torch.clamp(delta + 0.25 * noise, -self.linf, self.linf)
            delta.requires_grad_(True)

            for step in range(max_step):
                x_adv = torch.clamp(x + delta, 0.0, 1.0)
                logits = surrogate(x_adv)

                loss = self._surrogate_loss(
                    logits=logits,
                    loss_label=loss_label,
                    loss_mode=loss_mode,
                    targeted=targeted,
                )

                # gradient ascent on loss (maximize)
                loss.backward()
                with torch.no_grad():
                    grad = delta.grad.sign()
                    delta += self.step_size * grad
                    delta.clamp_(-self.linf, self.linf)
                delta.grad.zero_()

                step_num = step + 1
                if step_num in checkpoint_deltas:
                    with torch.no_grad():
                        score_sur = self._score_surrogate(
                            surrogate, x, delta, label_idx
                        )
                    checkpoint_deltas[step_num].append(delta.detach().clone())
                    checkpoint_scores[step_num].append(score_sur)

                if step == 0 or step == max_step - 1 or step_num % 10 == 0:
                    with torch.no_grad():
                        score_sur = self._score_surrogate(surrogate, x, delta, label_idx)
                    # print(
                    #     f"[SurPGD] restart={r}, step={step+1}, loss_mode={loss_mode}, "
                    #     f"score_sur={score_sur:.4f}"
                    # )

        checkpoints = {}
        for step in checkpoint_steps:
            all_deltas = torch.stack(checkpoint_deltas[step], dim=0)   # [R, 1, C, H, W]
            R = all_deltas.size(0)
            all_deltas = all_deltas.reshape(R, C, H, W)
            all_scores = torch.tensor(checkpoint_scores[step], device=device)
            checkpoints[step] = self._select_diverse_topk(all_deltas, all_scores, top_k)
        return checkpoints


class SquarePatchGA:
    """
    GA that mutates square patches of the perturbation (Imagenet-friendly).

    Individuals: delta in [-linf, linf] with shape (C,H,W).
    Adv image: clamp(x + delta, 0, 1).

    For surrogate stages:
        - fitness = score_surrogate = 1 - p_true_sur
        - run until all individuals have score >= threshold
          (or max generations).
        - then check best individual (by surrogate score) on target model.

    Final stage:
        - fitness on target model (same score definition + success by margin).
    """

    def __init__(
        self,
        dataset_name: str,
        linf: float,
        max_query: int,
        popsize: int = 20,
        patch_min: int = 8,
        patch_max: int = 64,
        elite_frac: float = 0.25,
        mutation_per_child: int = 1,
        device: str = "cuda",
    ):
        self.dataset_name = dataset_name
        self.linf = linf
        self.max_query = max_query
        self.popsize = popsize
        self.patch_min = patch_min
        self.patch_max = patch_max
        self.elite_frac = elite_frac
        self.mutation_per_child = mutation_per_child
        self.device = device

    # ---- internal ops ----

    def _init_population(self, x: torch.Tensor, delta_init: torch.Tensor) -> torch.Tensor:
        """
        x: [1, C, H, W]
        delta_init: [1, C, H, W] initial perturbation (e.g. from MCG)
        Returns population: [P, C, H, W]
        """
        pop = delta_init.repeat(self.popsize, 1, 1, 1)
        noise = torch.empty_like(pop).uniform_(-self.linf, self.linf)
        pop = pop + noise
        pop = torch.clamp(pop, -self.linf, self.linf)
        print(
            f"[HybridGA][GA] Initialized population with popsize={self.popsize}, "
            f"linf={self.linf}, patch_min={self.patch_min}, patch_max={self.patch_max}"
        )
        return pop

    def _mutate_square_patch(self, delta: torch.Tensor) -> None:
        """
        delta: [C, H, W] (single individual)
        In-place mutation on a random square patch.
        """
        c, h, w = delta.shape
        side = np.random.randint(self.patch_min, self.patch_max + 1)
        side = max(1, min(side, min(h, w)))
        top = np.random.randint(0, h - side + 1)
        left = np.random.randint(0, w - side + 1)

        # random values in [-linf, linf]
        patch = torch.empty((c, side, side), device=delta.device).uniform_(
            -self.linf, self.linf
        )
        delta[:, top:top + side, left:left + side] = patch

    def _evolve_population(
        self,
        population: torch.Tensor,
        fitness_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        One GA step using elitism + mutation-only reproduction.
        population: [P, C, H, W]
        fitness_scores: [P] higher is better
        """
        P = population.shape[0]
        elite_k = max(1, int(self.elite_frac * P))

        sorted_idx = torch.argsort(fitness_scores, descending=True)
        elites = population[sorted_idx[:elite_k]].clone()

        new_pop = [elites]

        # generate rest by mutating random elites
        for _ in range(P - elite_k):
            parent_idx = np.random.randint(0, elite_k)
            child = elites[parent_idx].clone()
            for _m in range(self.mutation_per_child):
                self._mutate_square_patch(child)
            child = torch.clamp(child, -self.linf, self.linf)
            new_pop.append(child.unsqueeze(0))

        population = torch.cat(new_pop, dim=0)
        return population

    # ---- evaluation helpers ----

    @torch.no_grad()
    def _eval_surrogate_scores(
        self,
        surrogate: torch.nn.Module,
        x: torch.Tensor,
        population: torch.Tensor,
        label_idx: int,
    ) -> torch.Tensor:
        """
        surrogate: model
        x: [1,C,H,W]
        population: [P,C,H,W]
        Returns:
            scores: [P] (1 - p_true_surrogate)
        """
        P = population.shape[0]
        x_rep = x.repeat(P, 1, 1, 1)
        x_adv = torch.clamp(x_rep + population, 0.0, 1.0)
        logits = torch.nn.functional.softmax(surrogate(x_adv), dim=1)
        scores = 1.0 - logits[:, label_idx]
        return scores

    @torch.no_grad()
    def _eval_target_scores_and_margins(
        self,
        loss_func,
        x: torch.Tensor,
        y: int,
        population: torch.Tensor,
        label_idx: int,
        targeted: bool,
    ):
        """
        loss_func: margin_loss_interface(T, class_num=...)
        x: [1,C,H,W]
        population: [P,C,H,W]
        Returns:
            margins: [P]
            logits: [P,C] (softmax probs)
            scores: [P] = 1 - p_true_target
        """
        P = population.shape[0]
        x_rep = x.repeat(P, 1, 1, 1)
        x_adv = torch.clamp(x_rep + population, 0.0, 1.0)
        output = loss_func(x_adv, y, targeted)
        margins = output["margin"]
        logits = output["logits"]
        scores = 1.0 - logits[:, label_idx]
        return margins, logits, scores

    # ---- main GA pipeline ----

    def attack(
        self,
        x: torch.Tensor,
        y: int,
        delta_init: torch.Tensor,
        loss_func,
        surrogate: torch.nn.Module,
        args,
        thresholds=(0.8, 0.85, 0.9, 0.95),
        max_gens_per_stage: int = 50,
    ):
        """
        x: [1,C,H,W] clean image
        y: scalar int label
        delta_init: [1,C,H,W]
        loss_func: margin_loss_interface for target model
        surrogate: first surrogate model
        args: parsed args (for targeted flag)
        thresholds: list of surrogate score thresholds
        """
        device = x.device
        label_idx = get_label_idx(y, args)

        print(
            f"[HybridGA][GA] Starting GA attack on image with label={y}, "
            f"thresholds={thresholds}, max_gens_per_stage={max_gens_per_stage}, "
            f"max_query={self.max_query}, popsize={self.popsize}"
        )

        # init population
        population = self._init_population(x, delta_init).to(device)

        total_target_queries = 0
        best_adv = torch.clamp(x + delta_init, 0.0, 1.0)
        best_logits = None
        success = False

        # ---- surrogate stages ----
        for stage_idx, thr in enumerate(thresholds):
            print(
                f"[HybridGA][GA][surrogate] Stage {stage_idx + 1}/{len(thresholds)} "
                f"with threshold={thr:.2f} started."
            )

            for gen in range(max_gens_per_stage):
                scores_sur = self._eval_surrogate_scores(
                    surrogate, x, population, label_idx
                )

                min_s = float(scores_sur.min().item())
                max_s = float(scores_sur.max().item())
                mean_s = float(scores_sur.mean().item())

                if gen == 0 or gen % 5 == 0 or gen == max_gens_per_stage - 1:
                    print(
                        f"[HybridGA][GA][surrogate] Stage {stage_idx + 1}, "
                        f"gen={gen}, scores_sur: min={min_s:.3f}, "
                        f"mean={mean_s:.3f}, max={max_s:.3f}"
                    )

                # if all individuals above thr, stop stage
                if torch.all(scores_sur >= thr):
                    print(
                        f"[HybridGA][GA][surrogate] Stage {stage_idx + 1} reached "
                        f"threshold={thr:.2f} at gen={gen}."
                    )
                    break

                population = self._evolve_population(population, scores_sur)

            # after stage: check best individual (by surrogate score) on target
            scores_sur = self._eval_surrogate_scores(
                surrogate, x, population, label_idx
            )
            min_s = float(scores_sur.min().item())
            max_s = float(scores_sur.max().item())
            mean_s = float(scores_sur.mean().item())
            print(
                f"[HybridGA][GA][surrogate] Stage {stage_idx + 1} finished. "
                f"Final scores_sur: min={min_s:.3f}, mean={mean_s:.3f}, max={max_s:.3f}"
            )

            best_idx = torch.argmax(scores_sur).item()
            best_delta = population[best_idx : best_idx + 1]  # [1,C,H,W]
            x_best = torch.clamp(x + best_delta, 0.0, 1.0)

            # one target query
            if total_target_queries >= self.max_query:
                print(
                    "[HybridGA][GA][surrogate] Reached max_query before target check; "
                    "returning best so far."
                )
                return {
                    "success": False,
                    "query_cnt": total_target_queries,
                    "adv": best_adv,
                    "logits_best": best_logits,
                }

            output = loss_func(x_best, y, args.targeted)
            total_target_queries += 1
            margin = output["margin"]
            logits = output["logits"]
            score_tgt = score_from_logits(logits, label_idx)

            print(
                f"[HybridGA][GA][surrogate] Stage {stage_idx + 1} target check: "
                f"margin={float(margin.item()):.4f}, score_target={score_tgt:.3f}, "
                f"total_target_queries={total_target_queries}"
            )

            if best_logits is None:
                best_adv = x_best
                best_logits = logits

            if margin <= 0:
                # success on target
                success = True
                best_adv = x_best
                best_logits = logits
                print(
                    f"[HybridGA][GA][surrogate] SUCCESS on target at stage "
                    f"{stage_idx + 1} with margin={float(margin.item()):.4f}."
                )
                return {
                    "success": success,
                    "query_cnt": total_target_queries,
                    "adv": best_adv,
                    "logits_best": best_logits,
                }

        # ---- final stage: GA on target model ----
        print("[HybridGA][GA][target] Entering target-model GA stage.")
        target_iter = 0
        while total_target_queries < self.max_query:
            margins, logits_pop, scores_tgt = self._eval_target_scores_and_margins(
                loss_func, x, y, population, label_idx, args.targeted
            )

            total_target_queries += population.shape[0]
            target_iter += 1

            min_margin = float(margins.min().item())
            max_margin = float(margins.max().item())
            best_score = float(scores_tgt.max().item())
            mean_score = float(scores_tgt.mean().item())
            any_success = bool(torch.any(margins <= 0))

            print(
                f"[HybridGA][GA][target] iter={target_iter}, "
                f"margins: min={min_margin:.4f}, max={max_margin:.4f}, "
                f"scores_tgt: mean={mean_score:.3f}, max={best_score:.3f}, "
                f"any_success={any_success}, total_target_queries={total_target_queries}"
            )

            # track best by score (and keep last margins for adv check)
            best_idx = torch.argmax(scores_tgt).item()
            cand_adv = torch.clamp(
                x + population[best_idx : best_idx + 1], 0.0, 1.0
            )
            cand_logits = logits_pop[best_idx : best_idx + 1]

            if best_logits is None or score_from_logits(
                cand_logits, label_idx
            ) > score_from_logits(best_logits, label_idx):
                best_adv = cand_adv
                best_logits = cand_logits

            # check any success
            if any_success:
                success = True
                print(f"[HybridGA][GA][target] SUCCESS on target at iter={target_iter}.")
                return {
                    "success": success,
                    "query_cnt": total_target_queries,
                    "adv": best_adv,
                    "logits_best": best_logits,
                }

            # evolve based on target scores
            population = self._evolve_population(population, scores_tgt)

        print(
            "[HybridGA][GA][target] Finished target GA without success "
            f"(total_target_queries={total_target_queries})."
        )
        return {
            "success": success,
            "query_cnt": total_target_queries,
            "adv": best_adv,
            "logits_best": best_logits,
        }


class HybridGA(BaseAttack):
    """
    Hybrid MCG + Square + (Surrogate-PGD or GA) attack.

    Logic:
      1. Initial MCG image has already been tried in attack.py.
         We receive its perturbation and loss_output.

      2. If initial score > 0.1 -> SquareAttack starting from that perturbation.

      3. If initial score <= 0.1 -> Surrogate-PGD:
         - Multi-start PGD on surrogate with multiple loss modes
         - Rank candidates by score_sur = 1 - p_true_sur
         - Take top-K diverse deltas
         - Check each candidate once on target; stop at first success
         - If none succeed, run a short Square refinement from the best candidate
           and then return the best overall.
    """

    def __init__(
        self,
        dataset_name: str,
        max_query: int,
        targeted: bool,
        class_num: int,
        linf: float = 0.05,
    ):
        super().__init__(dataset_name, max_query, targeted, class_num, linf)

        # Extra budget for SurPGD → Square refinement on target
        self.surpgd_refine_max_queries = 150

    def attack(
        self,
        loss_func,
        x: torch.Tensor,
        y: int,
        init: torch.Tensor = None,
        buffer=None,
        generate_function=None,
        latent=None,
        **kwargs,
    ):
        """
        Parameters follow the same pattern as other attacks.
        Extra kwargs (from attack.py):
            - surrogates: list of surrogate models
            - args: the full args object
            - init_output: dict with 'margin' and 'logits' for initial MCG adv
            - init_score: initial score (1 - p_true) for the MCG adv
        """
        surrogates = kwargs.get("surrogates", None)
        args = kwargs.get("args", None)
        init_output = kwargs.get("init_output", None)
        init_score = kwargs.get("init_score", None)

        # x: [1,C,H,W]; init: [1,C,H,W] perturbation from MCG
        if init is None:
            raise ValueError("HybridGA expects 'init' perturbation from MCG.")

        # Recompute init_output if not provided
        if init_output is None:
            adv_images = torch.clamp(x + init.reshape(x.shape), 0.0, 1.0)
            init_output = loss_func(adv_images, y, getattr(args, "targeted", False))

        logits_init = init_output["logits"]
        margin_init = init_output["margin"]

        if args is not None:
            label_idx = get_label_idx(y, args)
        else:
            label_idx = int(y)

        if init_score is None:
            score_init = score_from_logits(logits_init, label_idx)
        else:
            score_init = float(init_score)

        print(
            f"[HybridGA] Initial MCG result: margin={float(margin_init.item()):.4f}, "
            f"score_init={score_init:.3f}, label={y}, targeted={getattr(args, 'targeted', False)}"
        )

        # Safety: if for some reason it's already adversarial, just return it
        if torch.any(margin_init <= 0):
            print(
                "[HybridGA] Initial MCG already adversarial inside HybridGA.attack; "
                "returning without extra queries."
            )
            return {
                "success": True,
                "query_cnt": 0,  # extra queries beyond the one in attack.py
                "adv": torch.clamp(x + init.reshape(x.shape), 0.0, 1.0),
                "logits_best": logits_init,
            }

        # Branch 2: score > 0.1 -> Square Attack from MCG delta
        if score_init > 0.1:
            print(
                f"[HybridGA] score_init={score_init:.3f} > 0.1 → using SquareAttack "
                "starting from MCG perturbation."
            )
            square = SquareAttack(
                dataset_name=self.dataset,
                max_query=self.max_query,
                targeted=self.targeted,
                class_num=self.class_num,
                linf=self.linf,
            )
            attack_output = square.attack(
                loss_func, x, y, init=init, buffer=buffer
            )
            return attack_output

        # Branch 3: score <= 0.1 -> Surrogate-PGD (multi-loss) then target checks
        if not surrogates or len(surrogates) == 0:
            print(
                "[HybridGA] No surrogate models available, falling back to SquareAttack."
            )
            square = SquareAttack(
                dataset_name=self.dataset,
                max_query=self.max_query,
                targeted=self.targeted,
                class_num=self.class_num,
                linf=self.linf,
            )
            attack_output = square.attack(
                loss_func, x, y, init=init, buffer=buffer
            )
            return attack_output

        print(
            f"[HybridGA] score_init={score_init:.3f} <= 0.1 → running Surrogate-PGD "
            "to generate candidates, then checking on target."
        )
        surrogate = surrogates[0]
        delta_init = init.reshape(x.shape)

        # ---- Surrogate-PGD candidate generation ----
        pgd_gen = SurrogatePGDCandidateGenerator(
            linf=self.linf,
            num_restarts=32,         # tune if you like
            pgd_steps=40,            # tune steps
            step_size=0.01,          # tune step size
            loss_modes=["ce", "margin", "cw"],
            device=x.device.type,
            diversity_cosine_threshold=0.2,
        )

        deltas_topk, scores_topk = pgd_gen.generate(
            x=x,
            y=int(y),
            delta_init=delta_init,
            surrogate=surrogate,
            args=args,
            top_k=20,   # number of candidates to keep
        )

        # ---- Check candidates on target model ----
        total_target_queries = 0
        best_adv = None
        best_logits = None
        success = False

        for i in range(deltas_topk.shape[0]):
            delta_i = deltas_topk[i : i + 1]  # [1,C,H,W]
            x_adv_i = torch.clamp(x + delta_i, 0.0, 1.0)
            output_i = loss_func(x_adv_i, y, getattr(args, "targeted", False))
            total_target_queries += 1

            margin_i = output_i["margin"]
            logits_i = output_i["logits"]
            score_tgt_i = score_from_logits(logits_i, label_idx)

            print(
                f"[HybridGA][SurPGD] candidate {i}, "
                f"margin={float(margin_i.item()):.4f}, score_target={score_tgt_i:.4f}, "
                f"total_target_queries={total_target_queries}"
            )

            if best_logits is None or score_tgt_i > score_from_logits(
                best_logits, label_idx
            ):
                best_adv = x_adv_i
                best_logits = logits_i

            if margin_i <= 0:
                print(
                    f"[HybridGA][SurPGD] SUCCESS on target with candidate {i} "
                    f"(margin={float(margin_i.item()):.4f})."
                )
                success = True
                return {
                    "success": True,
                    "query_cnt": total_target_queries,
                    "adv": best_adv,
                    "logits_best": best_logits,
                }

        # ---- If none of the SurPGD candidates succeeded, try short Square refinement ----
        if best_adv is None:
            # fallback: use the initial MCG perturbation
            best_adv = torch.clamp(x + delta_init, 0.0, 1.0)
            best_logits = logits_init

        print(
            "[HybridGA][SurPGD] No direct success on target from surrogate candidates; "
            "attempting short Square refinement from best candidate."
        )

        # Prepare delta for Square refinement (respect L_inf)
        delta_refine = torch.clamp(best_adv - x, -self.linf, self.linf)

        # Short SquareAttack run with its own (smaller) max_query
        square_refine = SquareAttack(
            dataset_name=self.dataset,
            max_query=self.surpgd_refine_max_queries,
            targeted=self.targeted,
            class_num=self.class_num,
            linf=self.linf,
        )
        sq_out = square_refine.attack(
            loss_func, x, y, init=delta_refine, buffer=buffer
        )

        sq_q = sq_out.get("query_cnt", 0)
        total_queries_all = total_target_queries + sq_q

        if sq_out.get("success", False):
            print(
                "[HybridGA][SurPGD] Square refinement succeeded on target after "
                f"{total_queries_all} total target queries (SurPGD + Square)."
            )
            sq_out["query_cnt"] = total_queries_all
            return sq_out

        # If refinement failed, fall back to best candidate seen so far
        print(
            "[HybridGA][SurPGD] Square refinement failed to find adversarial example; "
            "falling back to best candidate so far."
        )

        best_adv_final = sq_out.get("adv", best_adv)
        best_logits_final = sq_out.get("logits_best", best_logits)

        return {
            "success": False,
            "query_cnt": total_queries_all,
            "adv": best_adv_final,
            "logits_best": best_logits_final,
        }
