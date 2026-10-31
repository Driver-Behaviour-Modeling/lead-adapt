"""Fit scoring temperatures and metric smoothing on normal validation data."""

import math

import torch
from scipy.optimize import minimize_scalar

from lead.scoring.distributions import MotionSmoother


class ObservedLikelihood:
    """Evaluate observed-token NLL without computing every smoothed output."""

    def __init__(self, member_logits, tokens, centroids, heading_weight):
        self.logits = torch.as_tensor(member_logits, dtype=torch.float64).detach().cpu()
        self.tokens = torch.as_tensor(tokens).detach().cpu()
        if (
            self.logits.ndim != 3
            or min(self.logits.shape) < 1
            or not torch.isfinite(self.logits).all()
        ):
            raise ValueError("Expected finite [members, samples, vocabulary] logits")
        if (
            self.tokens.dtype != torch.int64
            or self.tokens.shape != self.logits.shape[1:2]
            or not ((self.tokens >= 0) & (self.tokens < self.logits.shape[-1])).all()
        ):
            raise ValueError("Expected valid int64 observed tokens for every sample")
        self.centroids = torch.as_tensor(centroids, dtype=torch.float64).cpu()
        if len(self.centroids) != self.logits.shape[-1]:
            raise ValueError("Vocabulary does not match logits")
        self.heading_weight = heading_weight
        self.kernels = {}

    def observed(self, member, temperature, epsilon, sigma):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        log_p = (self.logits[member] / temperature).log_softmax(-1)
        observed = log_p.gather(-1, self.tokens[:, None]).squeeze(-1)
        if epsilon == 0:
            return observed
        if not 0 < epsilon < 1:
            raise ValueError("Smoothing epsilon must be in [0, 1)")
        if sigma not in self.kernels:
            smoother = MotionSmoother(
                self.centroids,
                self.heading_weight,
                epsilon,
                sigma,
            )
            self.kernels[sigma] = smoother.log_kernel[self.tokens]
        blurred = torch.logsumexp(log_p + self.kernels[sigma], dim=-1)
        return torch.logaddexp(
            observed + math.log1p(-epsilon),
            blurred + math.log(epsilon),
        )

    @staticmethod
    def nll(observed):
        return float(-(observed.logsumexp(0) - math.log(len(observed))).mean())

    def evaluate(self, temperatures, epsilon, sigma):
        return torch.stack(
            [self.observed(m, t, epsilon, sigma) for m, t in enumerate(temperatures)],
        )


def fit_parameters(
    member_logits,
    tokens,
    centroids,
    heading_weight,
    *,
    temperature_bounds=(0.25, 8.0),
    epsilons=(0.0, 0.001, 0.01, 0.05, 0.1),
    sigmas=(0.025, 0.05, 0.1, 0.25, 0.5),
    rounds=2,
):
    """Alternate bounded temperature fits and a smoothing grid by mixture NLL.

    Each trial must improve the validation objective before being accepted.
    This is a coordinate search, not a guarantee of a global joint optimum.
    Temperature/smoothing affect the reference distribution, never control.
    """
    lower, upper = temperature_bounds
    if not 0 < lower < upper or not math.isfinite(upper):
        raise ValueError("Temperature bounds must be finite and positive")
    if (
        not epsilons
        or not sigmas
        or any(not math.isfinite(e) or not 0 <= e < 1 for e in epsilons)
        or any(not math.isfinite(s) or s <= 0 for s in sigmas)
        or type(rounds) is not int
        or rounds < 1
    ):
        raise ValueError("Invalid smoothing grid or number of search rounds")
    likelihood = ObservedLikelihood(member_logits, tokens, centroids, heading_weight)
    temperatures = [min(upper, max(lower, 1.0))] * len(likelihood.logits)
    epsilon, sigma = 0.0, sigmas[0]
    observed = likelihood.evaluate(temperatures, epsilon, sigma)
    initial_nll = likelihood.nll(observed)
    history = []
    for iteration in range(rounds):
        for member in range(len(temperatures)):

            def objective(
                log_t,
                member=member,
                observed=observed,
                epsilon=epsilon,
                sigma=sigma,
            ):
                trial = observed.clone()
                trial[member] = likelihood.observed(
                    member,
                    math.exp(log_t),
                    epsilon,
                    sigma,
                )
                return likelihood.nll(trial)

            result = minimize_scalar(
                objective,
                method="bounded",
                bounds=(math.log(lower), math.log(upper)),
                options={"xatol": 1e-4},
            )
            if result.success and result.fun < likelihood.nll(observed):
                temperatures[member] = math.exp(float(result.x))
                observed[member] = likelihood.observed(
                    member,
                    temperatures[member],
                    epsilon,
                    sigma,
                )
        for candidate_e in epsilons:
            for candidate_s in sigmas[:1] if candidate_e == 0 else sigmas:
                trial = likelihood.evaluate(temperatures, candidate_e, candidate_s)
                if likelihood.nll(trial) < likelihood.nll(observed):
                    epsilon, sigma, observed = candidate_e, candidate_s, trial
        history.append({"round": iteration, "nll": likelihood.nll(observed)})
    return {
        "temperatures": temperatures,
        "epsilon": float(epsilon),
        "sigma": float(sigma),
        "initial_nll": initial_nll,
        "fitted_nll": likelihood.nll(observed),
        "search": {
            "method": "Coordinate temperature optimization and smoothing grid; uniform frame-weighted mixture NLL on normal validation drives only.",
            "temperature_bounds": list(temperature_bounds),
            "epsilons": list(epsilons),
            "sigmas": list(sigmas),
            "rounds": rounds,
            "history": history,
        },
    }
