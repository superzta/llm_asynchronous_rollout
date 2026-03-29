class StalenessDecision:
    def __init__(self, staleness, accepted, bound_k):
        self.staleness = staleness
        self.accepted = accepted
        self.bound_k = bound_k


def compute_staleness(current_policy_version, sample_policy_version):
    value = current_policy_version - sample_policy_version
    return max(0, value)


def bounded_staleness_accept(current_policy_version, sample_policy_version, bound_k):
    staleness = compute_staleness(current_policy_version, sample_policy_version)
    return StalenessDecision(staleness=staleness, accepted=staleness <= bound_k, bound_k=bound_k)
