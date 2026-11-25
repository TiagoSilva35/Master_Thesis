"""Step 5: the outcome Y."""

from src.causal_model.nodes.binary import BinaryNode


class OutcomeNode(BinaryNode):
    """The outcome Y: 1 is the favourable outcome, 0 is not.

    Y is a yes/no node (BinaryNode): its equation gives a chance,

        P(Y = 1) = σ(constant + Σ curve_parent(x_parent) + Σ surfaces),   σ(s) = 1 / (1 + e^-s)

    and it never has children. Y is not known for the person we ask about, so
    there is no noise to recover (`noise` is 0). It keeps only the tiny ridge
    of its fitting (no log-odds belief), so its evidence is the one the drift
    code has always used.
    """
