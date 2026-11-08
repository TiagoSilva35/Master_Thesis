"""Step 4: counterfactuals."""

from src.causal_model.kan import CausalKAN


class Counterfactual:
    """Answers: what would this person's values have been if we had set some of them?

    Three steps:
      1. abduction:  recover each variable's noise, what makes this person
                     different from the average person with the same parents;
      2. action:     replace the equations of the variables we set by the new values;
      3. prediction: recompute every other variable in causal order, keeping
                     the person's own noise.
    """

    def __init__(self, model: CausalKAN):
        self.model = model

    def of(self, person: dict[str, float], action: dict[str, float]) -> dict[str, float]:
        noise = self.abduction(person)
        return self.prediction(noise, action)

    def abduction(self, person: dict[str, float]) -> dict[str, float]:
        return {node.name: node.noise(person) for node in self.model.nodes}

    def prediction(self, noise: dict[str, float], action: dict[str, float]) -> dict[str, float]:
        values = {}
        for node in self.model.nodes:
            if node.name in action:  # the action step
                values[node.name] = float(action[node.name])
            else:
                values[node.name] = float(node.predict(values) + noise[node.name])
        return values
