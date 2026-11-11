Feature: Uncertainty bands
  A 95% band should contain the true curve about 95% of the time.
  Curves are compared after centring, as the model stores them (average zero over the data).

  Scenario: The 95% bands contain the true curves
    Given 5 samples of 2000 people from world 1
    When I fit a Bayesian causal KAN to each sample
    Then on average at least 90% of points of every band contain the true curve
