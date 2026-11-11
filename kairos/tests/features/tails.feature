Feature: Beyond the data
  After a drift, or when a person is asked to change something, a value can fall
  past the range the model was fitted on. There the curves continue as straight
  lines with the slope they have at the edge, and their bands widen.
  World 1: savings = 0.5 · income², with income between 0 and 4 in the data.

  Background:
    Given 2000 people from world 1
    And a Bayesian causal KAN fitted to them

  Scenario: Predictions past the data follow the trend
    When I predict savings for income 4.5
    Then the prediction is within 0.4 of the true 10.125

  Scenario: The band widens beyond the data
    Then the income -> savings band is wider at 5 than at 4
    And the income -> savings band is wider at 6 than at 5
