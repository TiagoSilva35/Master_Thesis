Feature: Drift detection
  ChangeFit fits a node once on people from before and after a drift:
  shared terms plus a change part per term. A term counts as changed when
  its change makes the data more than e^3 ≈ 20 times more probable (gain > 3).

  Scenario: A real change in an equation is flagged
    Given 5000 people from before and 5000 from after the drift in world 1
    When the savings node is fitted jointly on both
    Then income -> savings is flagged as changed

  Scenario: A shift in who the people are is not a change
    Given 5000 people from before and 5000 from after the drift in world 3
    When the income node is fitted jointly on both
    Then age -> income is not flagged

  Scenario: An interaction that flips sign is caught on the surface only
    Given 2500 people from before and 2500 from after the drift in world 5
    When the outcome node is fitted jointly on both, with screening
    Then income×savings -> Y is flagged as changed
    And income -> Y is not flagged
    And savings -> Y is not flagged

  Scenario: The drift verdict does not depend on the tiny ridge
    Given 5000 people from before and 5000 from after the drift in world 1
    When the savings node is fitted jointly with ridge 0.001 and with ridge 0.000001
    Then the gains of income -> savings differ by less than 1

  Scenario: The level-change verdict does not depend on the tiny ridge
    Given 5000 people from before and 5000 from after the drift in world 3
    When the income node is fitted jointly with ridge 0.001 and with ridge 0.000001
    Then the two level gains differ by less than 1
    And the level of income is not flagged

  Scenario: A node without a smoothness belief is refused
    Given 500 people from before and 500 from after the drift in world 1
    When I fit the savings node jointly with a plain fitting
    Then it refuses because it needs Bayesian fitting
