Feature: Evidence
  The evidence is how probable the data is under a smoothness strength λ,
  averaged over every curve that λ finds believable. Bayesian fitting picks
  λ, keeps surfaces and flags drift with it, so it must be computed exactly.

  Scenario Outline: A feature node's evidence equals the direct formula
    Given 300 people where y = sin(x) plus bell-shaped noise
    And a Bayesian node for y with parent x and 10 intervals
    When I compute its evidence with λ = <strength>
    Then it equals the direct marginal likelihood up to the constant n/2·(log 2π + 1)

    Examples:
      | strength |
      | 0.1      |
      | 10       |
      | 1000     |

  Scenario Outline: The outcome's approximate evidence matches importance sampling
    Given 400 people with a yes/no outcome whose chance rises with x
    And a Bayesian outcome node for Y with parent x and 5 intervals
    When I compute its Laplace evidence with λ = <strength>
    Then importance sampling with 60000 draws gives the same value within 0.1

    Examples:
      | strength |
      | 1        |
      | 100      |

  Scenario: Newton's method reaches the top of the hill
    Given 5000 people from the large world
    And its outcome node with surfaces income×debt and age×savings
    When I fit the outcome with every λ = 1
    Then the remaining slope is below 1e-8
