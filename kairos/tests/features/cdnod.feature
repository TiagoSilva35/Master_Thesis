Feature: CD-NOD ported from the authors' MATLAB code
  The port follows nonsta_cd_new.m and the KCI-test and GPML files it calls.
  Its pieces are checked against known answers, against causal-learn's port of
  the same KCI code, and on small graphs whose answer is known.

  Scenario: Rasmussen's conjugate gradients find the minimum of the Rosenbrock function
    When minimize.m's port starts the Rosenbrock function at (-1, 1)
    Then it stops at (1, 1)

  Scenario: The Gaussian-process likelihood's gradient matches finite differences
    Given 60 rows with 2 inputs and 3 outputs
    Then the gradient of the negative log likelihood matches finite differences

  Scenario Outline: The unconditional KCI statistic equals causal-learn's port of the same code
    Given <rows> rows where y = sin(2x) plus noise
    Then the unconditional KCI statistic equals causal-learn's

    Examples:
      | rows |
      | 150  |
      | 1500 |

  Scenario Outline: The conditional KCI test separates the ends of a chain only given the middle
    Given 300 rows of a chain a → b → c
    When the <kind> KCI test runs
    Then a and c are dependent, but independent given b

    Examples:
      | kind            |
      | exact           |
      | random-feature  |

  Scenario: Meek's first rule continues an arrow into an undirected link
    Given the graph a → b — c with a and c not linked
    When Meek's rules run
    Then b — c becomes b → c

  Scenario: A changing variable orients the links of its stable neighbours
    Given 400 rows over time where x1 → x2 → x3 and only x2's mechanism changes
    When CD-NOD runs with fixed kernel widths
    Then the arrows found are x1 → x2 and x2 → x3

  Scenario Outline: The changing variable whose changes look most independent becomes the sink
    Given the undirected chain x1 — x2 — x3 where all three mechanisms change
    When the Independent Change Principle step runs with a score that ranks <order> as most independent
    Then the arrows found are <arrows>

    Examples:
      | order        | arrows             |
      | x3, x2, x1   | x1 → x2 and x2 → x3 |
      | x2, x1, x3   | x1 → x2 and x3 → x2 |

  Scenario: The settings follow the authors' choice of test by sample size
    Then 1000 rows use KCI with widths learnt by a Gaussian process
    And 2000 rows use KCI with fixed widths
    And 2001 rows use the random-feature KCI with fixed widths
