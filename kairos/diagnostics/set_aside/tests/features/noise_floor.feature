Feature: Noise floor
  When a variable is an exact formula of its parents, the leftovers are almost
  zero and −N/2 · log σ̂² runs off to infinity: one such arrow outweighs many
  ordinary ones. The noise floor trusts every continuous variable only up to a
  share of its own variance. An equation's noise variance is then at least the
  child's own imprecision plus each parent's, carried through the equation:

      floor = share × var(child) + Σ slope² × share × var(parent)

  So no arrow can gain more than about N/2 · log(1 / share), and a formula gains
  the same whichever of its variables it is solved for.

  Scenario: A rounded formula's gain is capped by the floor
    Given 500 people where z is x minus w, rounded to 0.01
    When I score z with and without its parents x and w, with noise floor 0.01 and with none
    Then with the floor z's gain is at most N/2 · log(1.01 / 0.01)
    And without the floor z's gain is more than 1.5 times that

  Scenario: The floor keeps an ordinary relation's gain
    Given 300 people where y = sin(x) plus bell-shaped noise
    When I score y with and without its parent x, with noise floor 0.0001 and with none
    Then with the floor y keeps at least 99% of its gain

  Scenario: A floor of 0 gives the plain evidence
    Given 300 people where y = sin(x) plus bell-shaped noise
    Then a node with noise floor 0 has the same evidence as a node without one

  Scenario: The floor adds the child's imprecision and the parents', carried through the slopes
    Given 500 people where z is exactly 2 times x minus w
    When I fit z on x and w with noise floor 0.01
    Then each row's floor is 1% of var(z) + 4 var(x) + var(w), to within 1%

  Scenario: A formula has the same floor whichever variable it is solved for
    Given 500 people where z is exactly 2 times x minus w
    When I fit z on x and w, and x on z and w, with noise floor 0.01
    Then z's floor is 4 times x's floor, to within 1%, because z moves 2 when x moves 1

  Scenario: A categorical parent adds no imprecision
    Given 500 people where y is 2 higher in group 1 than in group 0, plus noise
    When I fit y on its group with noise floor 0.01
    Then each row's floor is 1% of var(y)

  Scenario Outline: A formula no longer turns around the graph
    Given 1000 people where z is exactly <a> times x minus <b> times y, with <noise> x and y
    When I score the true graph and the two graphs that solve the formula for x or y, with <score> noise
    Then with only the child's own share, 0.01, a wrong graph beats the true one by more than 200
    And with the carried floor of share 0.01, no wrong graph beats the true one by more than 20
    And with the carried floor of share 0.0001, no wrong graph beats the true one by more than 3

    Examples:
      | a   | b  | noise | score       |
      | 1   | 1  | bell  | bell-shaped |
      | 1   | 1  | flat  | bell-shaped |
      | 100 | 30 | bell  | bell-shaped |
      | 100 | 30 | flat  | bell-shaped |
      | 1   | 1  | bell  | learnt      |
      | 1   | 1  | flat  | learnt      |
      | 100 | 30 | bell  | learnt      |
      | 100 | 30 | flat  | learnt      |

  Scenario: The floor does not blur the direction of an ordinary relation
    Given pairs y = f(x) plus noise of 20% of y's spread, for 5 formulas, 2 shapes and 3 seeds
    When I score both directions of every pair with learnt noise and noise floor 0.0001
    Then at least 90% of the pairs point the right way

  Scenario: Slopes match small steps along the curve and the surface
    Given a node with two curves and a surface, fitted with noise floor 0.0001
    Then its slopes match the change in the prediction for a small step, along each parent
