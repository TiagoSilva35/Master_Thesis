Feature: Learnt noise spread
  A FlexibleNode also learns how far its values stray from the prediction, as a
  spread curve per parent (log spread² = c + Σ g_parent), fitted to the squared
  leftovers and pulled towards flat. When the noise's spread changes with the
  causes, the leftovers divided by their spread are unrelated to the causes again,
  so the learnt shape can tell the direction.

  Scenario: The spread curve follows a spread that grows with the cause
    Given 1000 people where y = x plus noise whose spread grows from 0.3 to 1.3 as x goes from 0 to 2
    When y is fitted with learnt noise
    Then the spread² at x near 2 is more than 5 times the spread² at x near 0

  Scenario: The spread curve stays flat when the spread does not change
    Given 1000 people where y = x plus noise with a constant spread
    When y is fitted with learnt noise
    Then the largest spread² is within 10% of the smallest

  Scenario: A sigmoid-shaped spread no longer turns the direction around
    Given 1000 people where y = x plus evenly spread noise whose spread follows a sigmoid of x
    When both directions are scored with bell-shaped and with learnt noise
    Then the bell-shaped score prefers y → x by more than 3
    And the learnt noise prefers x → y by more than 3

  Scenario: A rounded formula's gain stays capped by the noise floor
    Given 500 people where z is x minus w, rounded to 0.01
    When z is scored with learnt noise and noise floor 0.01, with and without its parents
    Then z's gain is at most 1.1 times N/2 · log(1.01 / 0.01)
