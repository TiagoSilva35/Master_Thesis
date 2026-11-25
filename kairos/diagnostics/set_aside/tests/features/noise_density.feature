Feature: Learnt noise shape
  A FlexibleNode replaces the bell's fit term −N/2 · log σ̂² with the leftovers'
  log-density under a learnt shape f (NoiseDensity: a P-spline on the log scale,
  fitted to binned leftovers). The bell measures only how spread out the
  leftovers are; f also sees their shape, and so whether the leftovers are
  unrelated to the cause, which is what tells the direction.

  Scenario: Bell-shaped numbers keep a smooth shape that does not pay for itself
    Given 1000 bell-shaped numbers
    When their shape is learnt
    Then λ is at least 100 and the log-likelihood is within 6 of the bell's
    And the shape's flexibility costs more than it gains over the bell

  Scenario: Evenly spread numbers get a flat shape that beats the bell
    Given 1000 numbers spread evenly between −1 and 1
    When their shape is learnt
    Then f is within 10% of ½ at −0.5, 0 and 0.5
    And the log-likelihood beats the bell's by more than 150

  Scenario: A straight line with evenly spread cause and noise
    Given 1000 people where y = 2x plus noise, with x and the noise spread evenly
    When both directions are scored with bell-shaped and with learnt noise
    Then the learnt noise prefers x → y by more than 3
    And the bell-shaped score does not prefer x → y by more than 3

  Scenario: A uniform cause with y = x² plus small noise
    Given 1000 people where y = x² plus small bell-shaped noise, with x spread evenly
    When both directions are scored with bell-shaped and with learnt noise
    Then the learnt noise prefers x → y by more than 3
    And the bell-shaped score also prefers x → y by more than 3

  Scenario: The learnt-noise evidence is close to importance sampling
    Given 300 people where y = sin(x) plus heavy-tailed noise
    When y is fitted with learnt noise and 4 intervals
    Then importance sampling over the curve's heights, with the spread and f fixed, gives the same value within 0.5

  Scenario: Values on a grid are recognised
    Then values rounded to 0.01 have a rounding step of 0.01, and values not rounded have none
