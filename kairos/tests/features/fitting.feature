Feature: Choosing the smoothness
  Each smoothness strength λ is the one under which the data was most probable. The
  search sets λ = noise variance · freedom / bending until the evidence stops rising.

  Scenario Outline: The search reaches the best evidence of a fine grid
    Given 500 people where y = <curve> plus noise of spread <noise>
    When y is fitted, and every λ of a fine grid is tried
    Then the fit's evidence is at least the grid's best, within 0.1

    Examples:
      | curve  | noise |
      | sin 3x | 0.02  |
      | sin 3x | 0.3   |
      | 2x     | 0.3   |
      | 0      | 1     |

  Scenario: A wiggly curve with little noise needs a weak belief
    Given 500 people where y = sin 3x plus noise of spread 0.02
    When y is fitted, and every λ of a fine grid is tried
    Then the λ chosen is below 0.1

  Scenario: An exact formula's gain is capped
    Given 500 people where z is exactly 2 times x minus w
    When z is fitted on x and w with a noise floor of 0.0001
    Then its gain over no parents is at most N/2 · log(1 / 0.0001)

  Scenario: A formula has the same floor whichever variable it is solved for
    Given 500 people where z is exactly 2 times x minus w
    When z is fitted on x and w, and x on z and w, with a noise floor of 0.0001
    Then z's noise variance is 4 times x's, to within 1%, because z moves 2 when x moves 1
