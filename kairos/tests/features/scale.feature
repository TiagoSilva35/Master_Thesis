Feature: The scale an equation is written on
  Some equations are a sum of curves only after undoing a bend: x = e^(a + b) is
  not a sum, log x = a + b is. The evidence chooses between the plain, log and
  log-odds scales. Scores on different scales are comparable because each person's
  density is multiplied by how much the scale stretches the axis at their value.

  Scenario Outline: The evidence picks the scale on which the equation is a sum
    Given 1000 people where z is <form>
    When z is scored with its parents x and y
    Then the <scale> scale is chosen

    Examples:
      | form              | scale    |
      | a sum             | plain    |
      | e to a sum        | log      |
      | a logistic of one | log-odds |

  Scenario: A scale is offered only where it exists
    Then values with a zero are offered the plain scale only
    And values above 1 are offered plain and log
    And values between 0 and 1 are offered plain, log and log-odds

  Scenario Outline: Forward, backward and stretch agree
    Given values between 0.05 and 0.95 on the <scale> scale
    Then backward undoes forward
    And stretch is the log of forward's slope

    Examples:
      | scale    |
      | plain    |
      | log      |
      | log-odds |

  Scenario: Evidences on two scales differ as the true densities do
    Given 20000 numbers whose logs are bell-shaped with spread 0.7
    When they are scored without parents on the plain and on the log scale
    Then the difference is within 5% of the difference between the two bells' log-likelihoods

  Scenario: A node on a log scale predicts in the variable's own units
    Given 1000 people where z is e to a sum
    When z is fitted on the log scale
    Then its predictions are within 15% of the true values on average, and its noise plus its sum gives log z
