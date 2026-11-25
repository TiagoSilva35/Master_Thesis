Feature: Categorical children
  A yes/no, ordered or category child is not bell-shaped around its prediction.
  Its evidence comes from its own likelihood (logistic, ordered logit, softmax)
  through the Laplace approximation, as the outcome's always has. The hill
  Newton climbs has one top, so the approximation sits on the right one.

  Scenario: A yes/no child's evidence prefers its true parent
    Given 500 people where b is yes with a chance that rises with x, and w is unrelated
    When b is scored as a yes/no variable with parent x, and with parent w
    Then the evidence with x is higher by more than 3

  Scenario Outline: The Laplace evidence matches importance sampling
    Given 300 people with a <kind> child of x
    And a <kind> node for it with 4 intervals
    When I compute its Laplace evidence with λ = <strength>
    Then importance sampling with 60000 draws gives the same value within 0.1

    Examples:
      | kind     | strength |
      | ordered  | 1        |
      | ordered  | 100      |
      | category | 1        |
      | category | 100      |

  Scenario: A child that is 98% one category
    Given 400 people where only 2% have b = 1
    And a yes/no node for it with 4 intervals
    When I compute its Laplace evidence with λ = 10
    Then importance sampling with 60000 draws gives the same value within 0.2

  Scenario: Renumbering a category child's categories leaves its evidence unchanged
    Given 300 people with a category child of x
    When it is fitted with its codes and with the codes renumbered
    Then both evidences are equal

  Scenario: A yes/no child that its parent predicts perfectly gets a finite fit
    Given 300 people where b is yes exactly when x > 0
    When b is fitted as a yes/no variable with parent x
    Then its evidence is finite and no height is beyond 20 log-odds

  Scenario: A level that never occurs is dropped
    Given 300 people with an ordered child of x whose codes skip 2
    When it is fitted as an ordered variable with parent x
    Then its levels are 0, 1 and 3, with two cut-points

  Scenario: A child with a single level is skipped
    Given 300 people where c is always 4
    Then c scores 0 with any parent set

  Scenario: An unseen category gets the average score of those seen
    Given 300 people with a category child of x
    When it is fitted with its codes and with the codes renumbered
    Then a category never seen gets the average of the seen categories' scores
