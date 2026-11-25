Feature: The score points to the right graph
  A wrong graph can beat the true one only when the true equation is one the model
  cannot write. With a scale and surfaces, it can write products, S-shapes and gates
  as well as sums. Each scenario compares the true graph x → z ← y with every other
  graph over the three variables.

  Scenario Outline: No wrong graph scores above the true one
    Given 1000 people where <form>
    When every graph over the three variables is scored
    Then the true graph scores highest, or within 3 of the highest

    Examples:
      | form                                 |
      | z is a sum of curves                 |
      | z is e to a sum                      |
      | z is a logistic of a sum             |
      | z is a product that can be negative  |
      | z is a gate                          |
      | z has noise that multiplies          |
      | z is exactly x minus y               |
      | z is exactly 100x minus 30y          |
      | z is exactly x times y, all positive |

  Scenario Outline: The right form matters: without it a wrong graph wins
    Given 1000 people where <form>
    When every graph over the three variables is scored without scales and without surfaces
    Then a wrong graph beats the true one by more than 20

    Examples:
      | form                                 |
      | z is e to a sum                      |
      | z is a logistic of a sum             |
      | z is a gate                          |
      | z is exactly x times y, all positive |
