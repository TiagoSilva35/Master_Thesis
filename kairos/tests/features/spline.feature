Feature: Spline curves
  A curve is a sum of bumps, each scaled by its own height: curve(x) = Σ weight_k(x) · height_k.
  These scenarios check the bumps against the textbook definition of a cubic B-spline.

  Background:
    Given a spline from 0 to 4 with 8 intervals

  Scenario: The bump weights follow the textbook B-spline definition
    When I compute its bump weights at 500 random points in its range
    Then they match the Cox-de Boor recursion to within 1e-12

  Scenario: The bump weights add up to one
    When I compute its bump weights at 500 random points in its range
    Then every point's weights add up to 1

  Scenario Outline: Three bumps are active on a knot, four between knots
    When I look at the point <x>
    Then <count> bumps have a weight above zero

    Examples:
      | x   | count |
      | 2.0 | 3     |
      | 2.3 | 4     |

  Scenario: On a knot the weights are one sixth, four sixths and one sixth
    When I look at the point 2.0
    Then the weights above zero are 1/6, 4/6 and 1/6

  Scenario: A spline can draw any straight line
    Given its heights equal its bump centres
    Then the curve is the straight line y = x inside its range

  Scenario: A straight line keeps going beyond the range
    Given its heights equal its bump centres
    Then the curve at 5 is 5
    And the curve at -1 is -1

  Scenario: Beyond the range a curve continues with the slope it has at the edge
    Given its heights are the squares of its bump centres
    Then beyond the range the curve is a straight line with its slope at the edge

  Scenario: The weights still add up to one beyond the range
    When I compute its bump weights at -2 and 7
    Then every point's weights add up to 1

  Scenario: Points beyond the range are marked as outside
    Then the points -0.5, 0, 2, 4 and 4.5 are outside, inside, inside, inside and outside

  Scenario: An unfitted curve has no band
    When I ask for its band before fitting
    Then it refuses because the curve has no spread yet
