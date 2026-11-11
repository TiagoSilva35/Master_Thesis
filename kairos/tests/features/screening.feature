Feature: Screening
  A surface (a grid over two parents) is added only when the data shows an
  interaction: the effect of one parent depends on the other.

  Scenario Outline: A surface is kept only when there is an interaction
    Given 3000 people where y = sin(x1) + 0.5·x2 + <strength>·(x1 − 2)·(x2 − 2) plus noise
    When a node for y screens its pairs of parents
    Then the surface x1×x2 is <decision>

    Examples:
      | strength | decision |
      | 0        | rejected |
      | 0.1      | kept     |

  # The surface's simple saddle has a smoothness belief of its own (Surface.saddle);
  # before it had one, only the tiny ridge held it and this gain moved by about 3.45.
  Scenario: The decision to keep a surface does not depend on the tiny ridge
    Given 3000 people where y = sin(x1) + 0.5·x2 + 0.03·(x1 − 2)·(x2 − 2) plus noise
    When I compute the evidence gain of the surface with ridge 0.001 and with ridge 0.000001
    Then the two gains differ by less than 1
