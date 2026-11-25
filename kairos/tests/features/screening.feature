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

  Scenario: A gate gets a surface between the gate and what it lets through
    Given 1000 people where y = flag · 2x plus noise, with a yes/no flag
    When a node for y screens its pairs of parents
    Then the surface flag×x is kept
    And with the flag off, x moves y by less than a tenth of what it does with the flag on

  Scenario: The decision to keep a surface does not depend on the tiny ridge
    Given 3000 people where y = sin(x1) + 0.5·x2 + 0.03·(x1 − 2)·(x2 − 2) plus noise
    When I compute the evidence gain of the surface with ridge 0.001 and with ridge 0.000001
    Then the two gains differ by less than 1
