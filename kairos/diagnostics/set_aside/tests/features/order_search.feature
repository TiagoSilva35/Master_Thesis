Feature: Order search
  OrderSearch improves the greedy graph by searching over orders of the variables
  (Teyssier & Koller 2005): each variable takes its best parent set among the
  variables before it, so there are no loops to block an arrow, and a swap of two
  neighbours in the order changes only their two parent sets.

  Scenario: Order search escapes the collider trap
    Given 500 people where z = x + y plus bell-shaped noise, with x and y unrelated (seed 3)
    When the greedy search runs, and then the order search from its graph
    Then greedy does not find x → z ← y
    And the order search finds x → z ← y, with a higher total evidence

  Scenario: Order search recovers a random graph that greedy gets partly wrong
    Given 800 people from a random 8-variable graph with curved arrows and non-bell noise (seed 1)
    When the greedy search runs, and then the order search from its graph
    Then greedy's arrows differ from the true ones
    And the order search's arrows are the true ones
    And the order search never scores below greedy

  Scenario: The best parent set is the best option within the variables allowed
    Given 800 people from a random 8-variable graph with curved arrows and non-bell noise (seed 1)
    When parent sets are prepared with v0 as a root
    Then v0's only option is no parents
    And each variable's best option within a set uses only parents from that set and beats every small set within it

  Scenario: A swap's change in score equals scoring the whole order again
    Given 800 people from a random 8-variable graph with curved arrows and non-bell noise (seed 1)
    When parent sets are prepared with v0 as a root
    Then swapping each pair of neighbours in an order changes the total by the swap's computed change
