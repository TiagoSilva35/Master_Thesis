Feature: Interventional rows
  In a row where a variable was set by an intervention, its parents did not
  produce its value, so the row says nothing about its equation. Each variable
  is fitted only on the rows where it was not set, and those rows still count
  for every other variable (the interventional score of GIES, Hauser & Bühlmann
  2012).

  Scenario: Without interventions a straight line with bell-shaped noise has no direction
    Given 1000 observational people where y = 2x plus bell-shaped noise
    When the graph is searched
    Then x and y are linked, with the direction flagged doubtful

  Scenario: Rows that set x tell the direction
    Given 800 observational people where y = 2x plus bell-shaped noise, and 200 where x was set around 2
    When the graph is searched with the rows that set x marked
    Then the arrow x → y is found and not flagged doubtful
    And reversing it loses more than the margin

  Scenario: A variable is fitted only on the rows where it was not set
    Given 800 observational people where y = 2x plus bell-shaped noise, and 200 where x was set around 2
    Then x's score equals a node fitted on the 800 rows where x was not set
    And y's score uses all 1000 rows
