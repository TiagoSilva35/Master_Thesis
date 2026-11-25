Feature: Discovery
  The graph is found once, from the people in the training data: the search
  keeps making the single change (add, remove or reverse an arrow) that raises
  the evidence most, after paying a price per arrow, and restarts from shaken
  graphs. An arrow is flagged when its reverse scores about as well.

  Scenario Outline: The search finds a world's graph from its people
    Given 10000 people from world <world>
    And the background says only that Y is the outcome
    When the search runs
    Then the arrows found are the world's arrows

    Examples:
      | world |
      | 1     |
      | 2     |
      | 3     |
      | 4     |
      | 5     |

  # In world 6, education's curve into Y is 0·education, so that arrow has no effect.
  Scenario: With its roots given, the search finds world 6's graph
    Given 10000 people from world 6
    And the background says Y is the outcome and age, education and debt are roots
    When the search runs
    Then the arrows found are the world's arrows except education → Y

  Scenario: Without its roots, the search finds world 6's graph
    Given 10000 people from world 6
    And the background says only that Y is the outcome
    When the search runs
    Then the arrows found are the world's arrows except education → Y

  Scenario: The search never draws an arrow the background rules out
    Given 5000 people from world 1
    And the background says Y is the outcome and savings is a root
    When the search runs
    Then savings has no parents
    And Y is no variable's parent

  Scenario: A shortcut in a chain is not kept
    Given 5000 people where income → savings → Y, with no arrow income → Y
    And the background says only that Y is the outcome
    When the search runs
    Then there is no arrow income → Y

  Scenario: The climb removes a parent the data does not need
    Given 5000 people where income → savings → Y, with no arrow income → Y
    And the background says only that Y is the outcome
    When the search climbs from a graph where Y has the parents income and savings
    Then Y keeps savings and loses income

  Scenario: The climb can stop at a graph that no single change improves, and restarts leave it
    Given 800 people from a random 8-variable graph with curved arrows and non-bell noise
    And the background says nothing
    When the search climbs once, and then with restarts
    Then the restarts' graph is worth at least as much as the first one
    And the restarts' arrows are the true ones

  Scenario: An arrow must earn its price
    Given 5000 people where income → savings → Y, with no arrow income → Y
    And the background says only that Y is the outcome
    When the search runs
    Then every arrow's gain is above the price

  Scenario: A straight-line arrow is flagged as doubtful
    Given 5000 people where a is bell-shaped, b = 2·a plus noise, and b → Y
    And the background says only that Y is the outcome
    When the search runs
    Then the arrow between a and b is flagged as doubtful
    And the arrow b → Y is not flagged

  Scenario: An arrow that would close a loop is not allowed
    Given a graph with the arrows a → b and b → c
    Then the arrow c → a would close a loop
    And the arrow a → c would not
    And the order is a, b, c

  Scenario: Each parent set is fitted only once
    Given 2000 people from world 1
    When savings is scored twice with income as its parent
    Then one fit was run

  Scenario: The discovered graph becomes a kairos model
    Given 10000 people from world 1
    And the background says only that Y is the outcome
    When the search runs
    And kairos is built from the discovered graph and fitted
    Then its chances of Y = 1 are within 0.03 of the true chances on average
