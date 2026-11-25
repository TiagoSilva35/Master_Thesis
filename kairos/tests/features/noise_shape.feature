Feature: The size and the shape of the noise
  After the curves are fitted, the noise is read off the leftovers: its size, which may
  grow or shrink with the prediction, and one of 15 shapes (peaked, bell or flat; leaning
  right, left or neither). A noise that is not a bell of one size tells the
  direction of an arrow that such a bell cannot.

  Scenario: Noise that grows with the prediction is recognised
    Given 1000 people where y = x plus bell-shaped noise whose spread grows from 0.3 to 1.3 with x
    When y is fitted on x
    Then its noise grows with the prediction

  Scenario: Noise of one size keeps one size
    Given 1000 people where y = x plus bell-shaped noise of one size
    When y is fitted on x
    Then its noise has one size

  Scenario: Every shape is a density with average 0 and spread 1
    Then each of the 15 shapes has area 1, average 0 and spread 1

  Scenario Outline: The shape is recognised from the leftovers
    Given 1000 people where y = 2x plus <noise> noise
    When y is fitted on x
    Then its noise is <shape>

    Examples:
      | noise             | shape         |
      | bell-shaped       | bell          |
      | heavy-tailed      | peaked        |
      | evenly spread     | flat          |
      | exponential       | leaning right |
      | minus exponential | leaning left  |

  Scenario Outline: A straight arrow's direction is found from the shape
    Given 1000 people where y = 2x plus <noise> noise, with x of the same shape
    When both directions are scored
    Then x → y is ahead by more than <margin>

    Examples:
      | noise         | margin |
      | heavy-tailed  | 20     |
      | evenly spread | 20     |
      | exponential   | 100    |

  Scenario: With bell-shaped noise a straight arrow has no direction
    Given 1000 people where y = 2x plus bell-shaped noise, with x of the same shape
    When both directions are scored
    Then neither is ahead by more than 3

  Scenario: An exact formula earns nothing for its shape
    Given 500 people where z is x minus w, rounded to 0.01
    When z is fitted on x and w with a noise floor of 0.0001
    Then its shape adds less than 10 to the evidence
