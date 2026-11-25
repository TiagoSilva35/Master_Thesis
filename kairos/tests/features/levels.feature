Feature: Categorical parents
  A categorical parent enters its child's equation as one height per level
  (Levels), not as a curve over its codes. Ordered levels are pulled towards
  their neighbours, unordered ones towards their average, with a strength λ
  chosen by the evidence.

  Scenario: Renumbering an unordered parent's levels leaves the child's evidence unchanged
    Given 400 people where y depends on an unordered parent g with 4 levels
    When y is fitted with g's codes and with the codes renumbered
    Then both evidences are equal

  Scenario: An ordered parent with 5 levels recovers a step pattern
    Given 500 people where y steps up by 1 between levels 2 and 3 of an ordered parent g
    When y is fitted with g as an ordered parent
    Then the step between levels 2 and 3 is above 0.7 and every other step is below 0.2

  Scenario: A yes/no parent gives a single difference between its two heights
    Given 400 people where y is 0.8 higher when the yes/no parent g is 1
    When y is fitted with g as a yes/no parent
    Then g's edge has two heights, 0.8 apart within 0.15

  Scenario: A level never seen predicts the average
    Given 500 people where y steps up by 1 between levels 2 and 3 of an ordered parent g
    When y is fitted with g as an ordered parent
    Then a level of g never seen pushes y by the average of g's heights

  Scenario: Shuffling the order of ordered levels lowers the evidence
    Given 200 people where y rises evenly over the 8 levels of an ordered parent g
    When y is fitted with g's levels in their order and in a shuffled order
    Then the evidence in their order is higher by more than 3
