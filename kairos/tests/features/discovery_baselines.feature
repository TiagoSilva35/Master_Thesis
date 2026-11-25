Feature: Scoring and the causal-learn baselines of the benchmark
  How a found graph is scored against the true one, how causal-learn's graphs are
  read, and that PC and CD-NOD behave as expected on small graphs whose answer is
  known.

  Scenario: A found arrow is right, turned around or extra, and a true arrow can be missed
    Given the true arrows a → b, b → c and c → d
    And the found arrows a → b, c → b and a → d
    Then 1 arrow is right, 1 is turned around, 1 is extra and 1 is missed
    And the precision is 1/3 and the recall is 1/3
    And the structural Hamming distance is 3

  Scenario: An undirected pair counts in its true direction at best and the other way at worst
    Given the true arrows a → b and c → d
    And a graph with the arrow a → b and the undirected pairs c — d and a — e
    Then at best 2 arrows are right, 0 are turned around and 1 is extra
    And at worst 1 arrow is right, 1 is turned around and 1 is extra

  Scenario: causal-learn's graph matrix is read into arrows and undirected pairs
    Given a causal-learn matrix over x, y, z and w plus a changing factor, with x → y, z — w and y → factor
    Then the arrows read are x → y
    And the undirected pairs read are w — z

  Scenario: PC orients a collider between two independent causes
    Given 2000 rows where c = a + b plus noise and a and b are independent
    When PC with the Fisher z test runs without background
    Then the arrows found are a → c and b → c

  Scenario: CD-NOD orients an arrow from the variable that changes between runs
    Given two runs of 1000 rows where x shifts between the runs and y = 2·x plus noise
    When CD-NOD with the Fisher z test runs on the two runs
    Then the arrows found are x → y

  Scenario: CD-NOD draws no link between two variables given as roots
    Given two runs of 1000 rows where x shifts between the runs and y = 2·x plus noise
    When CD-NOD with the Fisher z test runs on the two runs with x and y as roots
    Then x and y are not linked
