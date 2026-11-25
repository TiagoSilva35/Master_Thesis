Feature: CausalArena benchmark
  The baselines run on CausalArena's semantic SCMs, read from files laid out as
  in its Hugging Face release. These scenarios use small hand-made files served by
  a fake repository, so they need no download.

  Scenario: A file is downloaded the first time it is needed, then read from disk
    Given a fake repository with one scenario
    When the release manifest is asked for twice
    Then it was fetched once

  Scenario: A sampled file is read into values, the true graph, measurement scales and C
    Given a fake repository with one scenario
    When its interventional file is loaded
    Then the true arrows are a → b and b → c
    And the raw values are recovered from the standardised ones
    And the measurement scales are continuous for a, binary for b and ordinal_categorical for c
    And C is 0 for the 4 observational rows and 1 for the 2 rows that set b
    And the variables without a cause are a

  Scenario: An observational file has no C
    Given a fake repository with one scenario
    When its observational file is loaded
    Then there is no C

  Scenario: Undirected pairs are pointed at random, the same way for the same file
    Given the true arrows a → b and c → d and a graph with the undirected pairs a — b, c — d and a — e
    Then pointing the pairs twice with the same seed gives the same arrows
    And the precision with random directions lies between the worst and the best

  Scenario: Each method runs once per background it takes, and only where it can
    Given a fake repository with one scenario
    When a method that takes background, one that does not, and one that needs interventions run on both files
    Then the observational file has 3 runs and the interventional file has 5

  Scenario: A method that fails is recorded and the others go on
    Given a fake repository with one scenario
    When a failing method and a working one run on the observational file
    Then the failing method's runs keep its error and the working method's runs have a graph

  Scenario Outline: Files run in separate processes give the same results as one after another
    Given a fake repository with one scenario, already downloaded
    When both files run with 1 worker and with <workers> workers
    Then the two runs have the same results

    Examples:
      | workers |
      | 2       |
      | 5       |

  Scenario: A stopped run goes on from the files it has finished
    Given a fake repository with one scenario, already downloaded
    When the observational file runs, and then both files with the same folder
    Then the second run reads the observational file's results back and runs only the other file

  Scenario: Kairos gets each variable's kind, with codes for the categorical ones
    Given a fake repository with one scenario
    When its interventional file is loaded
    Then kairos treats a as continuous, b as binary and c as ordinal
    And kairos gets the codes 0, 1, 0, 1, 1, 1 for b and 0, 1, 1, 2, 3, 3 for c

  Scenario Outline: A measurement scale written in words becomes a kind
    When the scale "<scale>" with <levels> distinct values is read
    Then kairos treats that variable as <kind>

    Examples:
      | scale                        | levels | kind       |
      | continuous (bounded)         | 999    | continuous |
      | positive_continuous          | 1000   | continuous |
      | proportion [0,1]             | 998    | continuous |
      | binary                       | 2      | binary     |
      | categorical_nominal          | 4      | nominal    |
      | ordinal                      | 3      | ordinal    |
      | ordinal_integer              | 20     | continuous |
      | discrete_count               | 277    | continuous |
      | count (non-negative integer) | 6      | ordinal    |
