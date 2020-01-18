Data file:
    openfoodfacts_simplified_database.xlsx

Python version:
    Python 3.7.4

Python required packages:
    requirements_pip.txt
    requirements_conda.txt

Python scripts:
    AdditiveNutriScore.py
    MajoritySortingNutriScore.py
    MachineLearningNutriScore.py

How to run?
    python <script-name>
    (The data file should be in the same directory as the scripts)

Outputs:
    AdditiveNutriScoreResults.txt => Preference order of 100 foods
    NutriScoreDecisionTree.pdf => The entire decision tree in a graphical format
    NutriScoreDecisionTree => The entire decision tree in a textual format
    
    The various accuracies are printed on the console.

    Many images will be generated as you run the scripts:
        Confusion matrices
        Marginal utilities

    (The outputs will be generated in the same directory as the scripts)

Sample console output:
    sample_console_output.jpg => A screenshot of sample console output