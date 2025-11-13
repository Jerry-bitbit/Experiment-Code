================================================================================
        Black-Box Online Data Poisoning Against Trimming Defenses
                   Experimental Code Run Guide Documentation
================================================================================

[Paper Title]
Black-Box Online Data Poisoning Against Trimming Defenses: From White-Box
Interactive Games to Bandits, Hardness, and Anti-Learning Defenses

[Project Overview]
This code repository contains implementations of five core experiments from the paper, 
researching the performance of UCB and KL-UCB algorithms against trimming defense 
mechanisms in black-box online data poisoning attacks.

[Experimental File List]
1. "UCB vs KL-UCB Cumulative Regret.py"          - Experiment 1: Cumulative Regret Comparison
2. "UCB vs KL-UCBPerformance under Strict Observation and Debiasing.py" 
                                                  - Experiment 2: Performance under Strict Observation and Debiasing
3. "Controller Stability Domain Comparative Analysis.py" 
                                                  - Experiment 3: Controller Stability Domain Analysis
4. "Performance Analysis of KI-UCB in Non-stationary Environments.py" 
                                                  - Experiment 4: Performance Analysis in Non-stationary Environments
5. "Analysis of KL-UCB Algorithm's Significant Advantages in Privacy-Preserving Environments.py"
                                                  - Experiment 5: Advantages in Privacy-Preserving Environments

[Environment Requirements]
==========================================================
Python 3.8+ environment, requiring the following dependency packages:

Core Scientific Computing:
- numpy
- scipy
- pandas

Machine Learning and Data Processing:
- scikit-learn
- matplotlib
- seaborn

[Installation Steps]
==========================================================
Install using pip one by one:
pip install numpy scipy pandas scikit-learn matplotlib seaborn

[Dataset Description]
==========================================================
All experiments use the following public datasets, which will be automatically downloaded from OpenML:

1. Adult Dataset (Income Prediction)
   - Task: Predict whether personal annual income exceeds $50K
   - Sample size: ~48,842
   - Automatically downloaded during first run, ensure stable internet connection

2. German Credit Dataset (Credit Risk Assessment)  
   - Task: Predict whether credit risk is "bad"
   - Sample size: ~1,000
   - Used as alternative dataset


[Experimental Run Guide]
==========================================================

Experiment 1: UCB vs KL-UCB Cumulative Regret Comparison
----------------------------------------------------------
File: UCB vs KL-UCB Cumulative Regret.py

Run Command:
python "UCB vs KL-UCB Cumulative Regret.py"

Output Files:
- figures/ucb_vs_klucb_comparison.png

Function Description:
- Compare cumulative regret of UCB and KL-UCB algorithms


Experiment 2: Performance under Strict Observation and Debiasing Conditions
----------------------------------------------------------
File: UCB vs KL-UCBPerformance under Strict Observation and Debiasing.py

Run Command:
python "UCB vs KL-UCBPerformance under Strict Observation and Debiasing.py"

Output Files:
- real_dataset_comparison.png
- real_dataset_comparison.pdf  
- real_data_results/real_dataset_comparison.json

Function Description:
- Test algorithm performance under LDP privacy protection and bucket feedback conditions
- Compare performance on two real datasets
- Include acceptance rate control and regret analysis

Experiment 3: Controller Stability Domain Comparative Analysis
----------------------------------------------------------
File: Controller Stability Domain Comparative Analysis.py

Run Command:
python "Controller Stability Domain Comparative Analysis.py"

Output Files:
- controller_stability_comparison_real_data.png
- performance_summary_real_data.png
- controller_grid_results_real_data.json

Function Description:
- Analyze the impact of PI controller parameters (kp, ki) on algorithm stability
- Generate parameter grid heatmaps
- Compare performance under different controller settings

Experiment 4: Performance Analysis in Non-stationary Environments  
----------------------------------------------------------
File: Performance Analysis of KI-UCB in Non-stationary Environments.py

Run Command:
python "Performance Analysis of KI-UCB in Non-stationary Environments.py"

Output Files:
- klucb_real_data_performance.png

Function Description:
- Test algorithm adaptability in non-stationary environments
- Use Page-Hinkley change point detection
- Compare algorithm performance with/without restart mechanism

Experiment 5: Advantages Analysis in Privacy-Preserving Environments
----------------------------------------------------------
File: Analysis of KL-UCB Algorithm's Significant Advantages in Privacy-Preserving Environments.py

Run Command:
python "Analysis of KL-UCB Algorithm's Significant Advantages in Privacy-Preserving Environments.py"

Output Files:
- ucb_klucb_heatmaps_real_data.png
- algorithm_comparison_heatmap_real_data.png

Function Description:
- Analyze algorithm performance under different privacy protection levels
- Generate heatmaps for LDP parameters and bucket quantities
- Visualize KL-UCB advantages in privacy-preserving environments

[Complete Run Process]
==========================================================
Recommended to run experiments in the following order:

Step 1: Install Dependencies
pip install numpy scipy pandas scikit-learn matplotlib seaborn

Step 2: Run Basic Comparison Experiment
python "UCB vs KL-UCB Cumulative Regret.py"

Step 3: Run Strict Observation Experiment  
python "UCB vs KL-UCBPerformance under Strict Observation and Debiasing.py"

Step 4: Run Controller Analysis
python "Controller Stability Domain Comparative Analysis.py"

Step 5: Run Non-stationary Environment Experiment
python "Performance Analysis of KI-UCB in Non-stationary Environments.py"

Step 6: Run Privacy Protection Analysis
python "Analysis of KL-UCB Algorithm's Significant Advantages in Privacy-Preserving Environments.py"

[Output File Description]
==========================================================
Image Files (.png, .pdf):
- Cumulative regret curves
- Parameter heatmaps
- Performance comparison charts
- Controller stability domain analysis charts

Data Files (.json):
- Experimental configurations and parameters
- Detailed performance metrics
- Statistical result data

[Windows CMD Run Examples]
==========================================================
1. Open Command Prompt (CMD)
   - Press Win+R, type "cmd", press Enter

2. Navigate to Code Directory
   cd /d C:\path\to\your\code\directory

3. Run Experiments (Examples)
   python "UCB vs KL-UCB Cumulative Regret.py"
   python "UCB vs KL-UCBPerformance under Strict Observation and Debiasing.py"
   python "Controller Stability Domain Comparative Analysis.py"
   python "Performance Analysis of KI-UCB in Non-stationary Environments.py"  
   python "Analysis of KL-UCB Algorithm's Significant Advantages in Privacy-Preserving Environments.py"


================================================================================
                              Documentation End
================================================================================