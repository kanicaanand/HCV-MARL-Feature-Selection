Multi-Agent Reinforcement Learning for HCV Diagnosis Feature Selection
1.	Introduction
This project implements a Multi-Agent Reinforcement Learning (MARL) framework for automated feature selection in Hepatitis C Virus (HCV) diagnosis. The goal is to identify a minimal yet highly predictive set of clinical biomarkers from patient data, improving diagnostic accuracy and interpretability. The framework employs distinct RL agents (Greedy, Exploratory, Parsimonious) with varied exploration-exploitation strategies and penalty configurations, alongside a Random baseline, to navigate the feature space. The selected features are then evaluated using a suite of traditional machine learning classifiers.
2.	Key Findings
Effective Feature Selection: The MARL framework successfully identifies compact sets of clinical biomarkers (e.g. ALP, SGOT, Bilirubin) that achieve high diagnostic performance.
Agent Performance:  The Parsimonious DQNAgent consistently demonstrates superior performance, often achieving F1-scores close to 1.0 with a small number of features, highlighting its ability to balance exploration with feature sparsity.
Computational Efficiency: The CPU-optimized environment ensures the training and evaluation processes are efficient, with total runtime for 6 runs and 100 episodes being approximately 241.62 seconds(approximately 4.3 minutes). The average time per run was 40.27 seconds. 
Statistical Significance: Paired t-tests reveal significant differences in F1-scores between certain agent policies for specific classifiers, indicating the distinct impact of different RL strategies.
         Feature Importance: Biomarkers like SGOT, Bilirubin, and SGPT frequently emerge as    highly influential features across various agent policies and runs.
Robustness to Penalties: The sensitivity analysis demonstrates that while penalty configurations can influence selected features, high-performing agents generally maintain strong performance even under varying penalty regimes. 
3.	How to run the Code
To replicate the experiments and analysis  presented in this notebook, follow these steps:
1.	Environment Setup: This notebook is designed to run in a Google Colab environment. Ensure you have access to Colab.
2.	Dataset: Upload the “dataset bakshi nagar 1.xlsx” file to your Google Colab environment’s current working directory. The code assumes this file path.
3.	Execute Cells: Run all the code cells sequentially. The notebook is structured to perform data preprocessing, multi-agent RL training, evaluation, and visualization in a step-by-step manner.
•	The initial cells set up the environment and define helper functions.
•	The main execution block (‘if_ name_ == “_main_”:’) performs the multi-run experiments and caches the results.
•	Subsequent cells generate various plots, tables, and statistical analyses based on the cached results.
4.	Dependence
This project requires the following Python libraries:
•	‘os’
•	‘numpy’
•	‘pandas’
•	‘matplotlib’
•	‘scikit-learn’ (sklearn)
	‘LogisticRegression’, ‘DecisionTreeClassifier’, ‘RandomForestClassifier’
	‘train_test_split’, ‘cross_val_score’,‘MinMaxScaler’
	’accuracy_score’, ‘precision_score’, ‘recall-score’,f1_score
•	‘xgboost’ (‘xgb.XGBClassifier’)
•	‘torch’
•	‘collections’(‘deque’, ‘Counter’)
•	‘time’
•	‘pickle’
•	‘warnings’
•	‘copy’(‘deepcopy’)
•	‘scipy’(‘scipy.stats’ for statistical tests)

All these libraries are commonly available in Google Colab’s default environment. If not, they can be installed using ‘!pip install <package-name>’.
5.	Data
Filename: ‘dataset bakshi nagar 1.xlsx’
Content : Contains clinical biomarker data for HCV diagnosis.
Preprocessing: The `preprocess_data()` function handles loading, target mapping 
(‘HCV RNA Detected’ to 1, others to 0), imputation of missing values (median imputation), and feature scaling (MinMaxScaler). Features related to age and other non-biomarkers are excluded.

6. Results Structure
The notebook generates various plots and tables:
F1-Score Heatmap: Visualizes the averaged F1-scores across agents and classifiers.
Accuracy/Precision/Recall Heatmaps: Similar heatmaps for other key metrics.
F1-Score Trends Across Runs: Line plots showing agent-classifier performance over individual experimental runs.
Feature Selection Patterns: Heatmap showing which features are selected by each agent (averaged across runs).
Global Feature Importance: Bar chart indicating the overall frequency of feature selection across all agents and runs.
Agent Training/Evaluation Runtimes: Bar plots illustrating the computational cost of agents and classifier evaluations.
Ablation Study Table: Summarizes the best F1-score, number of features, and frequently selected features for each agent policy.
 Statistical Tests (Paired T-Tests): Output of t-test results comparing agent performance for each classifier. Reward Function Sensitivity Analysis: Plots and summaries demonstrating how different penalty configurations impact agent performance and feature selection.
Performance Curves: Line plot showing the average F1-score trend for each agent across training episodes. 
Intermediate results and cached data from multiple runs are saved in the ‘saved_models/’ directory (e.g., ‘hcv_rl_multirun_6runs_cpu.pkl’). Plots are saved in ‘plots_multi_agent_final/’.

7. Experimental Setup and Configuration
   General Setup:
Device: All experiments are explicitly run on CPU (`device = torch.device('cpu')`).
Random Seeds: Fixed seeds for ‘numpy’ (‘42’) and ‘torch’ (‘42’) ensure reproducibility for initial conditions. Each of the `N_RUNS` uses a unique seed derived from ‘42 + run_idx * 1000’ for data splitting and agent initialization, ensuring varied data splits across runs.
Number of Independent Runs (‘N_RUNS’): The main experiment is conducted over 6 independent runs (`N_RUNS = 6`).
Episodes per Run (`N_EPISODES`): Each RL agent is trained for 100 episodes per run (`N_EPISODES = 100`).
Maximum Steps per Episode (`max_steps`): Each episode allows a maximum of 15 steps (feature selection actions).

RL Agents Configuration
“FeatureSelectionEnv”:
Reward Function: ‘f1-penalty len(selected_features)’. An additional penalty of ‘0.1 * (len(selected_features))’ is applied if more than 4 features are selected, encouraging parsimony.
Base Classifier for Reward: LogisticRegression(solver ‘liblinear’, random_state=42, class_weight= ‘balanced’) is used internally for F1-score calculation.
Greedy (Q Learning Agent):
‘epsilon’: 0.2 (lower exploration, more exploitation)
‘alpha’: 0.15(learning rate)
‘gamma’: 0.9(discount factor)
‘epsilon decay’: 0.995, ‘min_epsilon’: 0.1
‘Penalty’ in FeatureSelectionEnv: 0.02(when ‘penalty_config’ is None)
Exploratory (Q Learning Agent):
‘epsilon’: 0.6 (higher exploration)
‘alpha’: 0.05(lower learning rate)
‘gamma’: 0.9(discount factor)
‘epsilon decay’: 0.995, ‘min_epsilon’: 0.1
‘Penalty’ in FeatureSelectionEnv: 0.02(when ‘penalty_config’ is None)
Parsimonious (DQN Agent)
‘epsilon’: 0.4 (moderate exploration)
‘alpha’: 0.01(learning rate for Adam optimizer)
‘gamma’: 0.9(discount factor)
‘epsilon decay’: 0.995, ‘min_epsilon’: 0.1
‘Penalty’ in FeatureSelectionEnv: 0.02(when ‘penalty_config’ is None)
‘Batch size’: 64
‘target_update_freq’: 100 steps
‘Penalty’  in FeatureSelectionEnv : 0.04(when ‘penalty_config’ is None)
Random (Random Agent):
Selects actions uniformly at random.
No learning parameters.
‘Penalty’ in FeatureSelectionEnv: 0.02(when ‘penalty_config’ is None)

Classifier Configuration (for evaluation)
Logistic Regression:
‘solver= liblinear’, ‘random_state=42’, ‘class_weight= balanced’, ‘max_iter=2000’, ‘n_jobs= -1’
Decision Tree:
‘random_state=42’, ‘class_weight=balanced’, ‘max_depth=12’
Random Forest:
‘n_estimators=100’, ‘random_state=42’, ‘class_weight= balanced’, ‘max_depth==12’, ‘n_jobs= -1’
XGBoost:
‘random_state = 42’, ‘eval_metric=logloss’, ‘max_depth=6’, ‘n_estimators=100’, ‘learning_rate = 0.1’, ‘tree_method=hist’, ‘n_jobs= -1’

Data Splitting:
‘train_test_split’ with ‘test_size = 0.2’, ‘stratify=y’, and ‘random_state’ set per run.
Cross-validation (‘cross_val_score’) uses ‘cv=5’ and ‘scoring’== ‘f1’.
Sensitivity Analysis:
‘N_Runs_Sensitivity = 2’, ‘N_Episodes_Sensitivity =50’ for quicker analysis.
Penalty Configurations:
Default Penalties: Uses the default penalties defined above.
Low Feature Penalty: {‘Greedy’: 0.01, ‘Exploratory’: 0.01, ‘Parsimonious’: 0.02} (halved penalties).
High Feature Penalty: {‘Greedy’: 0.05, ‘Exploratory’: 0.05, ‘Parsimonious’: 0.08} (increased penalties).
