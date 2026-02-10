# HCV-MARL-Feature-Selection
This code section sets up a comprehensive framework for multi-agent reinforcement learning (RL) based feature selection for HCV detection, optimized for CPU. Here's a breakdown of its key functionalities:
1.	Environment Setup: Initializes a GPU-optimized environment, setting up random seeds and creating directories for saving models and plots.
2.	Data Preprocessing: Defines a function to load and preprocess HCV dataset from an Excel file, including target mapping, feature selection (clinical biomarkers), median imputation for missing values, and Min-Max scaling.
3.	RL Environment: Implements a FeatureSelectionEnv class, modeling the feature selection problem as an RL task. It uses Logistic Regression to calculate rewards based on F1-score and feature count.
4.	RL Agents: Defines three types of agents for feature selection:
o	QLearningAgent: A traditional Q-learning agent with exploration-exploitation trade-off.
o	DQNAgent: A Deep Q-Network agent utilizing a neural network for Q-value approximation, leveraging PyTorch and GPU acceleration.
o	RandomAgent: A baseline agent that selects features randomly.
5.	Agent Training and Evaluation: Includes functions to:
o	train_agents: Train all defined RL agents over multiple episodes, allowing them to learn feature selection policies. It supports GPU acceleration and manages GPU memory.
o	evaluate_agents_multi_classifier: Evaluate the features selected by each agent using an array of machine learning classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost), also optimized for GPU where possible. It calculates various metrics like accuracy, precision, recall, F1-score, and cross-validation F1.
6.	Multi-Run Analysis: The main execution block orchestrates multiple runs (5 in this case) of the entire process (data splitting, agent training, evaluation) to ensure statistical robustness of the results. It caches the results to avoid re-computation.
7.	Results Averaging: A utility function average_results_across_runs calculates mean, standard deviation, min, and max for all performance metrics and identifies the most frequently selected features across runs.
8.	Visualization: Generates five distinct plots using matplotlib and seaborn to visualize and analyze the results:
o	Plot 1: Multi-Classifier Performance Comparison Across Agents (F1 Score).[Showing Mean and Standard deviation]
o	Plot 2: Feature Selection Efficiency (Performance vs. Number of Features), highlighting optimal efficiency.[This plot compares two approaches: Greedy versus Parsimonious for Random Forest] , must be considered in our future work.
o	Plot 3: Feature Selection Pattern Matrix, showing which features are selected by which agents and highlighting key biomarkers [Fig. 3: Heatmap of Feature Selection  Frequency across agents.]
o	Plot 4: Statistical Robustness Analysis with Confidence Intervals and Performance Variance.[Not explicitly inserted in our manuscript], we will consider in our future work.
o	Plot 5: Linear vs. Non-Linear Classifier Performance Analysis.[This displays the F1-score of the agents for all classifiers which we have shown in Fig. 7 with the help of heatmap instead of bargraph]
