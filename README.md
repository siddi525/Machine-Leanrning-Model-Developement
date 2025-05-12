Loading Motion Sensor Data into Pandas DataFrame
The following Python code imports necessary libraries and combines motion sensor data from multiple CSV files, storing it in a structured Pandas DataFrame. Subject-specific information, such as age, gender, height, and weight, is added to the dataset for comprehensive analysis.

Steps:
Imports the necessary libraries:

'os' for file operations.
'numpy' as 'np' for numerical operations.
'pandas' as 'pd' for data handling.
Specifies the path:

Defines the path to the subject data file and the directory containing motion sensor data.
Defines two functions:

get_all_dataset_paths: Recursively walks through the specified directory and collects paths to all CSV files.
load_whole_dataframe_from_paths: Reads and combines motion sensor data from these paths into a single Pandas DataFrame. It also enriches the data with subject information from the subject data file.
Loads the subject data:

Loads the subject data from the CSV file data_subjects_info.csv into a Pandas DataFrame.
Calls get_all_dataset_paths:

Obtains a list of paths to all CSV files in the specified directory.
Calls load_whole_dataframe_from_paths:

Creates a comprehensive DataFrame containing motion sensor data, with additional subject information.
This code is a critical step in preparing motion sensor data for analysis and is commonly used in data science and machine learning projects involving motion data.

DataFrame Manipulation
In this Python code, a copy of the original DataFrame 'data_frame' is created. Subsequently, several columns are removed from the copied DataFrame 'df' to streamline the dataset for further analysis. The columns removed are:

'Unnamed: 0'
'subject_id'
'session_id'
'age'
'gender'
'height'
'weight'
This step simplifies the dataset by eliminating non-essential features, making it more focused and ready for analysis.

Encoding Categorical Data for Machine Learning
The following Python code snippet utilizes the 'LabelEncoder' from the scikit-learn library to transform the 'category' column in the DataFrame 'df' into numerical codes. These codes are stored in a new 'code' column, and the original 'category' column is subsequently removed from the DataFrame, preparing the data for machine learning tasks.

This transformation allows categorical data to be represented numerically, making it suitable for machine learning algorithms.


Visualizing the data
We use Seaborn and Matplotlib to create a countplot, visualizing the distribution of numerical codes in the 'code' column of the DataFrame 'df.' This plot provides insight into the frequency of different categories in the dataset.

Splitting Data for Machine Learning
The following code uses the 'train_test_split' function from scikit-learn to divide the dataset into training and testing sets. It separates the input features ('x_columns') and the target variable ('y_columns') with a 20% test set size, ensuring that the lengths of the training sets for both features and labels are the same, as asserted.

This is a critical step in preparing data for machine learning models, allowing for the evaluation of model performance on unseen data.

Sequencing Data for Temporal Analysis
We define a sequence generator function that creates sequences of input features and corresponding target labels from the training and testing data. These sequences have a window length of 150 with a stride of 10. The mode of target labels within each sequence is calculated to represent the label for that sequence.

This prepares the data for temporal analysis tasks, making it suitable for time-series or sequence-based models such as LSTM (Long Short-Term Memory) networks, which are designed to capture patterns over time.



Defining LSTM-based Deep Learning Model
This code snippet utilizes the Keras library to build a deep learning model for sequence classification. The model is designed to process sequential data, which is common in time-series and motion-sensor tasks.

The architecture includes the following layers:

LSTM Layer:
The first layer is a Long Short-Term Memory (LSTM) layer with 6 units. LSTM is a type of Recurrent Neural Network (RNN) that excels at learning long-term dependencies in sequential data, which is essential for tasks where time is a critical factor, such as activity recognition. The return_sequences=True argument allows the LSTM to output a sequence, which is then flattened in the next step.

Flatten Layer:
After the LSTM layer, a Flatten layer is applied. This layer takes the 3D output from the LSTM (time steps, batch size, number of features) and flattens it into a 1D array that can be passed to the following fully connected layer.

Dense Layer:
This fully connected layer has 128 neurons and uses ReLU (Rectified Linear Unit) activation. ReLU is a popular activation function that helps the model learn non-linear relationships in the data, promoting sparsity in the activations and improving model performance.

Output Layer:
The output layer consists of NUM_CLASSES neurons, corresponding to the number of classes in the classification task. This layer uses the Softmax activation function, which transforms the output into a probability distribution over the classes, enabling multi-class classification.

Model Compilation:
The model is compiled using the following:

Categorical Cross-Entropy Loss: This is the standard loss function used for multi-class classification tasks where each label is represented as a one-hot encoded vector.
Adam Optimizer: Adam is an adaptive optimizer that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. It adapts the learning rate for each parameter and is known for converging faster.
Finally, a summary of the model architecture is displayed, showing the layer types, the number of parameters, and the shape of the outputs at each layer.

Training the LSTM Model
This code snippet is responsible for training the previously defined LSTM-based deep learning model. It uses the training sequences tx (input features) and their corresponding one-hot encoded labels tty (target variable). The model is trained over a specified number of epochs (EPOCHS_SIZE) and batch size (BATCH_SIZE). The training progress and performance metrics are captured and stored in the history variable for later analysis.

Key Parameters:
EPOCHS_SIZE: The number of complete passes through the training data. Each epoch allows the model to learn and adjust weights based on the loss function.
BATCH_SIZE: The number of samples processed before the model's internal parameters (weights) are updated. A smaller batch size might lead to better generalization, while a larger batch size can speed up training.
The model’s performance during training, such as accuracy and loss, is tracked through the history object. This object stores the values of the metrics at each epoch and can be used to visualize the model’s learning curve, assess overfitting or underfitting, and make adjustments as necessary.


Evaluating the LSTM Model
After training the LSTM model, the next crucial step is to evaluate its performance on unseen data. This code snippet evaluates the trained LSTM model using the validation sequences vx (input features) and their corresponding one-hot encoded labels vvy (target variable). The evaluation is done with a specified batch size (BATCH_SIZE), and the model's performance metrics, such as loss and accuracy, are calculated and returned.

Key Parameters:
vx: The validation input data (sequences of features) that the model has not seen during training. This helps to assess how well the model generalizes.
vvy: The one-hot encoded labels corresponding to the validation sequences. These are the true values against which the model's predictions are compared.
BATCH_SIZE: The number of samples processed in a batch during the evaluation.
The evaluation metrics, typically loss and accuracy, are returned by the evaluate method. These metrics provide a quantitative measure of the model's ability to predict correctly on the validation data.

Cross-Validation for Model Evaluation
Cross-validation is a powerful technique to assess the performance of a machine learning model by splitting the data into multiple subsets or "folds." In this case, stratified cross-validation is used to ensure that each fold contains a proportional distribution of classes. This approach is particularly useful for classification tasks, as it ensures that each fold has the same proportion of target classes as the overall dataset.

We perform Stratified K-Fold Cross-Validation using the StratifiedKFold method from scikit-learn. This method divides the data into 5 folds, trains the LSTM model on different train-test splits, and evaluates the model's performance on each fold.

Key Steps in Cross-Validation:
Stratified Split: The data is divided into 5 folds, ensuring that each fold has a balanced representation of each class in the target variable.
Model Training and Evaluation: For each fold, the model is trained on the training data and evaluated on the corresponding test data.
Accuracy Tracking: The accuracy of the model on each test set is recorded in a list called lst_accu_stratified for further analysis.

Analyzing Model Cross-Validation Results
Once the Stratified K-Fold Cross-Validation process is complete, it is essential to analyze the results to gain insights into the model's performance across different folds. The following code computes various statistics to assess the LSTM model’s performance, such as the list of accuracy values, maximum accuracy, minimum accuracy, mean accuracy, and the standard deviation of accuracy. These statistics provide an understanding of the model's stability and generalization ability.

Key Statistics:
List of Accuracy Values: The accuracy achieved by the model on each fold's test set.
Maximum Accuracy: The highest accuracy achieved across all folds, representing the best-case scenario.
Minimum Accuracy: The lowest accuracy achieved across all folds, which can indicate the worst-case performance.
Overall (Mean) Accuracy: The average accuracy across all folds, providing a general estimate of the model's performance.
Standard Deviation of Accuracy: A measure of the variability of the accuracy across the folds. A high standard deviation suggests that the model's performance varies greatly across different subsets of the data.

List of Accuracies per Fold:
[0.9598130583763123, 0.984173059463501, 0.9776227474212646, 0.9934850931167603, 0.9935559034347534]

📈 Maximum Accuracy Achieved: 0.9936 📉 Minimum Accuracy Achieved: 0.9598 📊 Overall (Mean) Accuracy: 0.9817 📐 Standard Deviation: 0.0125

