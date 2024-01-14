import xgboost as xgb
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer,mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem

# Train and Evaluate Each Model - Using a function for reusability
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    relative_error_mse = mean_squared_error(y_test, predictions)
    #relative_error = np.sqrt(mse) / np.mean(y_test)
    return relative_error_mse

def get_error_evaluation(model, x_test, y_test):
    predictions = model.predict(x_test)
    absolute_errors = np.abs(y_test - predictions)
    squared_errors = (y_test - predictions) ** 2
    mae_mean = np.mean(absolute_errors)
    mse_mean = np.mean(squared_errors) 
    mae_var = np.var(absolute_errors) 
    mse_var = np.var(squared_errors) 
    return mae_mean, mse_mean, mae_var, mse_var

def relative_error(y_true, y_pred):
    """
    Calculate the relative error between true and predicted values.
    """
    # Avoid division by zero
    relative_error_mse = mean_squared_error(y_true, y_pred)
    return relative_error_mse

# 载入Excel文件
file_path = 'drug.xlsx'
# 分别读取两个sheet(训练数据和测试数据)
train_data = pd.read_excel(file_path, sheet_name=0, header=None)
test_data = pd.read_excel(file_path, sheet_name=1, header=None)

# 根据药物simles生成分子指纹特征
def get_smiles_feature(data):
    smi_dicts={
    '奥氮平': "CN1CCN(CC1)C1=NC2=CC=CC=C2NC2=C1C=C(C)S2",
    '氟哌啶醇': "OC1(CCN(CCCC(=O)C2=CC=C(F)C=C2)CC1)C1=CC=C(Cl)C=C1",
    '齐拉西酮': "ClC1=C(CCN2CCN(CC2)C2=NSC3=CC=CC=C23)C=C2CC(=O)NC2=C1",
    '阿立哌唑': "ClC1=CC=CC(N2CCN(CCCCOC3=CC4=C(CCC(=O)N4)C=C3)CC2)=C1Cl",
    '利培酮': "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1",
    '喹硫平': "OCCOCCN1CCN(CC1)C1=NC2=CC=CC=C2SC2=CC=CC=C12",
    '奋乃静': "OCCN1CCN(CCCN2C3=CC=CC=C3SC3=C2C=C(Cl)C=C3)CC1"
    }
    smi_finger={}
    for key in smi_dicts:
        smi_mol = Chem.MolFromSmiles(smi_dicts[key])
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(smi_mol, 2, nBits=16)
        smi_finger[key] = [int(bit) for bit in fingerprint.ToBitString()]

    return np.array(data.map(smi_finger).to_list())

# 数据预处理
# 将药物类型进行独热编码
# encoder = OneHotEncoder(sparse=False)
# train_drug_encoded = encoder.fit_transform(train_data.iloc[:, 30:31])
# test_drug_encoded = encoder.transform(test_data.iloc[:, 30:31])

# 将药物类型进行分子指纹编码
train_drug_encoded = get_smiles_feature(train_data.iloc[:, 30])
test_drug_encoded = get_smiles_feature(test_data.iloc[:, 30])

X_train  = train_data.iloc[:, :30].values
X_test = test_data.iloc[:, :30].values

#合并编码后的药物类型和其他特征
X_train = np.hstack((X_train, train_drug_encoded))
y_train = train_data.iloc[:, 31].values
X_test = np.hstack((X_test, test_drug_encoded))
y_test = test_data.iloc[:, 31].values

#数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#Ridge模型
ridge = Ridge()

# 定义超参数网格
param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100, 500,900,1000,1100, 1500,2000,5000,10000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# MSE得分器，注意岭回归是回归问题的模型
mse_scorer = make_scorer(relative_error, greater_is_better=False)

# 使用网格搜索
best_ridge_model = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring=mse_scorer)
best_ridge_model.fit(X_train, y_train)

# 打印最优参数组合
print("最优岭回归参数组合:", best_ridge_model.best_params_)
relative_error_ridge_complex = evaluate_model(best_ridge_model, X_test, y_test)
print(f"final_relative_error_ridge_complex:{relative_error_ridge_complex}")
# 转换grid_search_ridge.cv_results_为DataFrame
results_df = pd.DataFrame(best_ridge_model.cv_results_)

# 由于我们只对alpha参数进行了网格搜索，我们将仅可视化这一个维度
# 如果有多个参数，可能需要不同的可视化策略

# 提取alpha参数和对应的测试得分
alpha_scores = results_df[['param_alpha', 'mean_test_score']]

# 因为我们使用了负MSE，所以取反以显示正的MSE值
alpha_scores['mean_test_score'] = -alpha_scores['mean_test_score']

# 绘制alpha参数和MSE之间的关系
plt.figure(figsize=(8, 6))
sns.lineplot(x='param_alpha', y='mean_test_score', data=alpha_scores, marker='o')

plt.xscale('log')  # 因为alpha是对数尺度，所以我们可以使用对数尺度来显示x轴
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Alpha for Ridge Regression')
#plt.savefig('MSE vs. Alpha for Ridge Regression') 

predictions_stacking = best_ridge_model.predict(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, predictions_stacking, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 对角线
#plt.savefig('MSE vs. Alpha for Ridge Regression_X_train')

predictions_stacking = best_ridge_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_stacking, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 对角线
#plt.savefig('MSE vs. Alpha for Ridge Regression_X_test')

results = pd.DataFrame(best_ridge_model.cv_results_)

# Pivot the data to create a matrix format suitable for a heatmap
# Assuming 'param_alpha' and 'param_solver' are the names of the hyperparameters
pivot_table = results.pivot("param_alpha", "param_solver", "mean_test_score")

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Cross-Validation Results for Ridge Model")
plt.xlabel("Solver")
plt.ylabel("Alpha")
plt.show()

plt.savefig("ridge.png")

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [5, 50, 200, 1000],
    'max_depth': [1, 2, 5, 10, 100],
    # "min_samples_split":[2],
    # "min_samples_leaf":[1],
    # "max_features":["auto"],
    # "bootstrap":[True]
}
# Initialize Random Forest model
rf_model = RandomForestRegressor()
# Setup GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring=mse_scorer)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
best_parameters_rf = grid_search_rf.best_params_
# 打印最优参数组合
print("rf最优参数组合:", best_parameters_rf)
relative_error_rf_complex = evaluate_model(best_rf_model, X_test, y_test)
print(f"error_rf:{relative_error_rf_complex}")
# Assuming grid_search_rf is your GridSearchCV object for Random Forest
results = pd.DataFrame(grid_search_rf.cv_results_)
pivot_table = results.pivot('param_n_estimators', 'param_max_depth', 'mean_test_score')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
#sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title('Random Forest MSE Grid Search Scores')
plt.xlabel('Max Depth')
plt.ylabel('Number of Estimators')
plt.show()
plt.savefig("rf.png")

# 创建XGBoost模型实例
xgb_model = xgb.XGBRegressor()

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 15],
    'learning_rate': [0.001, 0.01, 0.1, 0.5],
    'n_estimators': [10, 50, 200, 300],
    'subsample': [0.2, 0.5, 0.9, 1.0]
}
# 创建GridSearchCV实例
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring=mse_scorer)

# 对数据进行拟合
grid_search.fit(X_train, y_train)
# Find the best parameters based on RMSE
best_xgb_model = grid_search.best_estimator_
best_parameters_xgb = grid_search.best_params_
print("xgb最优参数组合:", best_parameters_xgb)
relative_error_xgb = evaluate_model(best_xgb_model, X_test, y_test)
print(f"error_xgb:{relative_error_xgb}")


# Convert the cv_results to a DataFrame for easier manipulation
results_df = pd.DataFrame(grid_search.cv_results_)  # Replace grid_search with your GridSearchCV object

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3 hyperparameters against the mean test score
sc = ax.scatter(
    results_df['param_max_depth'],
    results_df['param_learning_rate'],
    results_df['param_n_estimators'],
    c=-results_df['mean_test_score'],  # Assuming mean_test_score is negative MSE
    cmap='viridis'
)

# Labels and title
ax.set_xlabel('Max Depth')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('N Estimators')
plt.colorbar(sc, label='Mean Squared Error')
plt.title('3D Scatter Plot of XGBoost Hyperparameters')
plt.savefig('3D Scatter Plot of XGBoost Hyperparameters.png')


#MLP

param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (100, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive'],
}

mlp = MLPRegressor(max_iter=10,early_stopping=True)
mlp_grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring=mse_scorer)
mlp_grid_search.fit(X_train, y_train)
best_mlp_model = mlp_grid_search.best_estimator_
best_parameters_mlp = mlp_grid_search.best_params_
print("mlp最优参数组合:", best_parameters_mlp)
relative_error_mlp = evaluate_model(best_mlp_model, X_test, y_test)
print(f"error_mlp:{relative_error_mlp}")

# Convert the cv_results to a DataFrame for easier manipulation
results_df = pd.DataFrame(mlp_grid_search.cv_results_)  # Replace grid_search with your GridSearchCV object

# Convert tuples to a string or a numerical representation for plotting
# Here we sum the sizes of the layers in the tuple
results_df['param_hidden_layer_sizes'] = results_df['param_hidden_layer_sizes'].apply(lambda x: sum(x))

# Example: 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    results_df['param_hidden_layer_sizes'],
    results_df['param_activation'].apply(lambda x: {'tanh': 0, 'relu': 1}[x]),  # Convert activation to numerical
    results_df['param_alpha'],
    c=results_df['mean_test_score'],
    cmap='viridis'
)
ax.set_xlabel('Total Number of Hidden Layer Nodes')
ax.set_ylabel('Activation (0=tanh, 1=relu)')
ax.set_zlabel('Alpha')
plt.colorbar(sc, label='Mean Test Score')
plt.title('3D Scatter Plot of MLP Hyperparameters')
plt.savefig('3D Scatter Plot of MLP Hyperparameters.png')

# Define hyperparameter grid for GBM
gbm_param_grid = {
    'n_estimators': [10, 50, 150, 300, 500],
    'learning_rate': [0.001, 0.01, 0.2, 0.5],
    'max_depth': [3, 5, 7, 12],
}

# Initialize GBM model
gbm_model = GradientBoostingRegressor()
# Setup GridSearchCV for GBM
grid_search_gbm = GridSearchCV(estimator=gbm_model, param_grid=gbm_param_grid, cv=5, scoring=mse_scorer)
grid_search_gbm.fit(X_train, y_train)
best_gbm_model = grid_search_gbm.best_estimator_
relative_error_gbm = evaluate_model(best_gbm_model, X_test, y_test)
best_parameters_gbm = grid_search_gbm.best_params_
print("gbm最优参数组合:", best_parameters_gbm)
print(f"error_gbm:{relative_error_gbm}")
results_df = pd.DataFrame(grid_search_gbm.cv_results_)  # Replace grid_search with your GridSearchCV object

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3 hyperparameters against the mean test score
sc = ax.scatter(
    results_df['param_n_estimators'],
    results_df['param_learning_rate'],
    results_df['param_max_depth'],
    c=-results_df['mean_test_score'],  # Assuming mean_test_score is negative MSE
    cmap='viridis'
)

# Labels and title
ax.set_xlabel('MN Estimators')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Max Depth')
plt.colorbar(sc, label='Mean Squared Error')
plt.title('3D Scatter Plot of GBM Hyperparameters')
plt.savefig('3D Scatter Plot of GBM Hyperparameters.png')


# Stacking Model with multiple estimators
stacking_model = StackingRegressor(
    estimators=[
        ('ridge', best_ridge_model),
        ('rf', best_rf_model),
        ('gbm', best_gbm_model),
        #('xgb', best_xgb_model),
        ('mlp', best_mlp_model)
    ],
    final_estimator=MLPRegressor()
    #final_estimator=GradientBoostingRegressor()
)
stacking_model.fit(X_train, y_train)
# 评估模型性能
stacked_mse = evaluate_model(stacking_model, X_test, y_test)
print(f'error_Stacked Model: {stacked_mse}')

predictions_stacking = stacking_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_stacking, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('stacking model Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 对角线
plt.savefig("8.png")
# List of models
models = [
    best_ridge_model,
    best_rf_model,
    best_xgb_model,
    best_mlp_model,
    best_gbm_model,
    stacking_model
]


def get_error_evaluation(model, x_test, y_test):
    predictions = model.predict(x_test)
    absolute_errors = np.abs(y_test - predictions)
    squared_errors = (y_test - predictions) ** 2
    mae_mean = np.mean(absolute_errors)
    mse_mean = np.mean(squared_errors) 
    mae_var = np.var(absolute_errors) 
    mse_var = np.var(squared_errors) 
    return mae_mean, mse_mean, mae_var, mse_var

# Model names
model_names = ['Ridge', 'Random Forest', 'XGBoost', 'MLP', 'GBM', 'Stacking']

mse_values = []
mse_std_values = []
mae_values = []
mae_std_values = []
# Calculate MSE and MSE standard deviation for each model
for model in models:
    mae_mean, mse_mean, mae_var, mse_var = get_error_evaluation(model, X_test, y_test)
    mse_values.append(mse_mean)
    mse_std_values.append(mse_var)
    mae_values.append(mae_mean)
    mae_std_values.append(mae_var)

# Plotting
plt.figure(figsize=(10, 6))

# Creating a bar plot for MSE values with error bars for the standard deviation
plt.bar(model_names, mse_values, color='blue', alpha=0.7, capsize=5)

# Adding the MSE ± MSE_std as text on the bars
for i in range(len(model_names)):
    plt.text(i, mse_values[i] + mse_std_values[i], f'{mse_values[i]:.2f}±{mse_std_values[i]:.2f}', ha='center', va='bottom')

plt.title('Mean Squared Error (MSE) ± Standard Deviation of Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.savefig("9.png")

# Plotting
plt.figure(figsize=(10, 6))

# Creating a bar plot for MSE values with error bars for the standard deviation
plt.bar(model_names, mae_values, color='blue', alpha=0.7, capsize=5)

# Adding the MSE ± MSE_std as text on the bars
for i in range(len(model_names)):
    plt.text(i, mae_values[i] + mae_std_values[i], f'{mae_values[i]:.2f}±{mae_std_values[i]:.2f}', ha='center', va='bottom')

plt.title('Mean Absolute Error (MAE) ± Standard Deviation of Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.savefig("10.png")

