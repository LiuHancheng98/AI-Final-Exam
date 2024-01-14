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
    error_mse = mean_squared_error(y_test, predictions)
    return error_mse


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


if __name__ == "__main__":
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
    mse_error_ridge = evaluate_model(best_ridge_model, X_test, y_test)
    print(f"mse_error_ridge:{mse_error_ridge}")



    # Initialize Random Forest model
    rf_model = RandomForestRegressor()

    # Define hyperparameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [5, 50, 200, 1000],
        'max_depth': [1, 2, 5, 10, 100],
    }

    # Setup GridSearchCV for Random Forest
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring=mse_scorer)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    best_parameters_rf = grid_search_rf.best_params_
    # 打印最优参数组合
    print("rf最优参数组合:", best_parameters_rf)
    mse_error_rf = evaluate_model(best_rf_model, X_test, y_test)
    print(f"mse_error_rf:{mse_error_rf}")



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
    best_xgb_model = grid_search.best_estimator_
    best_parameters_xgb = grid_search.best_params_
    print("xgb最优参数组合:", best_parameters_xgb)
    mse_error_xgb = evaluate_model(best_xgb_model, X_test, y_test)
    print(f"mse_error_xgb:{mse_error_xgb}")



    #MLP
    mlp = MLPRegressor(max_iter=10, early_stopping=True)

    param_grid = {
        'hidden_layer_sizes': [(25,), (50,), (100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['adaptive'],
    }

    mlp_grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring=mse_scorer)
    mlp_grid_search.fit(X_train, y_train)
    best_mlp_model = mlp_grid_search.best_estimator_
    best_parameters_mlp = mlp_grid_search.best_params_
    print("mlp最优参数组合:", best_parameters_mlp)
    mse_error_mlp = evaluate_model(best_mlp_model, X_test, y_test)
    print(f"error_mlp:{mse_error_mlp}")


    # Initialize GBM model
    gbm_model = GradientBoostingRegressor()

    # Define hyperparameter grid for GBM
    gbm_param_grid = {
        'n_estimators': [10, 50, 150, 300, 500],
        'learning_rate': [0.001, 0.01, 0.2, 0.5],
        'max_depth': [3, 5, 7, 12],
    }

    # Setup GridSearchCV for GBM
    grid_search_gbm = GridSearchCV(estimator=gbm_model, param_grid=gbm_param_grid, cv=5, scoring=mse_scorer)
    grid_search_gbm.fit(X_train, y_train)
    best_gbm_model = grid_search_gbm.best_estimator_
    mse_error_gbm = evaluate_model(best_gbm_model, X_test, y_test)
    best_parameters_gbm = grid_search_gbm.best_params_
    print("gbm最优参数组合:", best_parameters_gbm)
    print(f"mse_error_gbm:{mse_error_gbm}")



    # Stacking Model with multiple estimators
    stacking_model = StackingRegressor(
        estimators=[
            ('ridge', best_ridge_model),
            ('rf', best_rf_model),
            ('gbm', best_gbm_model),
            ('mlp', best_mlp_model)
        ],
        final_estimator=MLPRegressor()
    )
    stacking_model.fit(X_train, y_train)
    # 评估模型性能
    mse_error_Stacked = evaluate_model(stacking_model, X_test, y_test)
    print(f'error_Stacked Model: {mse_error_Stacked}')


    # List of models
    models = [
        best_ridge_model,
        best_rf_model,
        best_xgb_model,
        best_mlp_model,
        best_gbm_model,
        stacking_model
    ]


    # Model names
    model_names = ['Ridge', 'Random Forest', 'XGBoost', 'MLP', 'GBM', 'Stacking']

    mse_values = []
    mse_var_values = []
    mae_values = []
    mae_var_values = []
    # Calculate MSE and MSE standard deviation for each model
    for model in models:
        mae_mean, mse_mean, mae_var, mse_var = get_error_evaluation(model, X_test, y_test)
        mse_values.append(mse_mean)
        mse_var_values.append(mse_var)
        mae_values.append(mae_mean)
        mae_var_values.append(mae_var)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Creating a bar plot for MSE values with error bars for the standard deviation
    plt.bar(model_names, mse_values, color='blue', alpha=0.7, capsize=5)

    # 在条形图上添加 MSE ± 方差的文本
    for i in range(len(model_names)):
        plt.text(i, mse_values[i], f'{mse_values[i]:.2f}±{mse_var_values[i]:.2f}', ha='center', va='bottom')

    plt.title('Mean Squared Error (MSE) ± Variance of Models')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    # 调整 y 轴的刻度范围以放大高度差异
    plt.ylim(min(mse_values) - 15, max(mse_values) + 10)
    # Show plot
    plt.tight_layout()
    plt.show()

