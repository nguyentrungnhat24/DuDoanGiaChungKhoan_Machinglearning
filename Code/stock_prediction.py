import sys
sys.stdout.reconfigure(encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def get_stock_data(ticker, start_date, end_date):
    """
    Lấy dữ liệu chứng khoán từ yfinance
    """
    try:
        print(f"Đang tải dữ liệu cho {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: Không có dữ liệu cho {ticker} trong khoảng thời gian đã chọn")
            return pd.DataFrame()
            
        print(f"Đã lấy {len(df)} dòng dữ liệu cho {ticker}")
        if not df.empty:
            print(f"Phạm vi thời gian: {df.index.min()} đến {df.index.max()}")
        
        return df
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu cho {ticker}: {str(e)}")
        return pd.DataFrame()

def prepare_features(df, window_size=5):
    """
    Chuẩn bị features cho mô hình
    """
    # Tính toán các chỉ báo kỹ thuật
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD Moving Average Convergence Divergence
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    # Tạo target (giá đóng cửa ngày tiếp theo)
    df['Target'] = df['Close'].shift(-1)
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    # Chọn features theo thứ tự cố định
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 
                'MACD', 'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 
                'Volume_MA5', 'Volume_MA20', 'Volatility']
    
    return df[features], df['Target']

def calculate_rsi(prices, period=14):
    """
    Tính toán chỉ báo RSI
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_preprocessing(X_train, n_components=0.95):
    """
    Fit scaler và PCA với dữ liệu training
    """
    print("Đang fit scaler và PCA...")
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    
    # In thông tin về PCA
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(explained_variance_ratio)
    
    print(f"Số thành phần chính: {n_components}")
    print(f"Tổng phương sai giải thích được: {total_variance:.4f} ({total_variance*100:.2f}%)")
    
    return scaler, pca

def apply_preprocessing(X, scaler, pca):
    """
    Áp dụng scaler và PCA đã được fit
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca

def train_models(X_train, y_train):
    """
    Huấn luyện các mô hình
    """
    print("Đang huấn luyện các mô hình...")
    models = {}
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    
    # K-Nearest Neighbors
    print("Training KNN...")
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    models['KNN'] = knn_model
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # Support Vector Regression
    print("Training SVR...")
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    models['SVR'] = svr_model
    
    # Ridge Regression
    print("Training Ridge...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    models['Ridge'] = ridge_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Đánh giá các mô hình
    """
    results = {}
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        predictions[name] = y_pred
    
    # Tạo bảng so sánh các mô hình
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[model]['MSE'] for model in results],
        'R2': [results[model]['R2'] for model in results]
    })
    
    comparison = comparison.sort_values('MSE')
    print("\nSo sánh hiệu suất các mô hình:")
    print(comparison)
    
    return results, predictions

def save_models(models, pca, scaler, ticker):
    """
    Lưu các mô hình, PCA và scaler
    """
    save_dir = os.path.join('models', ticker)
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Lưu các mô hình
    for name, model in models.items():
        model_path = os.path.join(save_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Đã lưu {name} model vào {model_path}")
    
    # Lưu PCA và scaler
    pca_path = os.path.join(save_dir, "pca.pkl")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    
    joblib.dump(pca, pca_path)
    joblib.dump(scaler, scaler_path)
    print(f"Đã lưu PCA và scaler vào {save_dir}")

def train_stock_model(ticker, start_date, end_date):
    """
    Huấn luyện mô hình cho một mã cổ phiếu
    """
    print(f"\nBắt đầu huấn luyện mô hình cho {ticker}")
    
    # 1. Lấy và chuẩn bị dữ liệu
    df = get_stock_data(ticker, start_date, end_date)
    if df.empty:
        print(f"Không thể huấn luyện mô hình cho {ticker} do không có dữ liệu")
        return
    
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 2. Fit scaler và PCA
    scaler, pca = train_preprocessing(X_train)
    
    # 3. Transform dữ liệu
    X_train_transformed = apply_preprocessing(X_train, scaler, pca)
    X_test_transformed = apply_preprocessing(X_test, scaler, pca)
    
    # 4. Train models
    models = train_models(X_train_transformed, y_train)
    
    # 5. Đánh giá models
    results, predictions = evaluate_models(models, X_test_transformed, y_test)
    
    # 6. Lưu models, scaler và PCA
    save_models(models, pca, scaler, ticker)
    
    print(f"\nHoàn thành huấn luyện mô hình cho {ticker}")
    return models, scaler, pca

def main():
    """
    Hàm chính để huấn luyện mô hình cho các mã cổ phiếu
    """
    # Danh sách các mã cổ phiếu cần train
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
    
    # Khoảng thời gian lấy dữ liệu
    start_date = '2021-01-01'
    end_date = '2025-01-01'
    
    for ticker in tickers:
        # Lấy dữ liệu và kiểm tra chất lượng
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            # Kiểm tra chất lượng dữ liệu
            df, is_valid = check_data_quality(df, ticker)
            if is_valid:
                # Chuẩn bị features
                X, y = prepare_features(df)
                # Lưu dữ liệu vào CSV
                save_to_csv(df, X, y, ticker)
                # Train model
                train_stock_model(ticker, start_date, end_date)
            else:
                print(f"Dữ liệu của {ticker} không đạt yêu cầu chất lượng")

def check_data_quality(df, file_prefix):
    """
    Kiểm tra chất lượng dữ liệu và xuất ra các dòng có vấn đề.
    Tự động xử lý dữ liệu trùng lặp bằng cách giữ lại dòng đầu tiên.
    """
    issues = []
    data_modified = False
    
    # Kiểm tra và xử lý dữ liệu trùng lặp
    duplicates = df[df.index.duplicated(keep=False)]
    if not duplicates.empty:
        issues.append({
            'issue_type': 'Duplicates',
            'data': duplicates,
            'description': 'Các dòng có ngày trùng lặp - Đã giữ lại dòng đầu tiên'
        })
        # Xử lý duplicates - giữ lại dòng đầu tiên
        df = df[~df.index.duplicated(keep='first')]
        data_modified = True
        print(f"Đã xử lý {len(duplicates)//2} cặp dòng trùng lặp")
    
    # Kiểm tra dữ liệu thiếu (NaN)
    missing_data = df[df.isnull().any(axis=1)]
    if not missing_data.empty:
        issues.append({
            'issue_type': 'Missing',
            'data': missing_data,
            'description': 'Các dòng có dữ liệu thiếu'
        })
    
    # Kiểm tra giá trị bất thường (ví dụ: giá âm)
    price_columns = ['Open', 'High', 'Low', 'Close']
    invalid_prices = df[df[price_columns].le(0).any(axis=1)]
    if not invalid_prices.empty:
        issues.append({
            'issue_type': 'Invalid_Prices',
            'data': invalid_prices,
            'description': 'Các dòng có giá <= 0'
        })
    
    # Kiểm tra volume bất thường
    invalid_volume = df[df['Volume'] <= 0]
    if not invalid_volume.empty:
        issues.append({
            'issue_type': 'Invalid_Volume',
            'data': invalid_volume,
            'description': 'Các dòng có volume <= 0'
        })
    
    # Kiểm tra High < Low
    invalid_hl = df[df['High'] < df['Low']]
    if not invalid_hl.empty:
        issues.append({
            'issue_type': 'Invalid_HL',
            'data': invalid_hl,
            'description': 'Các dòng có High < Low'
        })
    
    # Nếu có vấn đề, xuất ra file
    if issues:
        # Tạo thư mục data_issues nếu chưa tồn tại
        if not os.path.exists('data_issues'):
            os.makedirs('data_issues')
        
        # Xuất từng loại vấn đề ra file riêng
        for issue in issues:
            issue_file = f'data_issues/{file_prefix}_{issue["issue_type"]}.csv'
            issue['data'].to_csv(issue_file)
            print(f"Đã lưu {issue['description']} vào: {issue_file}")
        
        # Tạo file tổng hợp
        summary_file = f'data_issues/{file_prefix}_quality_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Báo cáo chất lượng dữ liệu cho {file_prefix}\n")
            f.write(f"Thời gian tạo: {pd.Timestamp.now()}\n\n")
            
            for issue in issues:
                f.write(f"{issue['description']}:\n")
                f.write(f"Số lượng dòng: {len(issue['data'])}\n")
                f.write("-------------------\n")
        
        print(f"Đã lưu báo cáo tổng hợp vào: {summary_file}")
        
        # Nếu chỉ có vấn đề duplicates và đã xử lý, vẫn cho phép tiếp tục
        if len(issues) == 1 and 'Duplicates' in issues[0]['issue_type'] and data_modified:
            return df, True
            
        return df, False
    
    return df, True

def save_to_csv(df, features, target, file_prefix):
    """
    Lưu dữ liệu gốc và đặc trưng ra file CSV
    """
    # Tạo thư mục data nếu chưa tồn tại
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Lưu dữ liệu gốc
    raw_data_file = f'data/{file_prefix}_raw_data.csv'
    df.to_csv(raw_data_file)
    print(f"Đã lưu dữ liệu gốc vào: {raw_data_file}")
    
    # Lưu đặc trưng và target
    features_data = pd.DataFrame(features)
    features_data['Target'] = target
    features_file = f'data/{file_prefix}_features.csv'
    features_data.to_csv(features_file)
    print(f"Đã lưu đặc trưng vào: {features_file}")

if __name__ == "__main__":
    main() 