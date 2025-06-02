import sys
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, render_template, request, jsonify, current_app
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from stock_prediction import prepare_features
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
from functools import lru_cache
from datetime import datetime, timedelta
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache cho models để tránh load lại nhiều lần
model_cache = {}
CACHE_DURATION = timedelta(hours=1)
last_cache_update = {}

# Danh sách mã cổ phiếu được hỗ trợ
SUPPORTED_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']

class ModelLoadError(Exception):
    """Custom exception cho lỗi load model"""
    pass

class DataProcessError(Exception):
    """Custom exception cho lỗi xử lý dữ liệu"""
    pass

def validate_request_data(ticker, start_date, end_date):
    """
    Kiểm tra tính hợp lệ của dữ liệu đầu vào
    """
    errors = []
    
    # Kiểm tra ticker
    if not ticker:
        errors.append("Thiếu mã cổ phiếu")
    elif ticker not in SUPPORTED_TICKERS:
        errors.append(f"Mã cổ phiếu {ticker} không được hỗ trợ")
    
    # Kiểm tra ngày
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            errors.append("Ngày bắt đầu phải trước ngày kết thúc")
        
        if end > datetime.now():
            errors.append("Ngày kết thúc không thể trong tương lai")
            
        if (end - start).days < 30:
            errors.append("Khoảng thời gian phải ít nhất 30 ngày")
            
    except ValueError:
        errors.append("Định dạng ngày không hợp lệ (YYYY-MM-DD)")
    
    return errors

@lru_cache(maxsize=10)
def load_saved_models(ticker):
    """
    Load các model, PCA và scaler đã được lưu sẵn cho mã cổ phiếu với cache
    """
    # Kiểm tra cache
    if ticker in model_cache:
        last_update = last_cache_update.get(ticker)
        if last_update and datetime.now() - last_update < CACHE_DURATION:
            return model_cache[ticker]

    try:
        models = {}
        models_dir = os.path.join('models', ticker)
        
        if not os.path.exists(models_dir):
            raise ModelLoadError(f"Không tìm thấy model cho mã {ticker}")
            
        # Load PCA và scaler
        pca_path = os.path.join(models_dir, 'pca.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        
        if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
            raise ModelLoadError(f"Không tìm thấy file PCA hoặc scaler cho mã {ticker}")
            
        pca = joblib.load(pca_path)
        scaler = joblib.load(scaler_path)
            
        model_files = {
            'XGBoost': 'XGBoost_model.pkl',
            'LightGBM': 'LightGBM_model.pkl',
            'KNN': 'KNN_model.pkl',
            'RandomForest': 'RandomForest_model.pkl',
            'SVR': 'SVR_model.pkl',
            'Ridge': 'Ridge_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                
        if not models:
            raise ModelLoadError(f"Không tìm thấy model nào cho mã {ticker}")
        
        # Cập nhật cache
        model_cache[ticker] = (models, pca, scaler)
        last_cache_update[ticker] = datetime.now()
        
        return models, pca, scaler
        
    except Exception as e:
        raise ModelLoadError(f"Lỗi khi load models: {str(e)}")

def process_stock_data(ticker, start_date, end_date):
    """
    Xử lý dữ liệu cổ phiếu và chuẩn bị features
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise DataProcessError(f"Không tìm thấy dữ liệu cho {ticker}")
            
        X, y = prepare_features(df)
        
        if len(X) < 30:
            raise DataProcessError('Không đủ dữ liệu để dự đoán. Vui lòng chọn khoảng thời gian dài hơn.')
        
        # Đảm bảo thứ tự các features
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                           'Returns', 'MA5', 'MA20', 'MA50', 'RSI', 
                           'MACD', 'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower',
                           'Volume_MA5', 'Volume_MA20', 'Volatility']
        X = X[expected_features]
        
        return X, y, df
        
    except Exception as e:
        raise DataProcessError(f"Lỗi khi xử lý dữ liệu: {str(e)}")

def apply_preprocessing(X, scaler, pca):
    """
    Áp dụng preprocessing (scaling và PCA)
    """
    try:
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        return X_scaled, X_pca
    except Exception as e:
        raise DataProcessError(f"Lỗi khi áp dụng preprocessing: {str(e)}")

def create_prediction_plot(y_test, predictions, ticker):
    """
    Tạo đồ thị dự đoán
    """
    try:
        plt.figure(figsize=(12, 6))
        dates = y_test.index
        
        plt.plot(dates, y_test.values, label='Thực tế', linewidth=2)
        
        # Hiển thị 3 mô hình tốt nhất
        for name, pred in predictions.items():
            plt.plot(dates, pred, label=f'Dự đoán {name}', linestyle='--')
        
        plt.title(f'Dự đoán giá cổ phiếu cho {ticker}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá ($)')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.legend()
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    except Exception as e:
        raise Exception(f"Lỗi khi tạo đồ thị: {str(e)}")

def evaluate_prediction_models(models, X_test_pca, y_test):
    """
    Đánh giá các mô hình dự đoán với dữ liệu đã qua PCA
    """
    try:
        results = {}
        predictions = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_pca)
            mse = np.mean((y_test - y_pred) ** 2)
            r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            results[name] = {
                'MSE': float(mse),
                'R2': float(r2)
            }
            predictions[name] = y_pred
            
        return results, predictions
        
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá mô hình: {str(e)}")
        raise Exception(f"Lỗi khi đánh giá mô hình: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy và validate dữ liệu từ form
        ticker = request.form.get('ticker', '').strip()
        start_date = request.form.get('start_date', '').strip()
        end_date = request.form.get('end_date', '').strip()
        
        logger.info(f"Nhận request dự đoán cho {ticker} từ {start_date} đến {end_date}")
        
        # Validate dữ liệu đầu vào
        errors = validate_request_data(ticker, start_date, end_date)
        if errors:
            logger.warning(f"Dữ liệu đầu vào không hợp lệ: {errors}")
            return jsonify({'error': '. '.join(errors)}), 400
        
        # Load models
        models, pca, scaler = load_saved_models(ticker)
        
        # Xử lý dữ liệu
        X, y, df = process_stock_data(ticker, start_date, end_date)
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Áp dụng preprocessing
        try:
            # Transform dữ liệu test
            X_test_scaled = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_scaled)
            
            logger.info(f"Đã áp dụng preprocessing thành công. Shape sau PCA: {X_test_pca.shape}")
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng preprocessing: {str(e)}")
            raise DataProcessError(f"Lỗi khi áp dụng preprocessing: {str(e)}")
        
        # Đánh giá mô hình với dữ liệu đã qua PCA
        results, predictions = evaluate_prediction_models(models, X_test_pca, y_test)
        
        # Tạo đồ thị
        plot_url = create_prediction_plot(y_test, 
                                        {k: v for k, v in predictions.items() if k in dict(sorted(results.items(), key=lambda x: x[1]['MSE'])[:3])},
                                        ticker)
        
        # Dự đoán cho ngày tiếp theo sử dụng dữ liệu đã qua PCA
        best_model_name = min(results.items(), key=lambda x: x[1]['MSE'])[0]
        next_day_features = X_test_pca[-1:] # Lấy dữ liệu ngày cuối cùng đã qua PCA
        next_day_prediction = models[best_model_name].predict(next_day_features)
        
        response_data = {
            'success': True,
            'results': results,
            'plot': plot_url,
            'next_day_prediction': float(next_day_prediction[0]),
            'current_price': float(df['Close'].iloc[-1])
        }
        
        logger.info(f"Dự đoán thành công cho {ticker}")
        return jsonify(response_data)
        
    except ModelLoadError as e:
        logger.error(f"Lỗi load model: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except DataProcessError as e:
        logger.error(f"Lỗi xử lý dữ liệu: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
        return jsonify({'error': f"Lỗi không xác định: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True) 