# DuDoanGiaChungKhoan_Machinglearning
# Stock Price Prediction System

Hệ thống dự đoán giá cổ phiếu sử dụng multiple machine learning models, tập trung vào 5 mã cổ phiếu công nghệ hàng đầu (AAPL, MSFT, GOOGL, META, AMZN).

## Tính năng chính

- **Thu thập dữ liệu tự động** từ Yahoo Finance
- **Xử lý dữ liệu thông minh:**
  - Tự động phát hiện và xử lý dữ liệu trùng lặp
  - Kiểm tra và báo cáo các vấn đề về chất lượng dữ liệu
  - Tạo đặc trưng kỹ thuật (technical features)

- **Đa dạng mô hình ML:**
  - XGBoost
  - LightGBM
  - Random Forest
  - Support Vector Regression
  - K-Nearest Neighbors
  - Ridge Regression

- **Feature Engineering:**
  - Moving Averages (MA5, MA20, MA50)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume Indicators
  - Volatility

## Cấu trúc Project
├── app.py # Flask API cho dự đoán
├── stock_prediction.py # Core logic và training
├── models/ # Pre-trained models
│ ├── AAPL/
│ ├── MSFT/
│ └── ...
├── data/ # Dữ liệu gốc và đặc trưng
│ ├── raw_data/
│ └── features/
└── data_issues/ # Báo cáo vấn đề dữ liệu


## Yêu cầu hệ thống

- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, yfinance, flask

## Cài đặt

```bash
git clone [repository-url]
cd stock-prediction-system
pip install -r requirements.txt
```

## Sử dụng

### Training mô hình mới

```python
python stock_prediction.py
```

### Chạy API dự đoán

```python
python app.py
```

## API Endpoints

- `GET /predict/<ticker>`: Dự đoán giá đóng cửa ngày tiếp theo
- `GET /models/<ticker>`: Thông tin về model hiện tại
- `GET /performance/<ticker>`: Metrics hiệu suất của model

## Đóng góp

Các hướng phát triển tiếp theo:
1. Tích hợp thêm nguồn dữ liệu
2. Thêm mô hình deep learning
3. Xây dựng dashboard monitoring
4. Mở rộng khung thời gian dự đoán
5. Tối ưu hiệu suất hệ thống

## License

MIT License

## Tác giả

[Your Name]

## Ghi nhận

- Dữ liệu được cung cấp bởi Yahoo Finance
- Sử dụng các thư viện ML: scikit-learn, XGBoost, LightGBM
