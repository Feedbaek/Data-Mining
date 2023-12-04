import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

def ARIMAM(filepath):
    # 데이터 전처리
    movie_data = pd.read_csv(filepath)
    movie_data['DATE'] = pd.to_datetime(movie_data['BASE_YEAR'].astype(str) + '-' +
                                        movie_data['BASE_MT'].astype(str) + '-' +
                                        movie_data['BASE_DAY'].astype(str))
    movie_data = movie_data.set_index('DATE')

    # 2019년부터 2022년까지 데이터 필터링
    train_data = movie_data[movie_data['BASE_YEAR'].between(2019, 2022)]

    # 2023년 데이터 필터링
    test_data = movie_data[movie_data['BASE_YEAR'] == 2023]

    # 시계열 데이터 추출 (전국 영화 관람객 수)
    ts_train = train_data['MOVIE_ADNC_CO'].resample('W').sum()  # 주간 데이터로 리샘플링
    ts_test = test_data['MOVIE_ADNC_CO'].resample('W').sum()

    # ARIMA 모델 적합
    model = ARIMA(ts_train, order=(1, 1, 1))
    model_fit = model.fit()

    # 2023년 데이터에 대한 예측
    predictions = model_fit.forecast(steps=len(ts_test))
    
    return model_fit, ts_train,ts_test,predictions


def VARM(filepath):
    # 다변량 시계열 데이터 준비
    # 'MOVIE_ADNC_CO'와 다른 컬럼들(예: 'KOREA_MOVIE_ADNC_CO', 'OVSEA_MOVIE_ADNC_CO')을 포함
    movie_data = pd.read_csv(filepath)
    movie_data['DATE'] = pd.to_datetime(movie_data['BASE_YEAR'].astype(str) + '-' +
                                            movie_data['BASE_MT'].astype(str) + '-' +
                                            movie_data['BASE_DAY'].astype(str))

    movie_data = movie_data.set_index('DATE')
    train = movie_data[movie_data['BASE_YEAR'].between(2019, 2022)].resample('W').sum()
    test = movie_data[movie_data['BASE_YEAR'] == 2023].resample('W').sum()
    multivariate_train = train[['MOVIE_ADNC_CO', 'KOREA_MOVIE_ADNC_CO', 'OVSEA_MOVIE_ADNC_CO']]
    multivariate_test = test[['MOVIE_ADNC_CO', 'KOREA_MOVIE_ADNC_CO', 'OVSEA_MOVIE_ADNC_CO']]

    # 2019년부터 2022년까지 데이터 필터링
    #multivariate_data = movie_data[['MOVIE_ADNC_CO', 'KOREA_MOVIE_ADNC_CO', 'OVSEA_MOVIE_ADNC_CO']]
    #multivariate_train = multivariate_data[multivariate_data['BASE_YEAR'].between(2019, 2022)].resample('W').sum()
    #multivariate_test = multivariate_data[multivariate_data['BASE_YEAR'] == 2023].resample('W').sum()

    # VAR 모델 적합
    var_model = VAR(multivariate_train)
    var_model_fit = var_model.fit()

    # 2023년 데이터에 대한 예측
    var_predictions = var_model_fit.forecast(multivariate_train.values[-var_model_fit.k_ar:], steps=len(multivariate_test))

    # 예측 결과를 DataFrame으로 변환
    var_predictions_df = pd.DataFrame(var_predictions, index=multivariate_test.index, columns=multivariate_test.columns)
    
    return var_model_fit, multivariate_train, multivariate_test, var_predictions_df



# 결과 시각화 - ARIMA
# plt.figure(figsize=(10, 6))
# plt.plot(ts_train.index, ts_train, label='Training Data')
# plt.plot(ts_test.index, ts_test, label='Actual Data', color='orange')
# plt.plot(ts_test.index, predictions, label='Forecast', color='green')
# plt.title('Movie Attendance Forecast')
# plt.xlabel('Date')
# plt.ylabel('Movie Attendance')
# plt.legend()
# plt.show()

# 결과 시각화 - VAR
# plt.figure(figsize=(12, 6))
# plt.plot(multivariate_train.index, multivariate_train['MOVIE_ADNC_CO'], label='Training Data - Total Attendance')
# plt.plot(multivariate_test.index, multivariate_test['MOVIE_ADNC_CO'], label='Actual Data - Total Attendance', color='orange')
# plt.plot(var_predictions_df.index, var_predictions_df['MOVIE_ADNC_CO'], label='Forecast - Total Attendance', color='green')
# plt.title('Multivariate Time Series Forecast with VAR')
# plt.xlabel('Date')
# plt.ylabel('Movie Attendance')
# plt.legend()
# plt.show()
