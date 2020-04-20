# PythonDataScienceExam
1. font setting
2. Initial set up konlpy
3. Kaggle API
    1) setting
        - download api(https://www.kaggle.com/(kaggleID)/account)
        - mkdir ~/.kaggle
        - cp ~/Downloads/kaggle.json ~/.kaggle/
        - chmod 600 ~/.kaggle/kaggle.json 
        - kaggle config set -n PATH -v ~/.kaggle/
        - kaggle config view
    2) command
        - kaggle competitions list -s (health)
        - kaggle competitions files -c (titanic)
        - kaggle competitions download -c titanic
4. 기타
    1) fbprophet
        - Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    2) Install : https://daewonyoon.tistory.com/266
                 https://hdongle.tistory.com/38
                 
    3) ubuntu-anaconda install : https://hiseon.me/python/ubuntu-anaconda-install/
