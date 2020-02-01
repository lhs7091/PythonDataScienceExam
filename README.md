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
