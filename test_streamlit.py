"""
Тестовый скрипт для проверки работы streamlit_app.py без запуска Streamlit
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Проверяем импорты
    print("Проверка импортов...")
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    print("✅ Все импорты успешны")
    
    # Проверяем наличие файла данных
    print("\nПроверка файла данных...")
    if os.path.exists("mining_block_model.csv"):
        print("✅ Файл mining_block_model.csv найден")
        df = pd.read_csv("mining_block_model.csv")
        print(f"✅ Файл прочитан: {len(df)} строк, {len(df.columns)} столбцов")
        print(f"   Столбцы: {list(df.columns)}")
    else:
        print("❌ Файл mining_block_model.csv не найден!")
        print(f"   Текущая директория: {os.getcwd()}")
        print(f"   Файлы в директории: {os.listdir('.')}")
        sys.exit(1)
    
    # Проверяем наличие целевой переменной
    if "Profit (USD)" not in df.columns:
        print("❌ Столбец 'Profit (USD)' не найден!")
        sys.exit(1)
    else:
        print("✅ Целевая переменная 'Profit (USD)' найдена")
    
    # Проверяем версию sklearn
    import sklearn
    print(f"\nВерсия scikit-learn: {sklearn.__version__}")
    
    # Проверяем синтаксис streamlit_app.py
    print("\nПроверка синтаксиса streamlit_app.py...")
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, "streamlit_app.py", "exec")
    print("✅ Синтаксис streamlit_app.py корректен")
    
    print("\n✅ Все проверки пройдены успешно!")
    print("\nДля запуска Streamlit выполните:")
    print("   streamlit run streamlit_app.py")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("\nУстановите зависимости:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
