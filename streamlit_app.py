"""
Streamlit приложение для прогноза прибыльности добычи блока
Использует модель Gradient Boosting Regressor
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Константы
DATA_PATH = "mining_block_model.csv"
TARGET_COL = "Profit (USD)"


@st.cache_resource
def load_data_and_train_model():
    """
    Загружает данные, выполняет подготовку и обучает модель GradientBoosting.
    Результат кэшируется для ускорения последующих запусков.
    """
    try:
        # Загрузка данных
        df = pd.read_csv(DATA_PATH)
        
        if df.empty:
            raise ValueError("Загруженный файл пуст")
        
        # Проверка наличия целевой переменной
        if TARGET_COL not in df.columns:
            raise ValueError(f"Столбец {TARGET_COL!r} не найден в данных. Доступные столбцы: {list(df.columns)}")
        
        # Удаление экстремальных выбросов по прибыли (3 * IQR)
        profit_values = df[TARGET_COL].dropna()
        
        if len(profit_values) == 0:
            raise ValueError(f"Нет валидных значений в столбце {TARGET_COL}")
        
        q1 = profit_values.quantile(0.25)
        q3 = profit_values.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            st.warning("IQR равен нулю, пропускаем фильтрацию выбросов")
            df_clean = df.copy()
        else:
            lower_extreme = q1 - 3 * iqr
            upper_extreme = q3 + 3 * iqr
            df_clean = df[
                (df[TARGET_COL] >= lower_extreme) & 
                (df[TARGET_COL] <= upper_extreme)
            ].reset_index(drop=True)
        
        if df_clean.empty:
            raise ValueError("После фильтрации выбросов датасет стал пустым")
        
        # Исключение служебных столбцов
        drop_cols = []
        for col in ["Block_ID", "Target"]:
            if col in df_clean.columns:
                drop_cols.append(col)
        
        feature_cols = [c for c in df_clean.columns if c not in drop_cols + [TARGET_COL]]
        
        if not feature_cols:
            raise ValueError("Не найдено признаков для обучения модели")
        
        # Разделение на числовые и категориальные признаки
        numeric_features = [c for c in feature_cols if df_clean[c].dtype in ['int64', 'float64']]
        categorical_features = [c for c in feature_cols if df_clean[c].dtype == 'object']
        
        # Формирование матриц признаков и целевой переменной
        X = df_clean[feature_cols].copy()
        y = df_clean[TARGET_COL].copy()
        
        # Проверка на пустые значения
        if X.isnull().all().any():
            st.warning("Обнаружены столбцы полностью состоящие из NaN. Они будут удалены.")
            X = X.dropna(axis=1, how='all')
            numeric_features = [c for c in numeric_features if c in X.columns]
            categorical_features = [c for c in categorical_features if c in X.columns]
            feature_cols = numeric_features + categorical_features
        
        # Настройка препроцессора
        transformers = []
        
        if numeric_features:
            transformers.append(("num", StandardScaler(), numeric_features))
        
        if categorical_features:
            # Используем sparse_output=False для совместимости с новыми версиями sklearn
            # Для старых версий можно убрать этот параметр
            try:
                ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            except TypeError:
                # Для старых версий sklearn используем sparse=False
                ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
            transformers.append(("cat", ohe, categorical_features))
        
        if not transformers:
            raise ValueError("Нет признаков для обучения (ни числовых, ни категориальных)")
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        
        # Создание модели GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        
        # Создание пайплайна
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Обучение модели
        with st.spinner("Обучаем модель GradientBoosting (это может занять 30-60 секунд)..."):
            pipe.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        # Метрики
        metrics = {
            "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "r2_train": float(r2_score(y_train, y_train_pred)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "r2_test": float(r2_score(y_test, y_test_pred)),
        }
        
        # Метаданные по признакам для UI
        feature_info = {
            "feature_cols": feature_cols,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_ranges": {},
            "categorical_values": {},
        }
        
        # Заполнение диапазонов для числовых признаков
        for col in numeric_features:
            if col in X.columns:
                col_values = X[col].dropna()
                if len(col_values) > 0:
                    feature_info["numeric_ranges"][col] = {
                        "min": float(col_values.min()),
                        "max": float(col_values.max()),
                        "median": float(col_values.median()),
                    }
        
        # Заполнение значений для категориальных признаков
        for col in categorical_features:
            if col in X.columns:
                unique_vals = X[col].dropna().unique().tolist()
                if unique_vals:
                    feature_info["categorical_values"][col] = sorted(unique_vals)
        
        return pipe, metrics, feature_info
        
    except FileNotFoundError:
        st.error(f"Файл {DATA_PATH} не найден. Убедитесь, что файл находится в той же папке, что и streamlit_app.py")
        raise
    except Exception as e:
        st.error(f"Ошибка при загрузке данных или обучении модели: {str(e)}")
        raise


def main():
    """Основная функция приложения Streamlit"""
    
    st.set_page_config(
        page_title="Прогноз прибыльности блока",
        page_icon="⛏️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⛏️ Прогноз прибыльности добычи блока")
    st.markdown("""
    Этот сервис использует модель **Gradient Boosting Regressor** для прогноза прибыли 
    `Profit (USD)` по геолого-экономическим параметрам блока.
    
    Модель обучена на данных `mining_block_model.csv` с предварительной обработкой:
    - Удаление экстремальных выбросов (3 * IQR)
    - Масштабирование числовых признаков
    - One-hot кодирование категориальных признаков
    """)
    
    # Загрузка и обучение модели
    try:
        model_pipeline, metrics, feature_info = load_data_and_train_model()
    except Exception as e:
        st.error("Не удалось загрузить модель. Проверьте логи выше.")
        st.stop()
    
    # Отображение метрик качества модели
    st.subheader("📊 Качество модели")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train RMSE", f"{metrics['rmse_train']:,.2f}")
    with col2:
        st.metric("Train R²", f"{metrics['r2_train']:.4f}")
    with col3:
        st.metric("Test RMSE", f"{metrics['rmse_test']:,.2f}")
    with col4:
        st.metric("Test R²", f"{metrics['r2_test']:.4f}")
    
    st.markdown("---")
    
    # Интерактивный прогноз для одного блока
    st.subheader("🔮 Интерактивный прогноз для одного блока")
    
    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]
    
    st.sidebar.header("📝 Параметры блока")
    st.sidebar.markdown("Заполните параметры блока для прогноза прибыли.")
    
    input_data = {}
    
    # Ввод числовых признаков
    if numeric_features:
        st.sidebar.subheader("Числовые параметры")
        for col in numeric_features:
            if col in feature_info["numeric_ranges"]:
                r = feature_info["numeric_ranges"][col]
                span = r["max"] - r["min"]
                if span > 0:
                    min_val = r["min"] - 0.05 * span
                    max_val = r["max"] + 0.05 * span
                else:
                    min_val = r["min"] - 1
                    max_val = r["max"] + 1
                
                default_val = r["median"]
                
                input_data[col] = st.sidebar.number_input(
                    col,
                    value=float(default_val),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    step=0.01 if r["max"] - r["min"] < 100 else 1.0,
                    help=f"Диапазон: [{r['min']:.2f}, {r['max']:.2f}]"
                )
    
    # Ввод категориальных признаков
    if categorical_features:
        st.sidebar.subheader("Категориальные параметры")
        for col in categorical_features:
            values = feature_info["categorical_values"].get(col, [])
            if values:
                default_index = 0
                input_data[col] = st.sidebar.selectbox(
                    col,
                    options=values,
                    index=default_index,
                    help=f"Доступные значения: {', '.join(values)}"
                )
    
    # Кнопка предсказания
    predict_button = st.sidebar.button("🚀 Предсказать прибыль", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Формируем DataFrame с одним наблюдением
            X_new = pd.DataFrame([input_data])
            
            # Предсказание
            y_pred = model_pipeline.predict(X_new)[0]
            
            # Отображение результата
            st.success("✅ Прогноз успешно рассчитан!")
            
            col_pred1, col_pred2 = st.columns([1, 1])
            with col_pred1:
                st.metric(
                    "Прогнозная прибыль блока",
                    f"${y_pred:,.2f}",
                    delta=None
                )
            with col_pred2:
                if y_pred > 0:
                    st.success("💰 Блок прибыльный")
                else:
                    st.warning("⚠️ Блок убыточный")
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")
    
    st.markdown("---")
    
    # Пакетное прогнозирование
    st.subheader("📁 Пакетное прогнозирование (CSV)")
    st.markdown("""
    Загрузите CSV-файл с параметрами блоков (те же столбцы, что и во входных признаках).
    Сервис рассчитает прогноз прибыли для каждого блока.
    """)
    
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл",
        type=["csv"],
        help="Файл должен содержать все необходимые столбцы признаков"
    )
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            
            if df_new.empty:
                st.error("Загруженный файл пуст")
            else:
                feature_cols = feature_info["feature_cols"]
                missing_cols = [c for c in feature_cols if c not in df_new.columns]
                
                if missing_cols:
                    st.error(f"В загруженном файле отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
                    st.info(f"Требуемые столбцы: {', '.join(feature_cols)}")
                else:
                    # Предсказания для всех строк
                    with st.spinner("Рассчитываем прогнозы..."):
                        preds = model_pipeline.predict(df_new[feature_cols])
                    
                    # Формирование результата
                    df_result = df_new.copy()
                    df_result["Predicted Profit (USD)"] = preds
                    
                    st.success(f"✅ Прогноз рассчитан для {len(df_result)} блоков")
                    
                    # Статистика по прогнозам
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Средняя прибыль", f"${df_result['Predicted Profit (USD)'].mean():,.2f}")
                    with col_stat2:
                        st.metric("Максимальная прибыль", f"${df_result['Predicted Profit (USD)'].max():,.2f}")
                    with col_stat3:
                        profitable = (df_result['Predicted Profit (USD)'] > 0).sum()
                        st.metric("Прибыльных блоков", f"{profitable} / {len(df_result)}")
                    
                    # Таблица результатов
                    st.dataframe(
                        df_result.head(100),
                        use_container_width=True,
                        height=400
                    )
                    
                    if len(df_result) > 100:
                        st.info(f"Показаны первые 100 строк из {len(df_result)}")
                    
                    # Кнопка скачивания
                    csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Скачать результаты в CSV",
                        data=csv_bytes,
                        file_name="predicted_profit.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except pd.errors.EmptyDataError:
            st.error("Загруженный CSV файл пуст или поврежден")
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # Информация о модели в футере
    st.markdown("---")
    with st.expander("ℹ️ Информация о модели"):
        st.markdown(f"""
        **Модель**: Gradient Boosting Regressor
        
        **Параметры модели**:
        - Количество деревьев: 100
        - Learning rate: 0.1
        - Максимальная глубина: 5
        
        **Количество признаков**:
        - Числовых: {len(feature_info['numeric_features'])}
        - Категориальных: {len(feature_info['categorical_features'])}
        - Всего: {len(feature_info['feature_cols'])}
        
        **Размер обучающей выборки**: ~{len(feature_info['feature_cols'])} признаков использовано для обучения
        """)


if __name__ == "__main__":
    main()
