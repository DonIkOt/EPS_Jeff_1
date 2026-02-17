import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "mining_block_model.csv"
TARGET_COL = "Profit (USD)"


@st.cache_resource
def load_data_and_train_model():
    """Загружаем данные, выполняем ту же подготовку, что и в ноутбуке, и обучаем модель."""
    df = pd.read_csv(DATA_PATH)

    # Удаляем экстремальные выбросы по прибыли (3 * IQR)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Столбец {TARGET_COL!r} не найден в данных.")

    s = df[TARGET_COL].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_extreme = q1 - 3 * iqr
    upper_extreme = q3 + 3 * iqr

    df_clean = df[(df[TARGET_COL] >= lower_extreme) & (df[TARGET_COL] <= upper_extreme)].reset_index(drop=True)

    # Исключаем служебные столбцы
    drop_cols = [c for c in ["Block_ID", "Target"] if c in df_clean.columns]
    feature_cols = [c for c in df_clean.columns if c not in drop_cols + [TARGET_COL]]

    numeric_features = [c for c in feature_cols if df_clean[c].dtype != "object"]
    categorical_features = [c for c in feature_cols if df_clean[c].dtype == "object"]

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

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
        "numeric_ranges": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
            }
            for c in numeric_features
        },
        "categorical_values": {
            c: sorted(X[c].dropna().unique().tolist()) for c in categorical_features
        },
    }

    return pipe, metrics, feature_info


def main():
    st.set_page_config(
        page_title="Прогноз прибыльности блока (Streamlit)",
        layout="centered",
    )

    st.title("Прогноз прибыльности добычи блока")
    st.markdown(
        """
        Этот сервис использует обученную модель **Gradient Boosting** по данным `mining_block_model.csv`
        для **прогноза прибыли `Profit (USD)`** по параметрам блока.
        """
    )

    with st.spinner("Загружаем данные и обучаем модель..."):
        model, metrics, feature_info = load_data_and_train_model()

    st.subheader("Качество модели (Gradient Boosting)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Train RMSE",
            f"{metrics['rmse_train']:,.2f}",
        )
        st.metric(
            "Train R²",
            f"{metrics['r2_train']:.4f}",
        )
    with col2:
        st.metric(
            "Test RMSE",
            f"{metrics['rmse_test']:,.2f}",
        )
        st.metric(
            "Test R²",
            f"{metrics['r2_test']:.4f}",
        )

    st.markdown("---")
    st.subheader("Интерактивный прогноз для одного блока")

    numeric_features = feature_info["numeric_features"]
    categorical_features = feature_info["categorical_features"]

    st.sidebar.header("Параметры блока")
    st.sidebar.markdown("Заполните входные параметры для прогноза прибыли.")

    input_data = {}

    # Ввод числовых признаков
    for col in numeric_features:
        r = feature_info["numeric_ranges"][col]
        # Безопасный диапазон с небольшим запасом
        span = r["max"] - r["min"]
        min_val = r["min"] - 0.05 * span
        max_val = r["max"] + 0.05 * span
        default_val = r["median"]

        input_data[col] = st.sidebar.number_input(
            col,
            value=float(default_val),
            min_value=float(min_val),
            max_value=float(max_val),
        )

    # Ввод категориальных признаков
    for col in categorical_features:
        values = feature_info["categorical_values"].get(col, [])
        if not values:
            continue
        default_index = 0
        input_data[col] = st.sidebar.selectbox(
            col,
            options=values,
            index=default_index,
        )

    if st.sidebar.button("Предсказать прибыль"):
        # Формируем DataFrame с одним наблюдением
        X_new = pd.DataFrame([input_data])
        y_pred = model.predict(X_new)[0]

        st.success("Прогноз успешно получен.")
        st.metric("Прогнозная прибыль блока, USD", f"{y_pred:,.2f}")

    st.markdown("---")
    st.subheader("Пакетное прогнозирование (CSV)")
    st.markdown(
        """
        Вы можете загрузить CSV-файл с параметрами блоков (те же столбцы, что и во входных признаках),
        и получить прогноз прибыли для каждого блока.
        """
    )

    uploaded = st.file_uploader("Загрузите CSV с блоками для прогнозирования", type=["csv"])

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
            return

        feature_cols = feature_info["feature_cols"]
        missing = [c for c in feature_cols if c not in df_new.columns]

        if missing:
            st.error(
                "В загруженном файле отсутствуют необходимые столбцы: "
                + ", ".join(missing)
            )
        else:
            preds = model.predict(df_new[feature_cols])
            df_result = df_new.copy()
            df_result["Predicted Profit (USD)"] = preds

            st.success("Прогноз для загруженного файла рассчитан.")
            st.dataframe(df_result.head(50))

            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать результаты в CSV",
                data=csv_bytes,
                file_name="predicted_profit.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

