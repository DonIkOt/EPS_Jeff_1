# Streamlit-сервис прогноза прибыльности блока

Этот проект содержит ноутбук с EDA (`Test_App_1.ipynb`), исходные данные (`mining_block_model.csv`) и веб-сервис на Streamlit (`streamlit_app.py`) для прогноза прибыли `Profit (USD)` по параметрам блока.

**Используемая модель**: Gradient Boosting Regressor (выбрана как лучшая по метрикам RMSE и R² после сравнения с LinearRegression, Ridge и RandomForest).

## Локальный запуск

1. Установите Python 3.10+.
2. Установите зависимости:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Запустите Streamlit-приложение:

```bash
streamlit run streamlit_app.py
```

Приложение откроется в браузере (обычно `http://localhost:8501`).

## Развёртывание на Streamlit Community Cloud

1. Создайте репозиторий на GitHub и загрузите в корень проекта:
   - `streamlit_app.py`
   - `requirements.txt`
   - `mining_block_model.csv` (или настройте чтение данных из другого источника/облака).
2. Зайдите на [Streamlit Community Cloud](https://share.streamlit.io/), авторизуйтесь через GitHub.
3. Нажмите **New app**, выберите репозиторий, ветку и файл `streamlit_app.py`.
4. Нажмите **Deploy** — через несколько минут сервис будет доступен по публичному URL.

## Развёртывание на собственном сервере

1. Скопируйте весь проект на сервер (через `git clone` или SCP).
2. На сервере:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

3. Откройте `http://<IP_сервера>:8501` в браузере или повесьте обратный прокси (nginx) на этот порт.

## Обновление модели на Streamlit Cloud

Если вы уже развернули сервис на Streamlit Cloud и хотите обновить его до версии с GradientBoosting:

1. Убедитесь, что все изменения закоммичены и запушены в GitHub:
   ```bash
   git add streamlit_app.py README.md
   git commit -m "Update: switch to GradientBoosting model"
   git push
   ```

2. Streamlit Cloud автоматически обнаружит изменения и перезапустит приложение (обычно в течение 1-2 минут).

3. Проверьте логи развертывания в панели Streamlit Cloud, чтобы убедиться, что модель обучилась успешно.

**Примечание**: GradientBoosting требует больше времени на обучение, чем LinearRegression (обычно 30-60 секунд при первом запуске), но это происходит только один раз благодаря кэшированию (`@st.cache_resource`).

