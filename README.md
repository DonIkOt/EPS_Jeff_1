# Streamlit-сервис прогноза прибыльности блока

Этот проект содержит ноутбук с EDA (`Test_App_1.ipynb`), исходные данные (`mining_block_model.csv`) и веб-сервис на Streamlit (`streamlit_app.py`) для прогноза прибыли `Profit (USD)` по параметрам блока.

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

