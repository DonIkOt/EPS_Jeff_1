# Инструкция по развертыванию сервиса с GradientBoosting

## Что было изменено

✅ **Модель обновлена**: `LinearRegression` → `GradientBoostingRegressor`
- Улучшено качество прогноза (меньший RMSE, выше R²)
- Модель автоматически обучается при первом запуске сервиса

✅ **Файлы обновлены**:
- `streamlit_app.py` — использует GradientBoosting
- `README.md` — обновлена документация
- `requirements.txt` — все зависимости уже включены (scikit-learn содержит GradientBoosting)

---

## Шаги для развертывания на Streamlit Cloud

### 1. Подготовка репозитория (если еще не сделано)

```bash
# В папке проекта
git init
git add .
git commit -m "Initial commit: GradientBoosting model for profit prediction"
```

### 2. Создание репозитория на GitHub

1. Зайдите на https://github.com
2. Нажмите **"New repository"**
3. Назовите репозиторий (например, `mining-profit-app`)
4. **НЕ** добавляйте README, .gitignore или лицензию (у вас уже есть файлы)
5. Нажмите **"Create repository"**

### 3. Подключение локального репозитория к GitHub

```bash
# Замените USER и REPO_NAME на ваши значения
git remote add origin https://github.com/USER/REPO_NAME.git
git branch -M main
git push -u origin main
```

Если Git запросит авторизацию:
- Используйте **Personal Access Token** (GitHub → Settings → Developer settings → Personal access tokens → Generate new token)
- В качестве пароля введите этот токен

### 4. Развертывание на Streamlit Cloud

1. Зайдите на https://share.streamlit.io/
2. Авторизуйтесь через GitHub (та же учетная запись, где находится репозиторий)
3. Нажмите **"New app"**
4. Заполните форму:
   - **Repository**: выберите ваш репозиторий (`USER/REPO_NAME`)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Нажмите **"Deploy"**

### 5. Ожидание развертывания

- Streamlit Cloud автоматически:
  - Установит зависимости из `requirements.txt`
  - Запустит `streamlit_app.py`
  - Обучит модель GradientBoosting (это займет ~30-60 секунд при первом запуске)
- После завершения вы получите **публичный URL** вида: `https://USER-REPO-NAME.streamlit.app`

### 6. Проверка работы

1. Откройте URL вашего приложения
2. Убедитесь, что:
   - Отображаются метрики модели (RMSE, R²)
   - Можно ввести параметры блока в sidebar
   - Прогноз прибыли рассчитывается корректно
   - Пакетное прогнозирование через CSV работает

---

## Обновление уже развернутого сервиса

Если сервис уже был развернут с LinearRegression и вы хотите обновить его до GradientBoosting:

```bash
# В папке проекта
git add streamlit_app.py README.md DEPLOYMENT.md
git commit -m "Update: switch to GradientBoosting model"
git push
```

Streamlit Cloud автоматически обнаружит изменения и перезапустит приложение (обычно в течение 1-2 минут).

---

## Локальное тестирование перед развертыванием

Перед загрузкой на GitHub рекомендуется протестировать локально:

```bash
# Активируйте виртуальное окружение (если еще не активировано)
.venv\Scripts\activate  # Windows
# или
source .venv/bin/activate  # Linux/Mac

# Установите зависимости
pip install -r requirements.txt

# Запустите приложение
streamlit run streamlit_app.py
```

Проверьте:
- ✅ Модель обучается без ошибок
- ✅ Метрики отображаются корректно
- ✅ Прогноз для одного блока работает
- ✅ Пакетное прогнозирование через CSV работает

---

## Возможные проблемы и решения

### Проблема: "ModuleNotFoundError: No module named 'sklearn'"
**Решение**: Убедитесь, что `scikit-learn>=1.5.0` указан в `requirements.txt` и установлен.

### Проблема: "FileNotFoundError: mining_block_model.csv"
**Решение**: Убедитесь, что файл `mining_block_model.csv` находится в корне репозитория и закоммичен в Git.

### Проблема: Модель обучается слишком долго на Streamlit Cloud
**Решение**: Это нормально для GradientBoosting при первом запуске (~30-60 сек). Последующие запросы будут быстрыми благодаря кэшированию (`@st.cache_resource`).

### Проблема: "Out of memory" на Streamlit Cloud
**Решение**: GradientBoosting использует больше памяти, чем LinearRegression. Если проблема сохраняется, можно:
- Уменьшить число деревьев в модели (параметр `n_estimators`)
- Или вернуться к Ridge/LinearRegression для более легкой модели

---

## Структура файлов для развертывания

Убедитесь, что в корне репозитория есть:
```
├── streamlit_app.py          # Основное приложение (обновлено для GradientBoosting)
├── requirements.txt          # Зависимости (уже содержит scikit-learn)
├── mining_block_model.csv    # Данные для обучения
├── README.md                 # Документация проекта
├── DEPLOYMENT.md             # Эта инструкция
└── Test_App_1.ipynb          # Ноутбук с EDA и сравнением моделей (опционально)
```

---

## Контакты и поддержка

Если возникли проблемы с развертыванием:
1. Проверьте логи в панели Streamlit Cloud
2. Убедитесь, что все файлы закоммичены и запушены в GitHub
3. Проверьте, что `requirements.txt` содержит все необходимые библиотеки
