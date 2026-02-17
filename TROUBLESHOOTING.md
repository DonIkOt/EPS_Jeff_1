# Решение проблем с запуском Streamlit

## Быстрая диагностика

Сначала запустите тестовый скрипт для проверки окружения:

```bash
python test_streamlit.py
```

Этот скрипт проверит:
- ✅ Все ли библиотеки установлены
- ✅ Находится ли файл `mining_block_model.csv`
- ✅ Корректен ли синтаксис `streamlit_app.py`

## Частые проблемы и решения

### 1. Ошибка: "streamlit: command not found"

**Проблема**: Streamlit не установлен или не в PATH

**Решение**:
```bash
# Установите streamlit
pip install streamlit

# Или через python -m
python -m pip install streamlit

# Проверьте установку
python -m streamlit --version
```

### 2. Ошибка: "ModuleNotFoundError: No module named 'sklearn'"

**Проблема**: scikit-learn не установлен

**Решение**:
```bash
pip install scikit-learn>=1.5.0
```

### 3. Ошибка: "FileNotFoundError: mining_block_model.csv"

**Проблема**: Файл данных не найден

**Решение**:
1. Убедитесь, что файл `mining_block_model.csv` находится в той же папке, что и `streamlit_app.py`
2. Проверьте текущую директорию:
   ```bash
   # Windows PowerShell
   Get-Location
   
   # Windows CMD
   cd
   
   # Linux/Mac
   pwd
   ```
3. Перейдите в правильную директорию:
   ```bash
   cd "c:\Ф\StudProjects\Cursor Proj\Test 2"
   ```

### 4. Ошибка: "TypeError: __init__() got an unexpected keyword argument 'sparse_output'"

**Проблема**: Старая версия scikit-learn (< 1.2)

**Решение**:
```bash
pip install --upgrade scikit-learn>=1.5.0
```

### 5. Ошибка при запуске: "Address already in use"

**Проблема**: Порт 8501 уже занят

**Решение**:
```bash
# Запустите на другом порту
streamlit run streamlit_app.py --server.port 8502
```

### 6. Ошибка: "UnicodeDecodeError" при чтении CSV

**Проблема**: Неправильная кодировка файла

**Решение**: Убедитесь, что файл `mining_block_model.csv` сохранен в UTF-8

### 7. Приложение запускается, но модель не обучается

**Проблема**: Ошибка в данных или недостаточно памяти

**Решение**:
1. Проверьте логи в консоли
2. Убедитесь, что в CSV файле есть все необходимые столбцы
3. Попробуйте уменьшить параметры модели в `streamlit_app.py`:
   ```python
   model = GradientBoostingRegressor(
       n_estimators=50,  # уменьшить с 100
       max_depth=3,      # уменьшить с 5
       ...
   )
   ```

## Правильный порядок запуска

1. **Откройте терминал в папке проекта**
   ```bash
   cd "c:\Ф\StudProjects\Cursor Proj\Test 2"
   ```

2. **Активируйте виртуальное окружение** (если используете)
   ```bash
   .venv\Scripts\activate  # Windows
   # или
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Установите зависимости** (если еще не установлены)
   ```bash
   pip install -r requirements.txt
   ```

4. **Запустите тестовый скрипт** (опционально, для диагностики)
   ```bash
   python test_streamlit.py
   ```

5. **Запустите Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Откройте браузер**
   - Streamlit автоматически откроет браузер
   - Или перейдите на `http://localhost:8501`

## Проверка версий

Убедитесь, что у вас установлены правильные версии:

```bash
python --version          # Должно быть 3.10+
pip list | findstr streamlit
pip list | findstr scikit-learn
pip list | findstr pandas
pip list | findstr numpy
```

## Альтернативный способ запуска

Если `streamlit` команда не работает, используйте:

```bash
python -m streamlit run streamlit_app.py
```

## Полная переустановка зависимостей

Если ничего не помогает:

```bash
# Удалите виртуальное окружение (если используете)
# Windows
rmdir /s .venv
# Linux/Mac
rm -rf .venv

# Создайте новое
python -m venv .venv

# Активируйте
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Установите зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Запустите
streamlit run streamlit_app.py
```

## Получение подробных логов

Для отладки запустите с подробными логами:

```bash
streamlit run streamlit_app.py --logger.level=debug
```

## Если проблема не решена

1. Запустите `python test_streamlit.py` и скопируйте весь вывод
2. Запустите `streamlit run streamlit_app.py` и скопируйте ошибку
3. Проверьте версии всех библиотек: `pip list`
4. Опишите проблему с этими данными
