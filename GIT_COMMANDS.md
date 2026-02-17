# Команды для отправки в GitHub и развертывания

## Шаг 1: Добавление файлов в Git

Выполните в терминале (Git Bash или PowerShell):

```bash
# Перейдите в папку проекта
cd "c:\Ф\StudProjects\Cursor Proj\Test 2"

# Добавьте все измененные и новые файлы
git add streamlit_app.py
git add requirements.txt
git add README.md
git add DEPLOYMENT.md
git add TROUBLESHOOTING.md
git add test_streamlit.py
git add Test_App_1.ipynb
git add .gitignore

# Или добавьте все файлы одной командой (кроме игнорируемых)
git add .
```

## Шаг 2: Создание коммита

```bash
git commit -m "Update: GradientBoosting model with improved error handling and diagnostics"
```

## Шаг 3: Отправка в GitHub

```bash
# Отправьте изменения в GitHub
git push origin main
```

Если Git запросит авторизацию:
- **Username**: ваш GitHub username
- **Password**: используйте **Personal Access Token** (не пароль от GitHub)

### Как создать Personal Access Token:

1. Зайдите на GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Нажмите "Generate new token (classic)"
3. Назовите токен (например, "Streamlit Deployment")
4. Выберите срок действия (например, 90 дней или "No expiration")
5. Отметьте права: **repo** (полный доступ к репозиториям)
6. Нажмите "Generate token"
7. **Скопируйте токен** (он показывается только один раз!)
8. Используйте этот токен как пароль при `git push`

## Шаг 4: Развертывание на Streamlit Cloud

### Если приложение еще не развернуто:

1. Зайдите на https://share.streamlit.io/
2. Авторизуйтесь через GitHub
3. Нажмите **"New app"**
4. Заполните форму:
   - **Repository**: выберите ваш репозиторий
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Нажмите **"Deploy"**
6. Дождитесь завершения развертывания (~1-2 минуты)
7. Получите публичный URL вашего приложения

### Если приложение уже развернуто:

После выполнения `git push`, Streamlit Cloud автоматически обнаружит изменения и перезапустит приложение в течение 1-2 минут.

Проверьте статус в панели Streamlit Cloud:
- Зеленый индикатор = приложение работает
- Красный индикатор = есть ошибки (проверьте логи)

## Проверка статуса

После выполнения команд проверьте:

```bash
# Проверьте статус репозитория
git status

# Проверьте историю коммитов
git log --oneline -5

# Проверьте подключение к удаленному репозиторию
git remote -v
```

## Быстрая команда (все в одном)

Если хотите выполнить все сразу:

```bash
cd "c:\Ф\StudProjects\Cursor Proj\Test 2"
git add .
git commit -m "Update: GradientBoosting model with improved error handling"
git push origin main
```
