@echo off
echo Running Ingestion...
.venv\Scripts\python.exe src\ingestion.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running Chunking...
.venv\Scripts\python.exe src\chunking.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running Embedding and Indexing...
.venv\Scripts\python.exe src\embed_and_index.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running Retrieval...
.venv\Scripts\python.exe src\retrieve.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running Generation...
.venv\Scripts\python.exe src\generate.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo All steps completed successfully!
