@echo off
REM Setup Script for NL2SQL Demo Project
echo Creating and activating virtual environment...

REM Activate venv
call venv\Scripts\activate.bat

REM Install packages
echo Installing required packages...
pip install --upgrade pip setuptools wheel
pip install torch==2.10.0 transformers==5.0.0 peft==0.18.1 accelerate==1.13.0 -q
pip install sentence-transformers==5.3.0 faiss-cpu==1.13.2 streamlit==1.45.1 -q
pip install sentencepiece==0.2.1 tokenizers==0.22.2 huggingface_hub==1.7.2 safetensors==0.7.0 -q
pip install tqdm==4.67.3 requests==2.32.5 scikit-learn==1.9.0 numpy==2.4.6 scipy==1.18.0 -q

echo.
echo ✓ Virtual environment setup complete!
echo ✓ All packages installed successfully!
echo.
echo To activate the environment in the future, run:
echo   .\venv\Scripts\activate
echo.
pause
