# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Start bot
echo "Starting bot..."
nohup python bot_new.py > /dev/null 2>&1 &