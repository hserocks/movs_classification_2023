# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Start bot
echo "Starting bot..."
nohup python scripts/bot_new.py > /dev/null 2>&1 &