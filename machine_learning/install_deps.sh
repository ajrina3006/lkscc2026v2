#!/bin/bash
# Install script untuk dependencies machine learning

echo "🔧 Installing Python dependencies..."
pip install -q -r machine_learning/requirements.txt

echo "✓ Installation complete!"
echo "Packages installed:"
pip list | grep -E 'boto3|pandas|numpy|scikit-learn|pyarrow'
