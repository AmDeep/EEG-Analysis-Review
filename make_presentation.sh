#!/bin/bash

# EEG Analysis Pipeline - PDF Presentation Generator
# Simple bash script to generate the PDF slide deck

set -e

echo "========================================================================"
echo "EEG Analysis Pipeline - PDF Presentation Generator"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if reportlab is installed
python3 -c "import reportlab" 2>/dev/null || {
    echo "Installing required dependencies..."
    pip install reportlab
}

echo "Generating PDF presentation..."
python3 make_presentation.py

echo ""
echo "========================================================================"
echo "PDF Presentation Generation Complete!"
echo "Output: docs/EEG_Pipeline_Slides.pdf"
echo "========================================================================"
