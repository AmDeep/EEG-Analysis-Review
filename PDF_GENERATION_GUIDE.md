# PDF Presentation Generator - Usage Guide

## Overview
This guide explains how to generate the EEG Analysis Pipeline PDF slide deck using the provided Python script.

## Prerequisites
- Python 3.7+
- reportlab library

## Installation

### 1. Install Required Dependencies
```bash
pip install reportlab
```

Or use the requirements file:
```bash
pip install -r requirements_pdf.txt
```

## Usage

### Generate the PDF Presentation
Run the presentation generator script:
```bash
python make_presentation.py
```

### Output
The script will:
- Create a `docs/` directory if it doesn't exist
- Generate `docs/EEG_Pipeline_Slides.pdf` with 13 professional slides
- Print success message with file location and slide count

### Expected Output
```
======================================================================
EEG Analysis Pipeline PDF Slide Deck Generator
======================================================================
✓ PDF generated: docs/EEG_Pipeline_Slides.pdf
✓ Total slides: 13
======================================================================
SUCCESS: Presentation created at docs/EEG_Pipeline_Slides.pdf
======================================================================
```

## Slide Contents

The generated PDF includes 13 slides covering:

1. **Title Slide** - Pipeline overview and repository info
2. **Pipeline Goals** - Objectives and validation approach
3. **Dataset Overview** - OpenNeuro ds005284, 26 subjects, 781 epochs
4. **Preprocessing Pipeline** - Channel mapping, resampling, filtering, epoching, artifact rejection
5. **Feature Extraction** - Seven feature families (spectral, nonlinear, time-domain, wavelet, MFCC, AR, connectivity)
6. **Feature Storage** - Naming conventions and output formats
7. **LOSO Validation** - Leave-one-subject-out cross-validation procedure
8. **Classifier Selection** - LDA, Random Forest, Gradient Boosting recommendations
9. **Ablation Studies** - Incremental feature family testing strategy
10. **Output Files** - Results tables, visualizations, audit files
11. **New Datasets** - Step-by-step checklist for applying pipeline
12. **Reproducibility** - Best practices and next steps
13. **Technical Summary** - Data flow, validation architecture, interpretability

## Customization

To modify slide content, edit `make_presentation.py`:

### Change Output Path
```python
generator = SlideGenerator("custom/path/output.pdf")
```

### Modify Slide Content
```python
generator.add_content_slide(
    "New Title",
    bullets=[
        "Bullet point 1",
        "Bullet point 2",
    ]
)
```

### Change Color Scheme
Edit the color constants at the top of the file:
```python
INDIGO = colors.HexColor("#3F51B5")
TEAL = colors.HexColor("#009688")
CORAL = colors.HexColor("#FF6F61")
```

## Troubleshooting

### ImportError: No module named 'reportlab'
Solution: Install reportlab
```bash
pip install reportlab
```

### Permission Denied Error
Solution: Make script executable on Linux/Mac
```bash
chmod +x make_presentation.py
python make_presentation.py
```

### PDF File Not Created
- Check that the `docs/` directory exists or is writable
- Ensure sufficient disk space
- Review error messages in console output

## Files

- `make_presentation.py` - Main PDF generator script
- `generate_pdf_slides.py` - Alternative generator (legacy)
- `requirements_pdf.txt` - Python dependencies
- `docs/EEG_Pipeline_Slides.pdf` - Generated output (created after running script)

## Repository
**AmDeep/EEG-Analysis-Review**
https://github.com/AmDeep/EEG-Analysis-Review

## Support
For issues or questions, refer to the main repository documentation or open an issue in GitHub.
