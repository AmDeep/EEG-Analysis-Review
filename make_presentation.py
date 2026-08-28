#!/usr/bin/env python3
"""
Execute PDF slide deck generation for EEG Analysis Pipeline.
This script runs the generator and produces EEG_Pipeline_Slides.pdf in the docs folder.
"""

import os
import sys
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

# Color palette
INDIGO = colors.HexColor("#3F51B5")
TEAL = colors.HexColor("#009688")
CORAL = colors.HexColor("#FF6F61")
LIGHT_GRAY = colors.HexColor("#F5F5F5")
DARK_GRAY = colors.HexColor("#333333")
WHITE = colors.HexColor("#FFFFFF")

class SlideGenerator:
    def __init__(self, output_path="docs/EEG_Pipeline_Slides.pdf"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.page_width, self.page_height = landscape(letter)
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(letter),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        self.elements = []
        self.slide_count = 0

    def _create_custom_styles(self):
        """Create custom paragraph styles for slides."""
        self.styles.add(ParagraphStyle(
            name='SlideTitle',
            parent=self.styles['Heading1'],
            fontSize=48,
            textColor=WHITE,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='SlideTitleDark',
            parent=self.styles['Heading1'],
            fontSize=40,
            textColor=INDIGO,
            spaceAfter=20,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='BulletText',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=DARK_GRAY,
            spaceAfter=10,
            leftIndent=20,
            alignment=TA_JUSTIFY,
        ))
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=DARK_GRAY,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
        ))

    def add_title_slide(self, title, subtitle, date_str=None):
        """Add a title slide with large formatted text."""
        title_style = self.styles['SlideTitle']
        subtitle_style = ParagraphStyle(
            name='SubtitleText',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=WHITE,
            spaceAfter=20,
            alignment=TA_CENTER,
        )
        
        title_para = Paragraph(title, title_style)
        subtitle_para = Paragraph(subtitle, subtitle_style)
        
        # Create colored background table
        title_table = Table(
            [[title_para], [Spacer(1, 0.5*inch)], [subtitle_para]],
            colWidths=[self.page_width - 1*inch],
        )
        title_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), INDIGO),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('TOPPADDING', (0, 0), (-1, -1), 40),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 40),
        ]))
        
        self.elements.append(title_table)
        if date_str:
            date_para = Paragraph(f"<i>{date_str}</i>", self.styles['BodyText'])
            self.elements.append(Spacer(1, 0.3*inch))
            self.elements.append(date_para)
        
        self.elements.append(PageBreak())
        self.slide_count += 1

    def add_content_slide(self, title, bullets=None, content_text=None):
        """Add a content slide with title and bullet points."""
        # Title with background
        title_para = Paragraph(title, self.styles['SlideTitleDark'])
        
        # Add horizontal line under title
        line_table = Table(
            [[title_para]],
            colWidths=[self.page_width - 1*inch],
        )
        line_table.setStyle(TableStyle([
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LINEBELOW', (0, 0), (-1, -1), 3, TEAL),
        ]))
        
        self.elements.append(line_table)
        self.elements.append(Spacer(1, 0.2*inch))
        
        # Add bullet points
        if bullets:
            for bullet in bullets:
                if isinstance(bullet, str):
                    bullet_para = Paragraph(f"• {bullet}", self.styles['BulletText'])
                    self.elements.append(bullet_para)
                    self.elements.append(Spacer(1, 0.1*inch))
        
        # Add content text
        if content_text:
            self.elements.append(Spacer(1, 0.15*inch))
            for text in content_text if isinstance(content_text, list) else [content_text]:
                content_para = Paragraph(text, self.styles['BodyText'])
                self.elements.append(content_para)
                self.elements.append(Spacer(1, 0.08*inch))
        
        self.elements.append(PageBreak())
        self.slide_count += 1

    def build(self):
        """Build and save the PDF."""
        self.doc.build(self.elements)
        print(f"✓ PDF generated: {self.output_path}")
        print(f"✓ Total slides: {self.slide_count}")
        return self.output_path


def generate_pdf():
    """Main function to generate the PDF presentation."""
    generator = SlideGenerator("docs/EEG_Pipeline_Slides.pdf")
    
    # Title Slide
    generator.add_title_slide(
        "EEG Pain Biomarker Pipeline",
        "Reproducible Preprocessing, Feature Engineering &amp; Validation",
        f"Repository: AmDeep/EEG-Analysis-Review | Generated: {datetime.now().strftime('%B %d, %Y')}"
    )
    
    # Slide 1: Goals
    generator.add_content_slide(
        "Pipeline Goals",
        bullets=[
            "Identify robust EEG biomarkers that discriminate pain vs no-pain states",
            "Produce leakage-safe held-out validation (LOSO cross-validation)",
            "Provide deployable, interpretable model with full reproducibility audit trail",
        ]
    )
    
    # Slide 2: Dataset Overview
    generator.add_content_slide(
        "Dataset &amp; Key Facts",
        bullets=[
            "<b>Source:</b> OpenNeuro ds005284 (Zhao et al.)",
            "<b>Acquisition:</b> Biosemi 64-channel, 1024 Hz sampled → resampled to 256 Hz",
            "<b>Sample size:</b> 26 subjects, 781 epochs (367 pain / 414 no-pain)",
            "<b>ROI:</b> C3, CZ, C4 (primary somatosensory cortex focus)",
        ]
    )
    
    # Slide 3: Preprocessing Pipeline
    generator.add_content_slide(
        "Preprocessing Pipeline (Step-by-Step)",
        bullets=[
            "Channel mapping: raw device labels → 10–20 standard (channels.tsv)",
            "Resample: 1024 Hz → 256 Hz using scipy.signal.resample",
            "Notch filter: 50 Hz &amp; 100 Hz (adjust to 60 Hz for US mains)",
            "Bandpass filter: 0.5–80 Hz (Butterworth IIR)",
            "Reference: average reference across all channels",
            "Epoching: Pain (0→+2s), No-pain baseline (−5→−3s) relative to stimulus onset",
            "Artifact rejection: amplitude threshold (|µV| &gt; 200)",
            "QC logging: per-epoch QC flags &amp; per-subject summary CSVs",
        ]
    )
    
    # Slide 4: Feature Families
    generator.add_content_slide(
        "Feature Extraction: Seven Families",
        bullets=[
            "<b>Spectral:</b> absolute/relative band power (δ,θ,α,β,γ), band ratios, spectral centroid, 1/f slope",
            "<b>Nonlinear:</b> Higuchi &amp; Petrosian FD, DFA, entropy (sample &amp; permutation), Lempel-Ziv, Hjorth",
            "<b>Time-domain:</b> RMS, std, peak-to-peak, line length, zero-crossing rate, skew, kurtosis",
            "<b>Wavelet:</b> db4 multiscale energy ratios, wavelet entropy across scales",
            "<b>MFCC:</b> first 5 Mel-frequency cepstral coefficients per channel",
            "<b>AR dynamics:</b> Burg AR coefficients (order 10) + innovation variance",
            "<b>Connectivity:</b> coherence &amp; phase-locking value (PLV) for C3-CZ, C3-C4, CZ-C4 across bands",
        ]
    )
    
    # Slide 5: Feature Storage &amp; Naming
    generator.add_content_slide(
        "Feature Storage &amp; Naming Conventions",
        bullets=[
            "<b>Naming scheme:</b> descriptive labels (e.g., CZ_higuchi, C3_alpha_power, CZ_C4_lowgamma_plv)",
            "<b>Primary outputs:</b>",
            "      – features.csv (rows = epochs, columns = 135+ features)",
            "      – features.npz (compressed NumPy: X, y, subject_ids, feature_names)",
            "      – per_epoch_qc.csv (artifact flags, amplitude stats)",
            "      – subject_summary.csv (per-subject acceptance rates, rejection counts)",
        ]
    )
    
    # Slide 6: Validation Strategy (LOSO)
    generator.add_content_slide(
        "Leave-One-Subject-Out (LOSO) Validation",
        bullets=[
            "<b>1. Leakage-safe procedure:</b> For each training fold, fit ALL preprocessing on train set only",
            "<b>2. Per-fold training pipeline:</b>",
            "      – Replace non-finite values (NaN, inf)",
            "      – Apply variance threshold (train-only) to remove near-zero variance features",
            "      – Fit StandardScaler on train → transform both train &amp; test",
            "      – Compute ANOVA-F ranking on train → select top-K features (e.g., K=60)",
            "      – Fit classifier (e.g., LDA with shrinkage estimation)",
            "<b>3. Evaluation:</b> Predict on held-out subject; collect per-fold AUC, sensitivity, specificity",
            "<b>4. Reporting:</b> Mean AUC ±SD, per-subject metrics, paired Wilcoxon tests for ablations",
        ]
    )
    
    # Slide 7: Recommended Classifiers
    generator.add_content_slide(
        "Classifier Selection &amp; Deployment",
        bullets=[
            "<b>Research / Exploration:</b> Random Forest, Gradient Boosting — excellent for feature importance &amp; overfitting detection",
            "<b>Deployment / Interpretability:</b> LDA with shrinkage estimation on parsimonious feature set",
            "<b>Recommended subset:</b> Top 10–20 nonlinear features from CZ channel (highest discriminative power)",
            "<b>QC gating:</b> Implement per-subject pre-test QC thresholds before decision-making",
        ]
    )
    
    # Slide 8: Ablation Study Design
    generator.add_content_slide(
        "Ablation Study Strategy",
        bullets=[
            "<b>Baseline model:</b> 135 features (spectral + nonlinear + time-domain + Hjorth)",
            "<b>Incremental ablations:</b> Baseline + each additional family (wavelet, MFCC, AR, connectivity)",
            "<b>Cross-validation consistency:</b> Use identical LOSO splits for all models",
            "<b>Statistical testing:</b> Paired Wilcoxon signed-rank tests (per-fold AUC differences)",
            "<b>Auditability:</b> Record fold-level outputs &amp; full config JSON for reproducibility",
        ]
    )
    
    # Slide 9: Output Files &amp; Artifacts
    generator.add_content_slide(
        "Typical Pipeline Outputs",
        bullets=[
            "<b>Feature matrices:</b> all_subjects_features.csv, features_xy.npz (X, y, metadata)",
            "<b>Results tables:</b> classification_results.csv, per_subject_accuracy.csv, subject_summary.csv",
            "<b>Exploration &amp; ablation:</b> biomarker_exploration_screening.csv, ablation_results.csv",
            "<b>Visualizations:</b> ROC curves (per-fold), feature importance heatmaps, per-subject violin plots, confusion matrices",
            "<b>Config &amp; audit trail:</b> experiment_config.json (hyperparameters, feature list, random seed)",
        ]
    )
    
    # Slide 10: Applying to New Datasets
    generator.add_content_slide(
        "Action Items for New Datasets",
        bullets=[
            "1. Prepare channels.tsv mapping &amp; verify event structure &amp; stimulus timing",
            "2. Run preprocessing pipeline &amp; inspect QC summaries (rejection rates, artifact flags)",
            "3. Extract baseline 135 features from preprocessed epochs",
            "4. Run LOSO cross-validation with training-only scaling &amp; feature selection",
            "5. Execute family ablations to identify complementary feature groups",
            "6. If stable biomarkers emerge (AUC &gt; 0.75 ±0.10), retrain compact model for deployment",
            "7. Validate on external dataset if available (gold-standard validation approach)",
        ]
    )
    
    # Slide 11: Repository &amp; Reproducibility
    generator.add_content_slide(
        "Reproducibility &amp; Best Practices",
        bullets=[
            "<b>Version control:</b> Commit all run outputs, config JSON, &amp; results to repo for every experiment",
            "<b>QC automation:</b> Add subject-level QC thresholds &amp; automated rejection gating",
            "<b>Real-time inference:</b> Move toward streaming LDA inference on ROI signals (C3, CZ, C4)",
            "<b>External validation:</b> Benchmark on held-out cohorts (chronic pain patients, multisite data)",
            "<b>Environment reproducibility:</b> Use Docker/Singularity containers &amp; pinned dependency versions",
        ]
    )
    
    # Slide 12: Next Steps
    generator.add_content_slide(
        "Next Steps &amp; Contact",
        bullets=[
            "Repository: <b>AmDeep/EEG-Analysis-Review</b> (https://github.com/AmDeep/EEG-Analysis-Review)",
            "Finalize feature definitions &amp; standardize gamma band (30–80 Hz vs 40–80 Hz)",
            "Compute missing ERD features &amp; document per-sub-experiment electrode availability",
            "Run habituation checks (trial-number regressors) &amp; add plots to analysis notebook",
            "Deploy compact LDA model for clinical pain assessment in real-time applications",
        ]
    )
    
    # Slide 13: Technical Summary
    generator.add_content_slide(
        "Technical Architecture Summary",
        content_text=[
            "<b>Data Flow:</b> Raw EEG (Biosemi, 1024 Hz) → Preprocessing → 256 Hz, 0.5–80 Hz bandpass → "
            "Epoching (2s windows) → 135+ features extracted per epoch",
            "<br/><br/>"
            "<b>Validation:</b> LOSO cross-validation (26 folds) ensures zero test-set leakage. Feature selection "
            "(ANOVA-F) &amp; scaling (StandardScaler) fit on training data only. Per-fold metrics (AUC, sensitivity, "
            "specificity) aggregated across all held-out subjects.",
            "<br/><br/>"
            "<b>Interpretability:</b> LDA coefficients rank features by discriminative power. Connectivity &amp; "
            "entropy measures reveal neural mechanisms underlying pain processing. Audit CSV files document all QC "
            "decisions &amp; preprocessing parameters for full reproducibility.",
        ]
    )
    
    output_file = generator.build()
    return output_file


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("EEG Analysis Pipeline PDF Slide Deck Generator")
        print("=" * 70)
        result = generate_pdf()
        print("=" * 70)
        print(f"SUCCESS: Presentation created at {result}")
        print("=" * 70)
        sys.exit(0)
    except ImportError as e:
        print(f"ERROR: Missing required library: {e}")
        print("Please install reportlab:")
        print("  pip install reportlab")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
