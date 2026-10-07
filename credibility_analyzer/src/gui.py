import sys
import torch  
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit)
from src.scraper import scrape_article
from src.predictor import ArticlePredictor


def get_status_badge(prob_real):
    """Returns an HTML badge styled with status colors based on real/credibility probability."""
    if prob_real >= 0.60:
        return (
            '<span style="background-color: #e8f5e9; color: #2e7d32; '
            'border: 1px solid #a5d6a7; padding: 3px 8px; border-radius: 4px; '
            'font-weight: bold;">LIKELY REAL / CREDIBLE</span>'
        )
    elif prob_real <= 0.40:
        return (
            '<span style="background-color: #ffebee; color: #c62828; '
            'border: 1px solid #ef9a9a; padding: 3px 8px; border-radius: 4px; '
            'font-weight: bold;">LIKELY FAKE / CLICKBAIT</span>'
        )
    else:
        return (
            '<span style="background-color: #fff3e0; color: #ef6c00; '
            'border: 1px solid #ffe0b2; padding: 3px 8px; border-radius: 4px; '
            'font-weight: bold;">UNCERTAIN / UNVERIFIED</span>'
        )


class CredibilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = ArticlePredictor()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Multi-Model Article Credibility & Clickbait Analyzer')
        self.setGeometry(100, 100, 600, 550)

        layout = QVBoxLayout()

        self.label = QLabel('Enter Article URL or Paste Raw Text:', self)
        layout.addWidget(self.label)

        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText('https://example.com/news-article')
        layout.addWidget(self.url_input)

        self.analyze_btn = QPushButton('Analyze Article', self)
        self.analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analyze_btn)

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        self.setLayout(layout)

    def run_analysis(self):
        user_input = self.url_input.text().strip()
        if not user_input:
            self.result_display.setText("Please enter a URL or paste text to analyze.")
            return

        if user_input.startswith("http://") or user_input.startswith("https://"):
            self.result_display.setText("Fetching article contents...")
            text_to_analyze, error = scrape_article(user_input)
            if error:
                self.result_display.setText(f"Error scraping URL: {error}")
                return
        else:
            text_to_analyze = user_input

        results = self.predictor.predict(text_to_analyze)

        # Convert fake probability to real probability (1.0 - prob_fake)
        avg_prob_real = sum((1.0 - data['prob_fake']) for data in results.values()) / len(results)
        overall_badge = get_status_badge(avg_prob_real)

        # Build Rich HTML Output focusing on % Real
        html_output = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h3 style="margin-bottom: 5px;">--- ANALYSIS RESULTS ---</h3>
            <p style="font-size: 14px; margin-top: 5px;">
                <b>Overall Credibility Score:</b> {avg_prob_real * 100:.2f}% Real<br>
                <b>Overall Verdict:</b> {overall_badge}
            </p>
            <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        """

        for model_name, data in results.items():
            prob_real = 1.0 - data['prob_fake']
            real_percentage = prob_real * 100
            badge = get_status_badge(prob_real)
            
            html_output += f"""
            <div style="margin-bottom: 15px;">
                <b style="font-size: 13px;">• {model_name}:</b><br>
                <div style="margin-top: 3px;">Credibility Probability: <b>{real_percentage:.2f}% Real</b></div>
                <div style="margin-top: 3px;">Status: {badge}</div>
            </div>
            """

        html_output += "</div>"

        # Render HTML Output
        self.result_display.setHtml(html_output)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CredibilityApp()
    window.show()
    sys.exit(app.exec_())