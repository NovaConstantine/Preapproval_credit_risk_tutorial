# Run this cell after generating y_pred_logreg, y_pred_rf, and y_pred_xgb.
# It saves presentation-quality images in the figures subfolder of the
# notebook's working directory (normally alongside lending_club_tutorial.csv).
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model_predictions = [
    ('Logistic Regression', y_pred_logreg),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb),
]
matrices = [confusion_matrix(y_test, pred, labels=[0, 1])
            for _, pred in model_predictions]
max_count = max(matrix.max() for matrix in matrices)
output_dir = Path('figures')
output_dir.mkdir(parents=True, exist_ok=True)

# A dedicated fourth column reserves space for the shared colorbar.
# Avoid tight_layout(): it can move the model panels onto a shared colorbar.
with plt.rc_context({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.transparent': False,
    'text.color': '#172B4D', 'axes.labelcolor': '#172B4D',
    'xtick.color': '#172B4D', 'ytick.color': '#172B4D',
    'font.size': 12, 'axes.grid': False,
    'pdf.fonttype': 42,
}):
    fig = plt.figure(figsize=(16, 5), dpi=160)
    grid = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.045],
        left=0.065, right=0.935, bottom=0.18, top=0.83, wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    colorbar_ax = fig.add_subplot(grid[0, 3])

    for ax, (name, _), matrix in zip(axes, model_predictions, matrices):
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=['No Default', 'Default'],
        )
        display.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        display.im_.set_clim(0, max_count)
        for value, text_label in zip(matrix.ravel(), display.text_.ravel()):
            text_label.set_color('white' if value > max_count / 2 else '#172B4D')
            text_label.set_fontsize(14)
        ax.set_title(name, fontsize=15, pad=12)
        ax.set_xlabel('Predicted label', fontsize=12, labelpad=9)
        ax.set_ylabel('True label', fontsize=12, labelpad=9)
        ax.grid(False, which='both')
        ax.axvline(0.5, color='white', linewidth=1.5)
        ax.axhline(0.5, color='white', linewidth=1.5)
        ax.tick_params(axis='both', which='both', length=0, pad=7)

    colorbar = fig.colorbar(display.im_, cax=colorbar_ax)
    colorbar.set_label('Number of loans', labelpad=12)
    fig.suptitle('Confusion Matrices at threshold = 0.50', fontsize=18, y=0.95)

    # PNG: exactly 9600 x 3000 pixels at 600 dpi; PDF keeps text/lines as vectors.
    png_path = output_dir / 'confusion_matrices_600dpi.png'
    pdf_path = output_dir / 'confusion_matrices.pdf'
    fig.savefig(png_path, dpi=600, facecolor='white', transparent=False)
    fig.savefig(pdf_path, dpi=600, facecolor='white', transparent=False)
    plt.show()

print(f'PNG saved to: {png_path.resolve()}')
print(f'PDF saved to: {pdf_path.resolve()}')
