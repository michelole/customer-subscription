from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


def plot_confusion_matrix(y, y_pred, title):
    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd', xticklabels=['0', '1'],
        yticklabels=['0', '1'])
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()
