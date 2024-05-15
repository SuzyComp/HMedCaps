import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_save_confmatrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    #plot_confusion_matrix(conf_mat=cm, figsize=(8,8), show_normed=True)
    #plt.savefig('result/conf.png')
    ax = plt.subplot()
    sns.set(font_scale=1.0)  # Adjust to fit
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g")


    # Labels, title and ticks
    label_font = {'size': '10'}  # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '12'}  # Adjust to fit
    ax.set_title('Confusion Matrix', fontdict=title_font)

    report= classification_report(y_test,y_pred)
    df = pd.DataFrame([report]).transpose()
    df.to_csv('result/base_model/report.csv')
    print(df)

    '''
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust to fit
    ax.xaxis.set_ticklabels(['False', 'True']);
    ax.yaxis.set_ticklabels(['False', 'True']);
    '''
    fig.savefig('result/base_model/conf.png')