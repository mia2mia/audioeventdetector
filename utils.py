import matplotlib.pyplot as plt

def plot_metrics(history, show=False):
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    plt.plot(history['loss'], '-o', label='train')
    plt.plot(history['val_loss'], '-o', label='val')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.plot(history['acc'], '-o', label='train')
    plt.plot(history['val_acc'], '-o', label='val')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig('metrics.png')
    plt.gcf().set_size_inches(15, 12)
    if show:
        plt.show()
    else:
        plt.close()
