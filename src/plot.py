import matplotlib.pyplot as plt


def plot_all(path, train_losses, dev_losses, dev_f2s, dev_ths, dev_d2s_1, dev_d2s_2):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label=f'train loss: {train_losses[-1]:.3}')
    plt.plot(dev_losses, label=f'dev loss: {dev_losses[-1]:.3}')
    plt.xlabel('#epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(ls='--')
    
    plt.subplot(2, 2, 2)
    plt.plot(dev_f2s, label=f'dev f2: {float(dev_f2s[-1]):.3} ({dev_ths[-1]})')
    plt.xlabel('#epoch')
    plt.ylabel('F2 score')
    plt.legend()
    plt.grid(ls='--')
    
    def _plot(arr):
        for key in arr[-1]:
            fb = [item[key] for item in arr]
            plt.plot(fb, label=f'{key} ({fb[-1]:.3})')
        plt.xlabel('#epoch')
        plt.ylabel('Fbeta score')
        plt.legend()
        plt.grid(ls='--')
    
    plt.subplot(2, 2, 3)
    _plot(dev_d2s_1)
    
    plt.subplot(2, 2, 4)
    _plot(dev_d2s_2)
    
    plt.savefig(path / 'training_evolution.png')
    plt.close()
