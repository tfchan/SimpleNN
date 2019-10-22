"""Main program to test simple NN."""
import numpy as np
import matplotlib.pyplot as plt
import simplenn


def generate_linear(n=100):
    """Generate linear dataset."""
    inputs = np.random.uniform(size=(n, 2))
    labels = []
    for pt in inputs:
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.asarray(inputs), np.asarray(labels)


def generate_xor_easy():
    """Generate xor dataset."""
    inputs = []
    labels = []
    for i in range(11):
        x = 0.1 * i
        inputs.append([x, x])
        labels.append(0)
        if x == 0.5:
            continue
        inputs.append([x, 1 - x])
        labels.append(1)
    return np.asarray(inputs), np.asarray(labels)


def show_result(x, y_true, y_pred):
    """Plot ground truth and prediction results."""
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    plt.axis('scaled')
    plt.scatter(x[:, 0], x[:, 1], c=y_true)
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    plt.axis('scaled')
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()


def main():
    """Test simple NN."""
    # Dataset 1
    x1, y1 = generate_linear()
    model = simplenn.SimpleNN()
    model.add(simplenn.Dense(2, activation='sigmoid', use_bias=False))
    model.add(simplenn.Dense(2, activation='sigmoid', use_bias=False))
    model.add(simplenn.Dense(1, activation='sigmoid', use_bias=False))
    model.fit(x1, y1, lr=1, epochs=5000, early_stopping_loss=0.1)
    y1_pred = model.predict(x1)
    y1_pred = ((y1_pred > 0.5) * 1).flatten()
    show_result(x1, y1, y1_pred)

    # Dataset 2
    x2, y2 = generate_xor_easy()
    model = simplenn.SimpleNN()
    model.add(simplenn.Dense(3, activation='sigmoid', use_bias=False))
    model.add(simplenn.Dense(3, activation='sigmoid', use_bias=False))
    model.add(simplenn.Dense(1, activation='sigmoid', use_bias=False))
    model.fit(x2, y2, lr=1, epochs=5000, early_stopping_loss=0.1)
    y2_pred = model.predict(x2)
    y2_pred = ((y2_pred > 0.5) * 1).flatten()
    show_result(x2, y2, y2_pred)


if __name__ == "__main__":
    main()
