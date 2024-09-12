"""
Authors: Ricardo Martinez and Aman Dwivedi
Instructor: Rich Thompson
ISTA 131 Final Project
Description: A convolutional neural network that was trained using the MNIST digit dataset of 60000 entries in CSV. It
             predicts what a user has drawn onto a canvas. We have also made a menu for visualizing many aspects such
             as the clustered data manifold in multidimensional latent space using principle component analysis (PCA).
             Some libraries used in this script are matplotlib, numpy, tensorflow, cv2, etc.
"""
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.decomposition import PCA
import cv2


def get_data():
    """
    This function reads in the training and testing data for the model to fit. It
    creates two separate DataFrames for train and test data.
    """
    train_df = pd.read_csv("mnist_train.csv")
    test_df = pd.read_csv("mnist_test.csv")

    train_images = train_df.drop("label", axis=1)
    train_labels = train_df["label"]
    test_images = test_df.drop("label", axis=1)
    test_labels = test_df["label"]

    return train_df, train_images, train_labels, test_images, test_labels


def shape_model(train_images, train_labels, test_images, test_labels):
    """
    This function reshapes the matrix shape to make it compatible and
    standardize it for the model.
    """
    train_images = train_images.to_numpy().reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.to_numpy().reshape(test_images.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    return input_shape, train_images, train_labels, test_images, test_labels


def train_model(epochs, input_shape, train_images, train_labels):
    """
    This function takes the data as arrays and feeds it to the
    convolutional neural network for training.
    """
    model = Sequential()
    print()
    print("Train:")
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("")
    model.summary()
    print("")
    logger = tf.keras.callbacks.CSVLogger("history.csv", separator=",", append=True)
    history = model.fit(x=train_images, y=train_labels, epochs=epochs, callbacks=[logger])

    return model, history


def test_model(model, test_images, test_labels):
    """
    This function takes the test data for the model to test on.
    """
    print()
    print("Test:")
    history = model.evaluate(test_images, test_labels)

    return history


def display_image(image_index, train_images, input_label=None):
    """
    Aman's Visualization 1
    This function is used to display a single image of a specified number on the plot.
    """
    plt.figure(figsize=(7, 7))
    pixel_data = train_images.iloc[image_index].to_numpy().reshape(28, 28)
    plt.imshow(pixel_data, interpolation="none", cmap='gray')  # "Greys" for reverse color effect
    plt.xlabel("X pixel data")
    plt.ylabel("Y pixel data")
    if input_label is None:
        plt.title("Number at index " + str(image_index))
    else:
        plt.title("Image of " + str(input_label) + " at index " + str(image_index))
    plt.show()


def display_multiple_images(train_df, train_images, train_labels, test_images, test_labels):
    """
    Aman's Visualization 1
    This function is used to display nine different images of nine randomly chosen numbers from the dataframe.
    """
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle("Nine random indexes from csv to images")
    rows = 3
    columns = 3
    image_index = random.randint(0, 59999)
    i = 1
    train_df, train_images, train_labels, test_images, test_labels = get_data()
    while i <= 9:
        pixel_data = train_images.iloc[image_index].to_numpy().reshape(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(pixel_data, interpolation="none",
               cmap='gray')  # "Greys" for reverse color effect
        plt.axis('off')
        plt.title(str(train_df.label.iloc[image_index]))
        i += 1
        image_index = random.randint(0, 59999)
    plt.show()


def display_bar_chart(dictionary):
    """
    Aman's Visualization 2
    This function is used to draw a bar graph depicting how scattered the data is.
    """
    plt.bar(dictionary.keys(), dictionary.values())
    plt.ylabel("Number of Occurrences")
    plt.xlabel("Digit")
    plt.title("Distribution of Data")
    plt.xticks(range(10))
    plt.show()


def display_linear_regression(image_index, train_df, train_images, train_labels):
    """
    Aman's Visualization 3
    This function uses PCA to condense the 784 dimensions into a 3 dimensional array and runs a 3 dimensional linear
    regression model. It limits to only 100 data points to generate an optimized model.
    """
    dataframe = pd.DataFrame(columns=train_df.columns)
    count = 0
    for i in image_index:
        dataframe = pd.concat([dataframe, train_df.loc[i].to_frame().T])
        count += 1
        if count > 100:
            break
    num = dataframe.iloc[0][0]
    feat_cols = (dataframe.columns).drop("label")
    new_dataframe = pd.DataFrame(dataframe, columns=feat_cols)
    new_dataframe['y'] = train_labels
    new_dataframe['label'] = new_dataframe['y'].apply(lambda i: str(i))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(new_dataframe[feat_cols].values)

    X = []
    Y = []

    for arr in pca_result[:100]:
        X.append([float(arr[0]), float(arr[1]), 1])
        Y.append(float(arr[2]))
    X = np.array(X)
    Y = np.array(Y)

    a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

    xx, yy, zz = np.meshgrid(X[:, 0], X[:, 1], X[:, 2])
    combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    Z = combinedArrays.dot(a)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y, color='r', label=num)
    ax.set_title('Data Manifold of the first 100 appearances of digit ' + str(num) + ' in the dataset')
    ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()

    plt.show()


def display_manifold(train_df, train_images, train_labels):
    """
    Ricardo's Visualization 1
    This function uses PCA to condense the 784 dimensions into a 3 dimensional array and runs a 3 dimensional linear
    regression model. This function displays the data manifold in three-dimensional latent space via principal component
    analysis. Each color represents the differently labeled data from 0-9 each with its respective color. PCA is
    used to shrink the 784 dimensions for each input data the convolutional neural network is fed into a perceivable
    rank 3 using eigenvalues and eigenvectors of the correlation tensor/matrix. PCA is a common technique for reducing
    the dimensionality of data and can be useful when visualizing high dimensional data when the data are correlated.
    With this, we can find relationships between each of the inputs and compare how they vary with one another. Using
    the visualization, we can infer how the model will conjure the decision boundary around the scattered data
    manifolds In the latent space using softmax and ReLu loss functions. In a way, you can treat the decision boundary
    as a function of the model's chosen multidimensional standard deviation of such statistical observations. The model
    still conjures the decision boundary above the three dimensions we are visualizing as it is still using the 784
    dimensions with gradient descent to achieve the result we desire.
    """
    feat_cols = (train_df.columns).drop("label")
    df = pd.DataFrame(train_images, columns=feat_cols)
    df['y'] = train_labels
    df['label'] = df['y'].apply(lambda i: str(i))

    rp = np.random.permutation(df.shape[0])

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=df.loc[rp, :]["pca-one"],
        ys=df.loc[rp, :]["pca-two"],
        zs=df.loc[rp, :]["pca-three"],
        c=df.loc[rp, :]["y"], cmap='tab10')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend(*scatter.legend_elements())
    ax.set_title('Clustered Data Manifold in Latent Space Using PCA')
    plt.show()


def display_accuracy(history):
    """
    Ricardo's Visualization 2
    This function takes the history data from the model and generates a line graph to visualize the accuracy over epoch.
    """
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def display_loss(history):
    """
    Ricardo's Visualization 3
    This function takes the history data from the model and generates a line graph to visualize the loss over epoch.
    """
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def display_canvas(model):
    """
    Ricardo's Visualization 4
    This function opens up a canvas for the user to write a digit for the model to predict.
    """
    run = False
    win = np.zeros((28, 28, 1), dtype="float64")

    def draw(event, x, y, flag, param):
        global run
        # win = np.zeros((28,28,1), dtype="float64")
        if event == cv2.EVENT_LBUTTONDOWN:
            run = True
            cv2.rectangle(win, (x, y), (x, y), (255, 255, 255), -1)
        if event == cv2.EVENT_LBUTTONUP:
            run = False
            print("The model guesses " + str(
                np.argmax(model.predict(win.reshape(1, 28, 28, 1)), axis=1)[0]))
        if event == cv2.EVENT_MOUSEMOVE:
            if run:
                cv2.rectangle(win, (x, y), (x, y), (255, 255, 255), -1)

    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Canvas", 500, 500)
    cv2.setMouseCallback("Canvas", draw)

    while (True):
        cv2.imshow("Canvas", win)
        k = cv2.waitKey(1)
        if k == ord("c"):
            win = np.zeros((28, 28, 1), dtype="float64")
        if k == 27:
            cv2.destroyAllWindows()
            break


def main():
    train_df, train_images, train_labels, test_images, test_labels = get_data()
    input_shape, train_images, train_labels, test_images, test_labels = shape_model(train_images, train_labels,
                                                                                    test_images, test_labels)

    print("")
    print("Number of training images:", train_images.shape[0])
    print("Number of test images:", test_images.shape[0])
    print("")
    print("Would you like to load trained model? (y/n)")
    print("(Enter n if you don't have a pretrained model to load)")

    load = input()

    if load == "y":
        model = keras.models.load_model("model")
        history = model
        print("")
        print("Model Loaded.")

    else:
        print("")
        print("Please enter number of epoch for training:")
        print("(10 epoch is recommended)")

        epochs = int(input())

        model, history = train_model(
            epochs, input_shape, train_images, train_labels)

        print("")
        print("Training complete.")

    choice = ""

    while choice != "10":
        print("")
        print("Menu:")
        print("----------------------------")
        print("")
        print("Visuals:")
        print("1. Display input images")
        print("2. Display the number of occurrences for each digit")
        print("3. Display linear regression for a specific digit")
        print("4. Display data manifold in latent space using principal component analysis")
        print("5. Display accuracy of the model over time")
        print("6. Display loss of the model over time")
        print("")
        print("Operations:")
        print("7. Save model")
        print("8. Evaluate the model to test its accuracy over untrained data")
        print("9. Draw on a canvas to challenge the model")
        print("10. Exit the program")

        choice = input()

        if choice == "1":
            print("")
            print("What do you want to search by: \n1. Label \n2. Index \n3. Random Images")
            search = input()
            if search == "1":
                print("Enter a label:")
                print("(Range is 0-9)")
                input_label = int(input())
                image_index = train_df.label[train_df.label == input_label].index[random.randint(0, 2000)]
                train_df, train_images, train_labels, test_images, test_labels = get_data()
                display_image(image_index, train_images, input_label)
            elif search == "2":
                print("Enter the index of the image you want to visualize:")
                print("(Maximum index is " + str(train_images.shape[0] - 1) + ")")
                image_index = int(input())
                train_df, train_images, train_labels, test_images, test_labels = get_data()
                display_image(image_index, train_images)
            elif search == "3":
                display_multiple_images(train_df, train_images, train_labels, test_images, test_labels)
        elif choice == "2":
            series = train_df.label
            dictionary = {}
            for key in series:
                if key not in dictionary:
                    dictionary[key] = 0
                dictionary[key] += 1
            display_bar_chart(dictionary)
        elif choice == "3":
            print("Enter a label:")
            print("(Range is 0-9)")
            input_label = int(input())
            image_index = list(train_df.label[train_df.label == input_label].index)
            train_df, train_images, train_labels, test_images, test_labels = get_data()
            display_linear_regression(image_index, train_df, train_images, train_labels)
        elif choice == "4":
            train_df, train_images, train_labels, test_images, test_labels = get_data()
            display_manifold(train_df, train_images, train_labels)
        elif choice == "5":
            train_df, train_images, train_labels, test_images, test_labels = get_data()
            input_shape, train_images, train_labels, test_images, test_labels = (
                shape_model(train_images, train_labels, test_images,
                            test_labels))
            display_accuracy(history)
        elif choice == "6":
            train_df, train_images, train_labels, test_images, test_labels = get_data()
            input_shape, train_images, train_labels, test_images, test_labels = (
                shape_model(train_images, train_labels, test_images,
                            test_labels))
            display_loss(history)
        elif choice == "7":
            model.save("model", overwrite=True)
            print("")
            print("Model has been saved.")
        elif choice == "8":
            train_df, train_images, train_labels, test_images, test_labels = get_data()
            input_shape, train_images, train_labels, test_images, test_labels = (
                shape_model(train_images, train_labels, test_images,
                            test_labels))
            test_model(model, test_images, test_labels)
            print("")
            print("Testing complete.")
        elif choice == "9":
            print("")
            print("Enter c to clear and esc to exit the canvas.")
            display_canvas(model)


main()
