{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/najiya123/Fcfs-/blob/main/alzheimersDisease.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Importing Required Packages and Libraries"
      ],
      "metadata": {
        "id": "5jsPCDpA2w0v"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Mount Google Drive"
      ],
      "metadata": {
        "id": "t08j9N94pe44"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H3Uy-iOBpc4V",
        "outputId": "72663d3a-10bc-4430-ef8b-e149ea0516b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd \n",
        "import numpy as np \n",
        "import os\n",
        "from distutils.dir_util import copy_tree, remove_tree\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "import warnings\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout\n",
        "from tensorflow.keras.models import Model\n",
        "from tensorflow.keras.utils import plot_model\n",
        "from tensorflow.keras.preprocessing import image, image_dataset_from_directory\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow import keras\n",
        "from random import randint"
      ],
      "metadata": {
        "id": "0-0AbuANli_4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Unzipping Dataset into the Work Enviornment "
      ],
      "metadata": {
        "id": "xShDX81p255-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!unzip 1.zip"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xBsY0DbXkUVq",
        "outputId": "e97122fc-3a9d-4f72-9b79-018426dc310c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "unzip:  cannot find or open 1.zip, 1.zip.zip or 1.zip.ZIP.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Setting Directory of Dataset inside the Work Enviornment "
      ],
      "metadata": {
        "id": "BRoEpRK43ALO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "default_dir = \"/content/Alzheimer_s Dataset/\"\n",
        "root_dir = \"/\"\n",
        "test_dir = default_dir + \"test/\"\n",
        "train_dir = default_dir + \"train/\"\n",
        "work_dir = root_dir + \"dataset/\"\n",
        "\n",
        "if os.path.exists(work_dir):\n",
        "    remove_tree(work_dir)\n",
        "    \n",
        "\n",
        "os.mkdir(work_dir)\n",
        "copy_tree(train_dir, work_dir)\n",
        "copy_tree(test_dir, work_dir)\n",
        "print(\"Working Directory Contents:\", os.listdir(work_dir))"
      ],
      "metadata": {
        "id": "Hk7I72C2Ld7M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Defining Dataset Parameters"
      ],
      "metadata": {
        "id": "vZ0bN4yH3K9y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "CLASSES = [ 'NonDemented',\n",
        "            'VeryMildDemented',\n",
        "            'MildDemented',\n",
        "            'ModerateDemented']\n",
        "\n",
        "IMG_SIZE = 176\n",
        "IMAGE_SIZE = [176, 176]\n",
        "DIM = (IMG_SIZE, IMG_SIZE)"
      ],
      "metadata": {
        "id": "VCKmxQDgMBib"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Augmentation using ImageDataGenerator"
      ],
      "metadata": {
        "id": "v97KMBuJ3fnj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ZOOM = [.99, 1.01]\n",
        "BRIGHT_RANGE = [0.8, 1.2]\n",
        "HORZ_FLIP = True\n",
        "FILL_MODE = \"constant\"\n",
        "DATA_FORMAT = \"channels_last\"\n",
        "\n",
        "image_generator = ImageDataGenerator(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, \n",
        "                                     data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)"
      ],
      "metadata": {
        "id": "6FzuoOEnMEzx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = image_generator.flow_from_directory(batch_size=4500,\n",
        "                                                    directory=work_dir,\n",
        "                                                    target_size=(176, 176),\n",
        "                                                    shuffle= True)"
      ],
      "metadata": {
        "id": "97hnqJgAMK9n"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Display Images some images in dataset"
      ],
      "metadata": {
        "id": "KPoaPb5L32oR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def show_images(generator,y_pred=None):\n",
        "  \n",
        "    labels =dict(zip([0,1,2,3], CLASSES))\n",
        "    \n",
        "    # get a lot of images\n",
        "    x,y = generator.next()\n",
        "    \n",
        "    # show a grid of 9 images\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    if y_pred is None:\n",
        "        for i in range(9):\n",
        "            ax = plt.subplot(3, 3, i + 1)\n",
        "            idx = randint(0,50)\n",
        "            plt.imshow(x[idx])\n",
        "            plt.axis(\"off\")\n",
        "            plt.title(\"Class:{}\".format(labels[np.argmax(y[idx])]))\n",
        "                                                     \n",
        "    else:\n",
        "        for i in range(9):\n",
        "            ax = plt.subplot(3, 3, i + 1)\n",
        "            plt.imshow(x[i])\n",
        "            plt.axis(\"off\")\n",
        "            plt.title(\"Actual:{} \\nPredicted:{}\".format(labels[np.argmax(y[i])],labels[y_pred[i]]))\n",
        "    \n",
        "# Display Train Images\n",
        "show_images(train_dataset)"
      ],
      "metadata": {
        "id": "pQhGhKAcMN2V",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 201
        },
        "outputId": "2cd1279c-bc16-4b9c-d05e-1d3b76e5dcb1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-4-2c9364d97ba9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     24\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     25\u001b[0m \u001b[0;31m# Display Train Images\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 26\u001b[0;31m \u001b[0mshow_images\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_dataset\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'train_dataset' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Visualizing Dataset on bar chart"
      ],
      "metadata": {
        "id": "KaQCTviP4Av6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = {'NonDemented':      0, \n",
        "        'VeryMildDemented': 0, \n",
        "        'MildDemented':     0,\n",
        "        'ModerateDemented': 0}\n",
        "\n",
        "# visualizing dataset\n",
        "for cls in os.listdir(work_dir):\n",
        "    for img in os.listdir(work_dir + '/' + cls):\n",
        "        data[cls] = data[cls] + 1\n",
        "        \n",
        "keys = list(data.keys())\n",
        "values = list(data.values())\n",
        "  \n",
        "fig = plt.figure(figsize = (10, 5))\n",
        " \n",
        "plt.bar(keys, values, color=('lightgreen'), width = 0.4)"
      ],
      "metadata": {
        "id": "7FQTs_fvMSQC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data, train_labels = train_dataset.next()"
      ],
      "metadata": {
        "id": "sromffVDNZOF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_data.shape, train_labels.shape)"
      ],
      "metadata": {
        "id": "uhFeZHvaNg0U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Resambling Data"
      ],
      "metadata": {
        "id": "mBtHYjNb5dWg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sm = SMOTE(random_state=42)\n",
        "train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)\n",
        "train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)\n",
        "print(train_data.shape, train_labels.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 218
        },
        "id": "cvKVMgb6NyIV",
        "outputId": "0717d850-e6a6-4c19-d958-c885c63c71d6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-4-3356d3d87ae8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0msm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSMOTE\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m42\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mtrain_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_labels\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msm\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_resample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mIMG_SIZE\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mIMG_SIZE\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_labels\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mtrain_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mIMG_SIZE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mIMG_SIZE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_labels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'train_data' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Preparing Model for Training"
      ],
      "metadata": {
        "id": "GmoGX7765inJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)\n",
        "train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)"
      ],
      "metadata": {
        "id": "9mGDnPlDTNJf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = keras.models.Sequential([keras.layers.Flatten(input_shape = [176,176,3]),  \n",
        "keras.layers.Dense(100, activation = 'relu' ),                               \n",
        "keras.layers.Dense(200, activation = 'relu' ),\n",
        "\n",
        "keras.layers.Dense(200, activation = 'relu' ),\n",
        "\n",
        "keras.layers.Dense(200, activation = 'relu' ),\n",
        "\n",
        "keras.layers.Dense(200, activation = 'relu' ),\n",
        "keras.layers.Dense(4, activation = 'softmax')])"
      ],
      "metadata": {
        "id": "uveB6GXcTpTc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "callback = keras.callbacks.EarlyStopping(monitor='val_loss',\n",
        "                                            patience=3,\n",
        "                                            restore_best_weights=True)"
      ],
      "metadata": {
        "id": "iJNQkGkETt1K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam',\n",
        "loss=tf.losses.CategoricalCrossentropy(),\n",
        "metrics=[keras.metrics.AUC(name='auc')])"
      ],
      "metadata": {
        "id": "nRA32xmcTzID"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training with 50 epochs"
      ],
      "metadata": {
        "id": "uF44fbjb6Lyb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "result = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50)"
      ],
      "metadata": {
        "id": "-MkbK80mT2jf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluating the model"
      ],
      "metadata": {
        "id": "Gem7jDtG6WIt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "loss, accuracy = model.evaluate(test_data, test_labels)\n",
        "print(\"Loss: \", loss)\n",
        "print(\"Accuracy: \", accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 201
        },
        "id": "VOX3OmEYfPd-",
        "outputId": "02bd8ca5-e67d-4cc9-ed0d-ef752a9e85cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-7-89076be33ca1>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mloss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mevaluate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_labels\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Loss: \"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloss\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Accuracy: \"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccuracy\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Saving Model"
      ],
      "metadata": {
        "id": "_hliqxgO6agH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "\n",
        "model.save('model.h5')\n",
        "files.download('model.h5')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 218
        },
        "id": "CRF9H9kKUQfJ",
        "outputId": "d0d24b03-1fd4-4a82-e41e-22c55b3244d0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-6-1633a95f0a29>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mgoogle\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolab\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mfiles\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'model.h5'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mfiles\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdownload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'model.h5'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ]
    }
  ]
}