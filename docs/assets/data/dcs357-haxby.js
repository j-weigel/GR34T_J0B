window.NOTEBOOKS = window.NOTEBOOKS || {};
window.NOTEBOOKS["dcs357-haxby"] = {
 "name": "CoNeu_LB+JW_Final.ipynb",
 "kind": "notebook",
 "download": "assets/notebooks/CoNeu_LB+JW_Final.ipynb",
 "cells": [
  {
   "type": "markdown",
   "source": "#***Project Overview***\n---\nThis project analyzes the Haxby fMRI dataset, in which a subject viewed images from eight categories (faces, houses, cats, bottles, scissors, shoes, chairs, and scrambled images), with ventral temporal cortex responses reduced to 50 principal components. We ask whether the brain represents these categories in distinguishable patterns. First, we explore how mean PC scores differ across categories and between animate and inanimate objects. Next, we train logistic regression and k-nearest-neighbor classifiers to decode whether the subject was viewing a face from brain activity alone, evaluating them with cross-validation. Finally, we use PCA and t-SNE to visualize how the image categories are arranged in the brain's representational space."
  },
  {
   "type": "markdown",
   "source": "\n#***Housekeeping***"
  },
  {
   "type": "code",
   "source": "#Initial importing!\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns",
   "outputs": []
  },
  {
   "type": "code",
   "source": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE",
   "outputs": []
  },
  {
   "type": "code",
   "source": "df = pd.read_csv('Haxby_Reduced.csv')",
   "outputs": []
  },
  {
   "type": "code",
   "source": "df.head()",
   "outputs": [
    {
     "kind": "text",
     "text": "         PC1        PC2       PC3       PC4       PC5       PC6       PC7  \\\n0  22.902924   9.937236  4.117603 -4.414197  6.464055 -0.358261  7.426979   \n1  23.118841  10.623929  6.116805 -1.897333  9.137955  1.991398  7.955241   \n2  23.164461  10.472020  5.781077 -2.949358  6.467790  0.118873  7.262615   \n3  24.276384   9.842486  5.717005 -5.073019  6.768382 -1.555429  9.634997   \n4  24.194920  11.153329  5.148042 -4.504280  6.590526 -1.010427  8.731315   \n\n        PC8       PC9      PC10  ...      PC43      PC44      PC45      PC46  \\\n0 -2.960018 -1.802101  7.378876  ... -0.562315 -0.725948  1.489982  0.163816   \n1  0.816305  0.881092  8.341912  ... -0.914649 -0.660885  1.910888 -0.279939   \n2  1.880418  1.607294  7.707685  ... -0.821352 -0.000830  0.868696 -1.362601   \n3  2.351414  0.969117  6.799834  ...  0.370702  0.213020  1.334632 -0.372242   \n4  2.035034  1.246951  6.895126  ... -1.020189  0.608991  0.897321 -0.988796   \n\n       PC47      PC48      PC49      PC50  category  session  \n0  0.458516 -1.301533  1.112287 -1.537357  scissors        0  \n1  0.136691 -0.187726  0.237617 -1.215965  scissors        0  \n2 -0.072011  0.342061 -0.746108 -0.617868  scissors        0  \n3 -0.045337 -0.603027 -0.150477  0.450504  scissors        0  \n4  0.843438 -0.126527 -0.386046  0.231578  scissors        0  \n\n[5 rows x 52 columns]"
    }
   ]
  },
  {
   "type": "code",
   "source": "df.shape",
   "outputs": [
    {
     "kind": "text",
     "text": "(864, 52)"
    }
   ]
  },
  {
   "type": "code",
   "source": "df['category'].unique()",
   "outputs": [
    {
     "kind": "text",
     "text": "array(['scissors', 'face', 'cat', 'shoe', 'house', 'scrambledpix',\n       'bottle', 'chair'], dtype=object)"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "#***Exploratory Analysis***\n---\nThe Haxby dataset we are analyzing contains fMRI data from a subject viewing an array of images (faces, houses, cats, bottled, scissors, schoes, chairs, and scrambled images). The brain resoonses have been projected into fifty individual principal components (PCs) to capture the dominat axes of variation in the ventral temporal cortex. The central question to our project is: does the brain represent different image categories in distinguishable patterns?"
  },
  {
   "type": "code",
   "source": "#Number of trials per category per session\ntrial_table = df.groupby(['session', 'category']).size().unstack()\nprint(trial_table)",
   "outputs": [
    {
     "kind": "text",
     "text": "category  bottle  cat  chair  face  house  scissors  scrambledpix  shoe\nsession                                                                \n0              9    9      9     9      9         9             9     9\n1              9    9      9     9      9         9             9     9\n2              9    9      9     9      9         9             9     9\n3              9    9      9     9      9         9             9     9\n4              9    9      9     9      9         9             9     9\n5              9    9      9     9      9         9             9     9\n6              9    9      9     9      9         9             9     9\n7              9    9      9     9      9         9             9     9\n8              9    9      9     9      9         9             9     9\n9              9    9      9     9      9         9             9     9\n10             9    9      9     9      9         9             9     9\n11             9    9      9     9      9         9             9     9\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Number of trials per image category\nplt.figure(figsize = (8,4))\norder = df['category'].value_counts().index\nsns.countplot(data = df, x = 'category', order = order, color = \"hotpink\")\nplt.xlabel(\"Category\")\nplt.ylabel(\"Count\")\nplt.title(\"Number of Trials per Image Category\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-10-0.png",
     "width": 695,
     "height": 393
    }
   ]
  },
  {
   "type": "code",
   "source": "#Calculate the average of each PC for each image category\npc_cols = [f'PC{i}' for i in range(1, 51)]\nmean_pc_by_category = df.groupby('category')[pc_cols].mean()\n\nprint(mean_pc_by_category)",
   "outputs": [
    {
     "kind": "text",
     "text": "                   PC1       PC2       PC3       PC4       PC5       PC6  \\\ncategory                                                                   \nbottle       -0.322317 -0.272482 -0.498198 -0.246988 -0.372392 -0.502033   \ncat          -0.454458 -0.095770  1.278729  1.166379 -0.350507 -0.599001   \nchair        -0.522013  1.391863  0.383528  2.136354  0.911452 -0.302272   \nface          0.151269 -2.378028 -1.406435 -2.143886 -3.450374 -2.572489   \nhouse         1.172834  1.961664  1.106213  2.308260  3.912506  4.315336   \nscissors      0.138620 -0.143393 -0.705993 -0.991656  0.710006  0.380989   \nscrambledpix -0.175803 -0.733296 -0.955505 -2.032776 -2.160929 -0.717540   \nshoe          0.011871  0.269444  0.797666 -0.195683  0.800240 -0.002989   \n\n                   PC7       PC8       PC9      PC10  ...      PC41      PC42  \\\ncategory                                              ...                       \nbottle       -0.303166  1.632075  0.630217  0.352516  ...  0.336613 -0.354195   \ncat          -0.386553 -0.160061  0.022413  0.869311  ...  0.103114  0.080579   \nchair        -0.793785  0.273726  0.125944 -0.970528  ... -0.281352  0.172546   \nface          0.227759 -1.427040 -0.208268  0.317265  ... -0.266309 -0.014586   \nhouse         1.520819 -3.084463 -0.992500 -2.386035  ...  0.160734 -0.047690   \nscissors      0.941701  2.691831  0.329359  1.123695  ...  0.327253  0.190500   \nscrambledpix  0.266602 -1.510451 -0.464110  0.229059  ... -0.032416  0.211383   \nshoe         -1.473380  1.584385  0.556944  0.464710  ... -0.347641 -0.238534   \n\n                  PC43      PC44      PC45      PC46      PC47      PC48  \\\ncategory                                                                   \nbottle       -0.012538  0.117231 -0.085516 -0.013725  0.215174  0.073239   \ncat           0.098173 -0.007712 -0.019898 -0.237498  0.080511 -0.152270   \nchair         0.030037  0.118422 -0.171819  0.054987 -0.114513 -0.232476   \nface         -0.111279 -0.060667 -0.000775  0.091142 -0.000042  0.082027   \nhouse        -0.092584 -0.119131  0.044127  0.013310 -0.051885  0.018831   \nscissors     -0.057760 -0.354951  0.114107  0.021013 -0.081635  0.126087   \nscrambledpix -0.026652 -0.006645  0.112492 -0.090550 -0.139762 -0.060492   \nshoe          0.172605  0.313453  0.007291  0.161328  0.092148  0.145047   \n\n                  PC49      PC50  \ncategory                          \nbottle       -0.146192 -0.033307  \ncat           0.146768  0.109889  \nchair        -0.105508  0.058200  \nface         -0.097700 -0.081035  \nhouse        -0.048916  0.032121  \nscissors      0.148191  0.022065  \nscrambledpix  0.066794  0.065730  \nshoe          0.036559 -0.173660  \n\n[8 rows x 50 columns]\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Heatmap of average PCs for each image category\nplt.figure(figsize=(12, 6))\nsns.heatmap(mean_pc_by_category, cmap='PiYG', center=0)\nplt.title(\"Mean PC Scores by Category\")\nplt.xlabel(\"Principal Components\")\nplt.ylabel(\"Image Category\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-12-0.png",
     "width": 986,
     "height": 569
    }
   ]
  },
  {
   "type": "code",
   "source": "mean_pc_by_category.T.plot(figsize=(12,6), cmap = \"hsv\")\nplt.title(\"PCs Across Categories\")\nplt.xlabel(\"Principal Component\")\nplt.ylabel(\"Mean Score\")\nplt.legend(title='Category')\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-13-0.png",
     "width": 999,
     "height": 547
    }
   ]
  },
  {
   "type": "code",
   "source": "#Mean PC1 Score Per Category\nmean_pc1 = df.groupby('category')['PC1'].mean().sort_values()\nplt.figure(figsize = (8,4))\nmean_pc1.plot(kind = 'bar', color = 'hotpink',)\nplt.title(\"Mean PC1 Score by Category\")\nplt.xlabel(\"Category\")\nplt.ylabel(\"Mean PC1 Score\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-14-0.png",
     "width": 711,
     "height": 473
    }
   ]
  },
  {
   "type": "code",
   "source": "#Mean PC2 Score Per Category\nmean_pc2 = df.groupby('category')['PC2'].mean().sort_values()\nplt.figure(figsize = (8,4))\nmean_pc2.plot(kind = 'bar', color = 'hotpink',)\nplt.title(\"Mean PC2 Score by Category\")\nplt.xlabel(\"Category\")\nplt.ylabel(\"Mean PC2 Score\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-15-0.png",
     "width": 689,
     "height": 473
    }
   ]
  },
  {
   "type": "code",
   "source": "import seaborn as sns\nplt.figure(figsize=(8, 6))\nsns.scatterplot(data=df, x='PC1', y='PC2', hue='category', palette='hsv')\nplt.title(\"PC1 vs. PC2 Colored by Category\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-16-0.png",
     "width": 698,
     "height": 547
    }
   ]
  },
  {
   "type": "markdown",
   "source": "**Animate vs. Inanimate Objects Response**"
  },
  {
   "type": "code",
   "source": "animate_categories = ['face', 'cat']\ninanimate_categories = ['house', 'scissors', 'shoe', 'scrambledpix', 'bottle', 'chair']\n\ndf['animacy'] = None\n\ndf.loc[df['category'].isin(animate_categories), 'animacy'] = 'animate'\ndf.loc[df['category'].isin(inanimate_categories), 'animacy'] = 'inanimate'\n\npc_cols = [f'PC{i}' for i in range(1, 51)]\nanimacy_means = df.groupby('animacy')[pc_cols].mean()\nprint(animacy_means)",
   "outputs": [
    {
     "kind": "text",
     "text": "                PC1       PC2       PC3       PC4       PC5       PC6  \\\nanimacy                                                                 \nanimate   -0.151595 -1.236899 -0.063853 -0.488754 -1.900441 -1.585745   \ninanimate  0.050532  0.412300  0.021285  0.162919  0.633480  0.528582   \n\n                PC7       PC8       PC9      PC10  ...      PC41      PC42  \\\nanimacy                                            ...                       \nanimate   -0.079397 -0.793551 -0.092927  0.593288  ... -0.081598  0.032996   \ninanimate  0.026465  0.264517  0.030976 -0.197764  ...  0.027199 -0.010998   \n\n               PC43      PC44      PC45      PC46      PC47      PC48  \\\nanimacy                                                                 \nanimate   -0.006553 -0.034190 -0.010337 -0.073178  0.040234 -0.035122   \ninanimate  0.002185  0.011396  0.003447  0.024394 -0.013412  0.011706   \n\n               PC49      PC50  \nanimacy                        \nanimate    0.024534  0.014427  \ninanimate -0.008178 -0.004808  \n\n[2 rows x 50 columns]\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "plt.figure(figsize=(12, 3))\nsns.heatmap(animacy_means, cmap='PiYG')\nplt.title(\"Mean PC Scores for Animate vs Inanimate Images\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-19-0.png",
     "width": 908,
     "height": 318
    }
   ]
  },
  {
   "type": "markdown",
   "source": "#***Supervised Task***\n\n---\n\nCan a classifier determine whether the subject was looking at a face based solely on the brain activity?"
  },
  {
   "type": "code",
   "source": "#Create X and y labels\nX = df[[f\"PC{i}\" for i in range (1,51)]].values\ny = df['category'].values\n\n#Remove scrambledpix because not real category for analysis\nmask = y != 'scrambledpix'\nX = X[mask]\ny = y[mask]\n\nprint(f\"Post scrambledpix datashape: {X.shape}\")\nprint(f\"Categories: {np.unique(y)}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Post scrambledpix datashape: (756, 50)\nCategories: ['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'shoe']\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Filter for face vs. not face images\ny_face = (y == 'face').astype(int)\nprint(f'Face trials (y=1): {y_face.sum()}')\nprint(f'Non-face trials (y=0): {(y_face==0).sum()}')",
   "outputs": [
    {
     "kind": "text",
     "text": "Face trials (y=1): 108\nNon-face trials (y=0): 648\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Train/test split\nX_train, X_test, y_train, y_test = train_test_split(X, y_face, test_size=0.2, random_state=42, stratify=y_face)\n\n#Regression\nmodel = LogisticRegression(max_iter=2000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n\n#Test accuracy + cross validation\naccuracy = accuracy_score(y_test, y_pred)\nprint(f\"Accuracy: {accuracy}\")\n\ncross_val = cross_val_score(model, X, y_face, cv=5)\nprint(f\"Cross-validation scores: {cross_val}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Accuracy: 0.9868421052631579\nCross-validation scores: [0.98684211 0.95364238 0.81456954 0.90066225 0.8807947 ]\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Sanity check\nprint(\"Total samples:\", len(y))\nprint(\"Face count:\", sum(y=='face'))\nprint(\"Binary positives:\", y_face.sum())",
   "outputs": [
    {
     "kind": "text",
     "text": "Total samples: 756\nFace count: 108\nBinary positives: 108\n"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Confusion Matrix\ncm = confusion_matrix(y_test, y_pred)\ndisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['not face', 'face'])\ndisp.plot(cmap = \"summer\")\nplt.title(\"Confusion Matrix Face vs. Non-Face\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-25-0.png",
     "width": 555,
     "height": 455
    }
   ]
  },
  {
   "type": "code",
   "source": "#K-NEAREST NEIGHBORS\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)",
   "outputs": []
  },
  {
   "type": "code",
   "source": "#Pick k\nk_values = range(1, 30)\nscores = []\nfor k in k_values:\n    model = KNeighborsClassifier(n_neighbors=k)\n    model.fit(X_train, y_train)\n    scores.append(model.score(X_test, y_test))\n\nbest_k = list(k_values)[np.argmax(scores)]\nprint(f\"Best k: {best_k}\")\n\n# Sweep\nplt.plot(list(k_values), scores, color = 'hotpink')\nplt.xlabel(\"k\")\nplt.ylabel(\"Accuracy\")\nplt.title(\"KNN Hyperparam. Sweep\")\nplt.show()",
   "outputs": [
    {
     "kind": "text",
     "text": "Best k: 1\n"
    },
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-27-1.png",
     "width": 578,
     "height": 455
    }
   ]
  },
  {
   "type": "code",
   "source": "#KNN vs. Logistic Regression\n\ncv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)\n\nlinear_scores = cross_val_score(LogisticRegression(max_iter = 2000), X, y_face, cv = cv, scoring = 'accuracy')\nKNN_scores = cross_val_score(KNeighborsClassifier(n_neighbors = 5), X_scaled, y_face, cv = cv, scoring = 'accuracy')\n\nprint(f\"Logistic Regression: {linear_scores.mean():.3f}\")\nprint(f\"KNN (k=5): {KNN_scores.mean():.3f}\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Logistic Regression: 0.991\nKNN (k=5): 0.983\n"
    }
   ]
  },
  {
   "type": "markdown",
   "source": "#***Unsupervised Task***\n---\n How do images sit in the brain's representational space?"
  },
  {
   "type": "code",
   "source": "#Setup PCA\npca = PCA(n_components = 2)\nX_pca = pca.fit_transform(X)\n\npca.explained_variance_ratio_",
   "outputs": [
    {
     "kind": "text",
     "text": "array([0.37907103, 0.13632541])"
    }
   ]
  },
  {
   "type": "code",
   "source": "#Plot PCA\nplt.figure(figsize = (8,6))\nsns.scatterplot(x = X_pca[:,0], y = X_pca[:,1], hue = y, palette = 'hsv')\nplt.title(\"PCA of fMRI responses\")\nplt.xlabel(\"PC1\")\nplt.ylabel(\"PC2\")\nplt.legend(title = \"Category\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-31-0.png",
     "width": 698,
     "height": 547
    }
   ]
  },
  {
   "type": "code",
   "source": "#t-SNE\ntsne = TSNE(n_components = 2, random_state = 42, max_iter = 1000)\nX_tsne = tsne.fit_transform(X)",
   "outputs": []
  },
  {
   "type": "code",
   "source": "#Plot t-SNE\nplt.figure(figsize = (8,6))\nsns.scatterplot(x = X_tsne[:,0], y = X_tsne[:,1], hue = y, palette = 'hsv')\nplt.title(\"t-SNE of fMRI responses\")\nplt.legend(title = \"Category\")\nplt.show()",
   "outputs": [
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-33-0.png",
     "width": 678,
     "height": 528
    }
   ]
  },
  {
   "type": "code",
   "source": "#Comparison\nfig, axes = plt.subplots(1, 2, figsize = (12,6))\nfig.suptitle(\"PCA and t-SNE of fMRI responses\")\n#PCA\nsns.scatterplot(x = X_pca[:,0], y = X_pca[:,1], hue = y, palette = 'hsv', ax = axes[0])\naxes[0].set_title(\"PCA\")\naxes[0].set_xlabel(\"PCA 1\")\naxes[0].set_ylabel(\"PCA 2\")\n#t-SNE\nsns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='hsv', ax=axes[1])\naxes[1].set_title(\"t-SNE\", fontsize=12)\naxes[1].set_xlabel(\"t-SNE 1\")\naxes[1].set_ylabel(\"t-SNE 2\")",
   "outputs": [
    {
     "kind": "text",
     "text": "Text(0, 0.5, 't-SNE 2')"
    },
    {
     "kind": "image",
     "src": "assets/img/dcs357-haxby/cell-34-1.png",
     "width": 1008,
     "height": 585
    }
   ]
  }
 ]
};
