# image-data-module
This repository hosts the teaching materials for processing image data in the [UvA Data Science course](https://multix.io/data-science-book-uva/).

All the content in this repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Credit: this teaching material is created by Bryan Fleming under the supervision of [Yen-Chia Hsu](https://github.com/yenchiah).

## Potential Error

Notice that the notebooks in this repository are created on Google Colab.
When copying the notebook created by Colab, we have to clear widget state to prevent the following error when building the JupyterBook.
```sh
Exception occurred:
  File "/opt/homebrew/Caskroom/miniconda/base/envs/jupyterbook/lib/python3.10/site-packages/jupyter_sphinx/execute.py", line 300, in contains_widgets
    return widgets and widgets["state"]
KeyError: 'state'
```

To fix this problem, use the solution in [this link](https://github.com/jupyter-widgets/ipywidgets/issues/2867#issuecomment-625418996).
Open the downloaded notebook on JupyterLab and follow the steps below:
- Make sure "Save Widget State Automatically" is checked under "Settings" in the menu bar
- Restart the kernel (no need to clean output)
- Save the notebook
