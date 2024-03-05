# image-data-module
This repository hosts the teaching materials for processing image data in the [UvA Data Science course](https://multix.io/data-science-book-uva/).

All the content in this repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Credit: this teaching material is created by Bryan Fleming under the supervision of [Yen-Chia Hsu](https://github.com/yenchiah).

## <a name="coding-standards"></a>Coding standards
When contributing code to this repository, please follow the guidelines below:

### Language
- The primary language for this repository is set to English. Please use English when writing comments and docstrings in the code. Please also use English when writing git issues, pull requests, wiki pages, commit messages, and the README file.

### Git workflow
- Follow the [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow). The master branch preserves the development history with no broken code. When working on a system feature, create a separate feature branch.
- Always create a pull request before merging the feature branch into the main branch. Doing so helps keep track of the project history and manage git issues.
- NEVER perform git rebasing on public branches, which means that you should not run "git rebase [FEATURE-BRANCH]" while you are on a public branch (e.g., the main branch). Doing so will badly confuse other developers since rebasing rewrites the git history, and other people's works may be based on the public branch. Check [this tutorial](https://www.atlassian.com/git/tutorials/merging-vs-rebasing#the-golden-rule-of-rebasing) for details.
- NEVER push credentials to the repository, for example, database passwords or private keys for signing digital signatures (e.g., the user tokens).
- Request a code review when you are not sure if the feature branch can be safely merged into the main branch.

### Coding style
- Use the functional programming style (check [this Python document](https://docs.python.org/3/howto/functional.html) for the concept). It means that each function is self-contained and does NOT depend on a state that may change outside the function (e.g., global variables). Avoid using the object-oriented programming style unless necessary. In this way, we can accelerate the development progress while maintaining code reusability.
- Minimize the usage of global variables, unless necessary, such as system configuration variables. For each function, avoid modifying its input parameters. In this way, each function can be independent, which is good for debugging code and assigning coding tasks to a specific collaborator.
- Use a consistent coding style.
  - For Python, follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/), for example, putting two blank lines between functions, using the lower_snake_case naming convention for variable and function names. Please use double quote (not single quote) for strings.
- Document functions and script files using docstrings.
  - For Python, follow the [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html). Here is an [example](https://numpydoc.readthedocs.io/en/latest/example.html#example). More detailed numpydoc style can be found on [LSST's docstrings guide](https://developer.lsst.io/python/numpydoc.html).
- Always comment the code, which helps others read the code and reduce our pain in the future when debugging or adding new features.

## Build this book

Below are the steps to update and build this book. First, clone this repository to your local machine. Assume that you already have [miniconda](https://docs.conda.io/en/main/miniconda.html) installed. Next, install the jupyter-book package:
```sh
$ conda create -n jupyterbook
$ conda activate jupyterbook
$ conda install python
$ pip install -U jupyter-book
$ pip install -U ghp-import
$ jupyter-book --help
```
Then, clone this repository and build the book:
```sh
$ git clone https://github.com/MultiX-Amsterdam/image-data-module
$ cd image-data-module
$ jupyter-book build .
```
To rebuild the entire book, use the following:
```sh
$ jupyter-book build --all .
```
Once it is done, you can view the book in the [html content](_build/html) folder using a web browser. To update the book online in this GitHub repository (in the gh-pages branch), run the following:
```sh
$ ghp-import -n -p -f _build/html
```
The above steps will update the gh-pages branch, which hosts the website. Finally, follow the normal git flow to commit the changes and push the code to the main branch in this repository.

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
