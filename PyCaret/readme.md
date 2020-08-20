## What is PyCaret?
PyCaret is an open source `low-code` machine learning library in Python that aims to reduce the hypothesis to insights cycle time in a ML experiment. It enables data scientists to perform end-to-end experiments quickly and efficiently. In comparison with the other open source machine learning libraries, PyCaret is an alternative low-code library that can be used to perform complex machine learning tasks with only few lines of code. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as `scikit-learn`, `XGBoost`, `Microsoft LightGBM`, `spaCy` and many more. 

The design and simplicity of PyCaret is inspired by the emerging role of `citizen data scientists`, a term first used by Gartner. Citizen Data Scientists are `power users` who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data related challenges in business setting. 

PyCaret is a great library which not only simplifies the machine learning tasks for citizen data scientists but also helps new startups to reduce the cost of investing in a team of data scientists. Therefore, this library has not only helped the citizen data scientists but has also helped individuals who want to start exploring the field of data science, having no prior knowledge in this field.

PyCaret is `simple`, `easy to use` and `deployment ready`. All the steps performed in a ML experiment can be reproduced using a pipeline that is automatically developed and orchestrated in PyCaret as you progress through the experiment. A `pipeline` can be saved in a binary file format that is transferable across environments.

For more information on PyCaret, please visit our official website https://www.pycaret.org

![alt text](https://github.com/pycaret/pycaret/blob/master/pycaret2-features.png)

## Current Release
PyCaret `2.0` is now available. See `2.0` release notes. The easiest way to install pycaret is using pip. 

```python
pip install pycaret==2.0
```

## Optional dependencies
Following libraries are not hard dependencies and are not automatically installed when you install PyCaret. To use all functionalities of PyCaret, these optional dependencies must be installed.

```shell
pip install psutil
pip install awscli 
pip install azure-storage-blob
pip install google-cloud-storage
pip install shap
```

## License

Copyright 2019-2020 Moez Ali <moez.ali@queensu.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2020 GitHub, Inc.
