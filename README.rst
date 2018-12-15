
Sklearn-pandas
==============

原项目地址：https://github.com/scikit-learn-contrib/sklearn-pandas

本文翻译： https://github.com/cycleuser

正文
=============

这个模块的扮演了桥梁的角色，联通了 `Scikit-Learn <http://scikit-learn.org/stable>` 中的机器学习方法和 `pandas <https://pandas.pydata.org>` 的数据结构DataFrame。

具体来说，此模块提供了下面的功能:

1. 映射 ``DataFrame`` 的列（columns）到SKlearn里面的变换（transformations）, 后续就可以重新组合成特征（features）。
2. 使老版本的SKlearn能兼容使用pandas ``DataFrame``作为输入（input）的pipeline来进行交叉验证（cross-validate）。这种兼容仅是针对 ``scikit-learn<0.16.0`` (具体参考 `#11 <https://github.com/paulgb/sklearn-pandas/issues/11>`)。目前已经不被支持（deprecated），在未来的版本``skearn-pandas==2.0``中可能就去掉了。
3. 还提供了一些适于处理pandas输入（inputs）的特殊转换（special transformers）: ``CategoricalImputer``和``FunctionTransformer``

安装
------------

使用``pip``就可以安装 ``sklearn-pandas``::

    # pip install sklearn-pandas

测试
-----


本文的例子中就有一些基本的测试，可以使用``doctest`来运行，如下所示::

    # python -m doctest README.rst

使用
-----

导入
******

首先自然是要从``sklearn_pandas``导入你需要的包，可以选择下面的:

* ``DataFrameMapper``, 这个类是用于映射pandas DataFrame 的雷（columns）到不同的 SKlearn 变换（transformations）
* ``cross_val_score``, 类似于``sklearn.cross_validation.cross_val_score``，不同是处理的是 pandas DataFrames

这里为了演示，两个都导入了::

    >>> from sklearn_pandas import DataFrameMapper, cross_val_score

在本文的例子中，还要导入用到的 pandas, numpy, sklearn::

    >>> import pandas as pd
    >>> import numpy as np
    >>> import sklearn.preprocessing, sklearn.decomposition, \
    ...     sklearn.linear_model, sklearn.pipeline, sklearn.metrics
    >>> from sklearn.feature_extraction.text import CountVectorizer

载入数据
**************


一般来说你都是从文件载入读取数据的吧，不过这里为了演示目的就直接用Python的dict创建一个pandas 的DataFrame了::

    >>> data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
    ...                      'children': [4., 6, 3, 3, 2, 3, 5, 4],
    ...                      'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})

变换映射（Transformation Mapping）
--------------------------------------

映射列（Columns）到变换（Transformations）
******************************************

映射器（mapper）接收的是一个元组列表（a list of tuples）。每个元组的第一个元素是pandas DataFrame 里面的列名（column name），或者是一个包含了一列或多列的列表（多列的例子后面会讲到）。第二个元素是要对这个列进行变换的一个对象（object）。第三个是可选的，如果可用的话就是一个字典（dict），包含了变换选项（transformation options），具体可以参考后文中的"为变换特征（ transformed features）设定列名（custom column names）"这部分。

看一个例子::

    >>> mapper = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     (['children'], sklearn.preprocessing.StandardScaler())
    ... ])


这里的列选择器可以指定为 ``'column'`` (一个简单字符串string) 或者 ``['column']`` (单元素列表list)，二者的区别就是传递到转换器（transformer）的数组形状。前者传递的是一个一维数组（1-dimensional array）；而后者则传递了一个单列（one column）的二维数组（2-dimensional array），也就是列向量（column vector）。


上述行为是模仿了pandas 中 DataFrame 的``__getitem__``索引（indexin）的模式:


    >>> data['children'].shape
    (8,)
    >>> data[['children']].shape
    (8, 1)

要注意有的变换器（transformers）要求一维输入（比如面向标签的the label-oriented ones），而另外一些比如 ``OneHotEncoder`` 或者 ``Imputer``则要求二维输入，形状为``[n_samples, n_features]``。


测试变换（Test the Transformation）
**************************************

使用``fit_transform``既可以拟合模型，也可以查看变换后的数据是啥样。在本文的这些例子中，使用了``np.round``将输出四舍五入到小数点后两位，考虑到了不同硬件平台的舍入误差::

    >>> np.round(mapper.fit_transform(data.copy()), 2)
    array([[ 1.  ,  0.  ,  0.  ,  0.21],
           [ 0.  ,  1.  ,  0.  ,  1.88],
           [ 0.  ,  1.  ,  0.  , -0.63],
           [ 0.  ,  0.  ,  1.  , -0.63],
           [ 1.  ,  0.  ,  0.  , -1.46],
           [ 0.  ,  1.  ,  0.  , -0.63],
           [ 1.  ,  0.  ,  0.  ,  1.04],
           [ 0.  ,  0.  ,  1.  ,  0.21]])


注意前面三列是``LabelBinarizer``的输出（对应的分别是 ``cat``, ``dog``, ``fish`` ），第四列是子数目的标准化值（standardized value for the number of children）。一般来说，这些列的排序是对应着``DataFrameMapper``构建的时候给出的顺序。

接下来就要训练这个变换了，要确定能够用于新数据::

    >>> sample = pd.DataFrame({'pet': ['cat'], 'children': [5.]})
    >>> np.round(mapper.transform(sample), 2)
    array([[1.  , 0.  , 0.  , 1.04]])


输出特征名（Output features names）
************************************

在具体案例中，比如学习某些模型的特征重要性（feature importances），我们想要能将原始特征和dataframe映射器生成的特征连接起来。在变换之后，通过检查映射器（mapper）自动生成的``transformed_names_``属性（attribute）就可以实现::

    >>> mapper.transformed_names_
    ['pet_cat', 'pet_dog', 'pet_fish', 'children']


为变换特征设定列名（Custom column names for transformed features）
**************************************************************************

除了使用自动生成的列名，我们还可以对变换后的特征提供一系列设定的名字，只要在特征定义的时候将其作为第三个参数（argument）即可::

  >>> mapper_alias = DataFrameMapper([
  ...     (['children'], sklearn.preprocessing.StandardScaler(),
  ...      {'alias': 'children_scaled'})
  ... ])
  >>> _ = mapper_alias.fit_transform(data.copy())
  >>> mapper_alias.transformed_names_
  ['children_scaled']


传递 Series/DataFrames 给变换器（transformers）
************************************************************

默认情况下变换器要求传递的是一个numpy的数组，由选中的列组成，作为输入。这是因为``sklearn``的变换器（transformers）在其发展早期就是被设计用来处理numpy数组的，而不是pandas的DataFrame，不过这两者的基本索引界面倒是很相似。

不过我们可以通过使用``input_df=True``来初始化 DataFrame 映射器（mapper），然后就可以传递Series/DataFrames给变换器（transformers）::


    >>> from sklearn.base import TransformerMixin
    >>> class DateEncoder(TransformerMixin):
    ...    def fit(self, X, y=None):
    ...        return self
    ...
    ...    def transform(self, X):
    ...        dt = X.dt
    ...        return pd.concat([dt.year, dt.month, dt.day], axis=1)
    >>> dates_df = pd.DataFrame(
    ...     {'dates': pd.date_range('2015-10-30', '2015-11-02')})
    >>> mapper_dates = DataFrameMapper([
    ...     ('dates', DateEncoder())
    ... ], input_df=True)
    >>> mapper_dates.fit_transform(dates_df)
    array([[2015,   10,   30],
           [2015,   10,   31],
           [2015,   11,    1],
           [2015,   11,    2]])


上述方法是针对整个映射器（mapper）进行的，还可以针对具体的每一组列来进行这样的设定::

  >>> mapper_dates = DataFrameMapper([
  ...     ('dates', DateEncoder(), {'input_df': True})
  ... ])
  >>> mapper_dates.fit_transform(dates_df)
  array([[2015,   10,   30],
         [2015,   10,   31],
         [2015,   11,    1],
         [2015,   11,    2]])

输出一个 DataFrame
**********************

DataFrame映射器（mapper）的默认输出是numpy数组。这是因为大多数SKlearn的估计器（estimator）都接收numpy数组作为输入。如果我们想让映射器输出一个DataFrame，可以在创建映射器的时候增加参数``df_out``来实现::

    >>> mapper_df = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     (['children'], sklearn.preprocessing.StandardScaler())
    ... ], df_out=True)
    >>> np.round(mapper_df.fit_transform(data.copy()), 2)
       pet_cat  pet_dog  pet_fish  children
    0        1        0         0      0.21
    1        0        1         0      1.88
    2        0        1         0     -0.63
    3        0        0         1     -0.63
    4        1        0         0     -1.46
    5        0        1         0     -0.63
    6        1        0         0      1.04
    7        0        0         1      0.21


列名就和 ``transformed_names`` 属性中的一样。

要注意，上述方法不适于设定了 ``default=True`` 或者 ``sparse=True`` 参数的映射器。

变换多列（Transform Multiple Columns）
*****************************************

有的变换（Transformations）可能需要多个输入列。这时候这些列就可以用一个列表来指定::


    >>> mapper2 = DataFrameMapper([
    ...     (['children', 'salary'], sklearn.decomposition.PCA(1))
    ... ])


这时候运行 ``fit_transform`` 就会在 ``children`` 和 ``salary`` 这两列上运行主成分分析（PCA），然后返回的就是第一主要成分（first principal component）::

    >>> np.round(mapper2.fit_transform(data.copy()), 1)
    array([[ 47.6],
           [-18.4],
           [  1.6],
           [-15.4],
           [-10.4],
           [ 16.6],
           [ -6.4],
           [-15.4]])

单列的多变换（Multiple transformers for the same column）
*****************************************************************


用于单列的多个变换（transformaer）也可以用一个列表来指定::

    >>> mapper3 = DataFrameMapper([
    ...     (['age'], [sklearn.preprocessing.Imputer(),
    ...                sklearn.preprocessing.StandardScaler()])])
    >>> data_3 = pd.DataFrame({'age': [1, np.nan, 3]})
    >>> mapper3.fit_transform(data_3)
    array([[-1.22474487],
           [ 0.        ],
           [ 1.22474487]])



无需变换的列
******************************************


只有在 DataFrameMapper 中列出的列会保存。要保存一个列又不对其进行任何变换，可以使用`None` 所谓变换器（transformer）::

    >>> mapper3 = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     ('children', None)
    ... ])
    >>> np.round(mapper3.fit_transform(data.copy()))
    array([[1., 0., 0., 4.],
           [0., 1., 0., 6.],
           [0., 1., 0., 3.],
           [0., 0., 1., 3.],
           [1., 0., 0., 2.],
           [0., 1., 0., 3.],
           [1., 0., 0., 5.],
           [0., 0., 1., 4.]])


使用默认变换器（default transformer）
*********************************************

默认变换器可以用于没有明确选择的列，只要带着``default``参数传递到映射器（mapper）即可::

    >>> mapper4 = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     ('children', None)
    ... ], default=sklearn.preprocessing.StandardScaler())
    >>> np.round(mapper4.fit_transform(data.copy()), 1)
    array([[ 1. ,  0. ,  0. ,  4. ,  2.3],
           [ 0. ,  1. ,  0. ,  6. , -0.9],
           [ 0. ,  1. ,  0. ,  3. ,  0.1],
           [ 0. ,  0. ,  1. ,  3. , -0.7],
           [ 1. ,  0. ,  0. ,  2. , -0.5],
           [ 0. ,  1. ,  0. ,  3. ,  0.8],
           [ 1. ,  0. ,  0. ,  5. , -0.3],
           [ 0. ,  0. ,  1. ,  4. , -0.7]])


默认设置是``default=False``，这时候会去掉未选择的列。如果设置``default=None``就会将未选择的列不进行任何变化保存下来。


对多列的同变换（Same transformer for the multiple columns）
***********************************************************************

有时候需要对几个不同的DataFrame的列应用同样的变换。要简化这个过程，我们可以使用``gen_features``函数，这个函数接受一个列（columns）的列表和特征变换类（或者类列表），然后生成一个特征定义，可以被``DataFrameMapper`接收。

举个例子，设想某个数据集有三个分类列：'col1', 'col2', 'col3'。要对每个都进行二值化（binarize），可以传递列名称和``LabelBinarizer`` 变换类到生成器（generator），然后使用返回的定义作为用于 ``DataFrameMapper``的``features`` 参数::

    >>> from sklearn_pandas import gen_features
    >>> feature_def = gen_features(
    ...     columns=['col1', 'col2', 'col3'],
    ...     classes=[sklearn.preprocessing.LabelEncoder]
    ... )
    >>> feature_def
    [('col1', [LabelEncoder()]), ('col2', [LabelEncoder()]), ('col3', [LabelEncoder()])]
    >>> mapper5 = DataFrameMapper(feature_def)
    >>> data5 = pd.DataFrame({
    ...     'col1': ['yes', 'no', 'yes'],
    ...     'col2': [True, False, False],
    ...     'col3': ['one', 'two', 'three']
    ... })
    >>> mapper5.fit_transform(data5)
    array([[1, 1, 0],
           [0, 0, 2],
           [1, 0, 1]])


如果需要覆盖某些变换参数，就需要用一个字典，包含有'class' 键值（key）和变换器参数。例如处理一个有缺失数据值的数据集就会如此。然后接下来的代码可以用来覆盖默认归因策略（imputing strategy）::

    >>> feature_def = gen_features(
    ...     columns=[['col1'], ['col2'], ['col3']],
    ...     classes=[{'class': sklearn.preprocessing.Imputer, 'strategy': 'most_frequent'}]
    ... )
    >>> mapper6 = DataFrameMapper(feature_def)
    >>> data6 = pd.DataFrame({
    ...     'col1': [None, 1, 1, 2, 3],
    ...     'col2': [True, False, None, None, True],
    ...     'col3': [0, 0, 0, None, None]
    ... })
    >>> mapper6.fit_transform(data6)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [1., 1., 0.],
           [2., 1., 0.],
           [3., 1., 0.]])


特征选择和其他监督变换 
******************************************************

``DataFrameMapper`` 支持同时要求X和y参数的变换器。例如特征选择。将'pet'这一列作为目标，就可以选择能进行最佳预测的列。

    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> mapper_fs = DataFrameMapper([(['children','salary'], SelectKBest(chi2, k=1))])
    >>> mapper_fs.fit_transform(data[['children','salary']], data['pet'])
    array([[90.],
           [24.],
           [44.],
           [27.],
           [32.],
           [59.],
           [36.],
           [27.]])

处理稀疏特征（sparse features）
*******************************************

默认情况下``DataFrameMapper``会返回一个密集特征数组（dense feature array）。在映射器（mapper）中设置``sparse=True``则会返回一个稀疏数组，无论提取的特征是否稀疏。例如::

    >>> mapper5 = DataFrameMapper([
    ...     ('pet', CountVectorizer()),
    ... ], sparse=True)
    >>> type(mapper5.fit_transform(data))
    <class 'scipy.sparse.csr.csr_matrix'>

这些稀疏特征（sparse features）的叠加（stacking）是在未致密化（densifying）的情况下实现的。


交叉验证（Cross-Validation）
*******************************


通过上面的示范，现在咱们就可以将pandas DataFrame 的特征结合起来了，可以使用交叉验证来检测咱们的模型是否正常工作。``scikit-learn<0.16.0`` 提供了交叉验证的功能，但只接收numpy数据结构体，不能使用``DataFrameMapper``。

为了解决这个问题，sklearn-pandas 对SKlearn的``cross_val_score``函数进行了打包，传递一个pandas DataFrame过去而不用传递numpy数组::

    >>> pipe = sklearn.pipeline.Pipeline([
    ...     ('featurize', mapper),
    ...     ('lm', sklearn.linear_model.LinearRegression())])
    >>> np.round(cross_val_score(pipe, X=data.copy(), y=data.salary, scoring='r2'), 2)
    array([ -1.09,  -5.3 , -15.38])

Sklearn-pandas的 ``cross_val_score`` 函数提供的界面和SKlearn里面的同名函数完全相同。

``CategoricalImputer``
**********************


由于目前（2018年12月15日） ``scikit-learn``  ``Imputer`` 的变换器（transformer）都只能处理数值，``sklearn-pandas`` 提供了一个等效的辅助变换器（equivalent helper transformer），能处理字符串和用列中最频繁的值来替代空值。或者你也可以指定使用某一个固定值。

例子：使用众数:

    >>> from sklearn_pandas import CategoricalImputer
    >>> data = np.array(['a', 'b', 'b', np.nan], dtype=object)
    >>> imputer = CategoricalImputer()
    >>> imputer.fit_transform(data)
    array(['a', 'b', 'b', 'b'], dtype=object)

例子：使用某值:

    >>> from sklearn_pandas import CategoricalImputer
    >>> data = np.array(['a', 'b', 'b', np.nan], dtype=object)
    >>> imputer = CategoricalImputer(strategy='constant', fill_value='a')
    >>> imputer.fit_transform(data)
    array(['a', 'b', 'b', 'a'], dtype=object)


函数变换器 ``FunctionTransformer``
**************************************


有时候可能需要对数据进行简单变换，比如取对数，要用到 ``np.log`` 。 ``FunctionTransformer`` 是一个简单的打包可以接收任意函数，然后进行向量化（applies vectorization），使其可以被用作变换器（transformer）

样例:

    >>> from sklearn_pandas import FunctionTransformer
    >>> array = np.array([10, 100])
    >>> transformer = FunctionTransformer(np.log10)

    >>> transformer.fit_transform(array)
    array([1., 2.])

更新记录（不翻译了）
---------------------------

1.8.0 (2018-12-01)
******************
* Add ``FunctionTransformer`` class (#117).
* Fix column names derivation for dataframes with multi-index or non-string
  columns (#166).
* Change behaviour of DataFrameMapper's fit_transform method to invoke each underlying transformers'
  native fit_transform if implemented. (#150)

1.7.0 (2018-08-15)
******************
* Fix issues with unicode names in ``get_names`` (#160).
* Update to build using ``numpy==1.14`` and ``python==3.6`` (#154).
* Add ``strategy`` and ``fill_value`` parameters to ``CategoricalImputer`` to allow imputing
  with values other than the mode (#144), (#161).
* Preserve input data types when no transform is supplied (#138).

1.6.0 (2017-10-28)
******************
* Add column name to exception during fit/transform (#110).
* Add ``gen_feature`` helper function to help generating the same transformation for multiple columns (#126).


1.5.0 (2017-06-24)
******************
* Allow inputting a dataframe/series per group of columns.
* Get feature names also from ``estimator.get_feature_names()`` if present.
* Attempt to derive feature names from individual transformers when applying a
  list of transformers.
* Do not mutate features in ``__init__`` to be compatible with
  ``sklearn>=0.20`` (#76).


1.4.0 (2017-05-13)
******************
* Allow specifying a custom name (alias) for transformed columns (#83).
* Capture output columns generated names in ``transformed_names_`` attribute (#78).
* Add ``CategoricalImputer`` that replaces null-like values with the mode
  for string-like columns.
* Add ``input_df`` init argument to allow inputting a dataframe/series to the
  transformers instead of a numpy array (#60).


1.3.0 (2017-01-21)
******************

* Make the mapper return dataframes when ``df_out=True`` (#70, #74).
* Update imports to avoid deprecation warnings in sklearn 0.18 (#68).


1.2.0 (2016-10-02)
******************

* Deprecate custom cross-validation shim classes.
* Require ``scikit-learn>=0.15.0``. Resolves #49.
* Allow applying a default transformer to columns not selected explicitly in
  the mapper. Resolves #55.
* Allow specifying an optional ``y`` argument during transform for
  supervised transformations. Resolves #58.


1.1.0 (2015-12-06)
*******************

* Delete obsolete ``PassThroughTransformer``. If no transformation is desired for a given column, use ``None`` as transformer.
* Factor out code in several modules, to avoid having everything in ``__init__.py``.
* Use custom ``TransformerPipeline`` class to allow transformation steps accepting only a X argument. Fixes #46.
* Add compatibility shim for unpickling mappers with list of transformers created before 1.0.0. Fixes #45.


1.0.0 (2015-11-28)
*******************

* Change version numbering scheme to SemVer.
* Use ``sklearn.pipeline.Pipeline`` instead of copying its code. Resolves #43.
* Raise ``KeyError`` when selecting unexistent columns in the dataframe. Fixes #30.
* Return sparse feature array if any of the features is sparse and ``sparse`` argument is ``True``. Defaults to ``False`` to avoid potential breaking of existing code. Resolves #34.
* Return model and prediction in custom CV classes. Fixes #27.


0.0.12 (2015-11-07)
********************

* Allow specifying a list of transformers to use sequentially on the same column.


Credits
-------

The code for ``DataFrameMapper`` is based on code originally written by `Ben Hamner <https://github.com/benhamner>`__.

Other contributors:

* Ariel Rossanigo (@arielrossanigo)
* Arnau Gil Amat (@arnau126)
* Assaf Ben-David (@AssafBenDavid)
* Brendan Herger (@bjherger)
* Cal Paterson (@calpaterson)
* @defvorfu
* Gustavo Sena Mafra (@gsmafra)
* Israel Saeta Pérez (@dukebody)
* Jeremy Howard (@jph00)
* Jimmy Wan (@jimmywan)
* Kristof Van Engeland (@kristofve91)
* Olivier Grisel (@ogrisel)
* Paul Butler (@paulgb)
* Richard Miller (@rwjmiller)
* Ritesh Agrawal (@ragrawal)
* @SandroCasagrande
* Timothy Sweetser (@hacktuarial)
* Vitaley Zaretskey (@vzaretsk)
* Zac Stewart (@zacstewart)
