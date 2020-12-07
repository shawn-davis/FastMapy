# FastMapy

Python implementation of FastMap<sup id="a1">[1](#f1)</sup> MDS technique for embedding objects into vector spaces and 
dimensionality reduction of existing vector spaces. The general idea is that the objects are embedding into a vector
space based on a defined distance metric over the objects. The resulting vector space attempts to maintain this relative
distance between the objects relative to the defined distance metric.

This package has common distance metrics already defined and ready to use over appropriate objects, such as
Jaccard distance over character shingled _n_-gram strings or Levenshtein edit distance for embedding string objects.
Euclidean distance and taxi cab distance are also available for vector objects. Dictionary objects also work assuming a
sparse vector style dictionary of _{index: count}_ where index can be an actual vector index or a token and it's
occurrence count.

Multiprocessing is leveraged for model building and object transformation, but is set to serially use a single core by
default.

# Example

```
from fastmap.distances import Jaccard
import fastmap

fm_model = fastmap.FastMap(dim=8, distance=Jaccard, dist_args={'shingle_size':4})

embedding = fm_model.fit_transform(string_data)
```
The above example defines a FastMap model that utilizes Jaccard distance. The target vector space is _8_-dimensional and
strings are shingled into _4_-grams before the distance is computed. A collection of strings are then used to fit the
model and the same strings are transformed into _8_-dimensional Numpy arrays.

#References
<b id="f1">1</b> Proceedings of the 1995 ACM SIGMOD international conference on Management of data  - SIGMOD  ’95. (1995). doi:10.1145/223784 [↩](#a1)