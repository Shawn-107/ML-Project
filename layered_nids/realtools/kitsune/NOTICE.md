# Notice: vendored third-party code

Source: https://github.com/ymirsky/Kitsune-py (commit 7c8ea51), MIT License, Copyright (c) 2018 Yisroel Mirsky (see LICENSE).
Paper: Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018.
`kitnet/da.py` contains portions adapted from Yusuke Sugomori's DeepLearning (MIT, Copyright (c) 2017 Yusuke Sugomori);
its notice is kept in the file header. `kitnet/LICENSE.txt` is KitNET's own MIT notice (Copyright (c) 2017 Yisroel Mirsky).

Copied: AfterImage.py, netStat.py, KitNET/{KitNET,corClust,dA,utils}.py. Not copied: FeatureExtractor.py (reads files via
scapy/tshark and runs a shell command built from the file path), Kitsune.py, example.py, the Cython code, mirai.zip.
File names were lower-cased and imports made absolute; the only other edits are the lines below, applied by a script
that asserts each edit happened (NumPy 2 removed `np.Inf`; the Cython import hook and unused imports were dropped).
Everything else is byte-for-byte upstream. `tests/test_kitsune_vendored.py` checks that the vendored copy gives the same
features and scores as the original on the Mirai capture.

## Exact changes

### AfterImage.py -> afterimage.py
```diff
--- original/AfterImage.py
+++ vendored/afterimage.py
@@ -253 +253 @@
-    def __init__(self,limit=np.Inf,default_lambda=np.nan):
+    def __init__(self,limit=np.inf,default_lambda=np.nan):
```

### netStat.py -> netstat.py
```diff
--- original/netStat.py
+++ vendored/netstat.py
@@ -3,5 +3 @@
-import os
-import subprocess
-import pyximport
-pyximport.install()
-import AfterImage as af
+from layered_nids.realtools.kitsune import afterimage as af
```

### KitNET/KitNET.py -> kitnet/kitnet.py
```diff
--- original/KitNET/KitNET.py
+++ vendored/kitnet/kitnet.py
@@ -2,2 +2,2 @@
-import KitNET.dA as AE
-import KitNET.corClust as CC
+from layered_nids.realtools.kitsune.kitnet import da as AE
+from layered_nids.realtools.kitsune.kitnet import corclust as CC
```

### KitNET/corClust.py -> kitnet/corclust.py
(no changes besides the file name)

### KitNET/dA.py -> kitnet/da.py
```diff
--- original/KitNET/dA.py
+++ vendored/kitnet/da.py
@@ -28 +28 @@
-from KitNET.utils import *
+from layered_nids.realtools.kitsune.kitnet.utils import *
@@ -48,2 +48,2 @@
-        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf
-        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
+        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.inf
+        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.inf
```

### KitNET/utils.py -> kitnet/utils.py
(no changes besides the file name)
