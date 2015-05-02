

### DataParallel.lua ###

Copyright 2004-present Facebook. All Rights Reserved.
This file defines an example class

<a name="fbcunn.DataParallel.dok"></a>


## fbcunn.DataParallel ##


DataParallel splits the input along separate columns, that run the
same models on distinct partitions of the input.

Pictorially
```
                        +--------+
        column 1        |        |         column 3
           +------------+  Input +-------------+
           |            |        |             |
           |            +----+---+             |
           |                 |                 |
           |                 |                 |
      +----+---+        +----+---+        +----+---+
      |        |        |        |        |        |
      | Linear |        | Linear |        | Linear |       row 1
      |        |        |        |        |        |
      +----+---+        +----+---+        +----+---+
           |                 |                 |
           |                 |                 |
      +----+---+        +----+---+        +----+---+
      |        |        |        |        |        |
      |  Tanh  |        |  Tanh  |        |  Tanh  |       row 2
      |        |        |        |        |        |
      +----+---+        +----+---+        +----+---+
           |                 |                 |
           |                 |                 |
           |                 |                 |
           |            +----+---+             |
           |            |        |             |
           +------------+ Output +-------------+
                        |        |
                        +--------+
```



#### Undocumented methods ####

<a name="fbcunn.DataParallel:name"></a>
 * `fbcunn.DataParallel:name()`
<a name="fbcunn.DataParallel:updateGradInput"></a>
 * `fbcunn.DataParallel:updateGradInput(_input, gradOutput)`
<a name="fbcunn.DataParallel:accUpdateGradParameters"></a>
 * `fbcunn.DataParallel:accUpdateGradParameters(_input, _gradOutput, lr)`
