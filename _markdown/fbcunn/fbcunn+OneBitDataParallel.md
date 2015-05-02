<a name="fbcunn.OneBitDataParallel.dok"></a>


## fbcunn.OneBitDataParallel ##

 OneBitDataParallel implements the "1-Bit Stochastic Gradient
Descent and Application to Data-Parallel Distributed Training of
Speech DNNs" paper of Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and
Dong Yu.

The implementation is similar to a vanilla DataParallel module, except we replace the averaging gradient step with a quantize-copy-merge-broadcast procedure.

<http://research.microsoft.com/apps/pubs/?id=230137>



#### Undocumented methods ####

<a name="fbcunn.OneBitDataParallel"></a>
 * `fbcunn.OneBitDataParallel(dimension, config)`
