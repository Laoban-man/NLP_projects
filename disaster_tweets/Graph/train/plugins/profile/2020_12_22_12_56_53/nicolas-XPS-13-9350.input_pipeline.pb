	���5>�?���5>�?!���5>�?	�����"@�����"@!�����"@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���5>�?e�`TR'�?A�o����?Y��T����?*	��n��f@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatBȗP��?!����K@)I���|�?1����OJ@:Preprocessing2U
Iterator::Model::ParallelMapV2��G���?!xa�a��.@)��G���?1xa�a��.@:Preprocessing2F
Iterator::ModelgaO;�5�?!�7�z�;@)��.Q��?1���\'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(�r�w�?!�9�g]+@)�(���ǒ?1�>��N.$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���'�z?!��7{`�@)���'�z?1��7{`�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3R臭��?!�W!A:R@)5��-</u?1���i�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���`�Ht?!�5����@)���`�Ht?1�5����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����c�?!�M,0fn-@)*�=%��^?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t13.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�����"@I�C�$ݡV@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e�`TR'�?e�`TR'�?!e�`TR'�?      ��!       "      ��!       *      ��!       2	�o����?�o����?!�o����?:      ��!       B      ��!       J	��T����?��T����?!��T����?R      ��!       Z	��T����?��T����?!��T����?b      ��!       JCPU_ONLYY�����"@b q�C�$ݡV@