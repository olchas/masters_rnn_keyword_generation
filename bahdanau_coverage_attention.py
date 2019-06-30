from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

class BahdanauCoverageAttention(BahdanauAttention):
  """
  Bahdanau attention with added coverage
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="BahdanauCoverageAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    super(BahdanauCoverageAttention, self).__init__(
        num_units=num_units,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        normalize=False,
        probability_fn=probability_fn,
        score_mask_value=score_mask_value,
        dtype=dtype,
        name=name)
    self._coverage_layer = layers_core.Dense(num_units, name="coverage_layer", use_bias=False, dtype=dtype)

  @property
  def coverage_layer(self):
    return self._coverage_layer

  def __call__(self, query, state, previous_alignment_sum):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
      previous_alignment_sum: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """

    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      coverage_vector = self.coverage_layer(previous_alignment_sum)
      score = _bahdanau_coverage_score(processed_query, self._keys, coverage_vector)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def _bahdanau_coverage_score(processed_query, keys, coverage_vector):
  """Implements Bahdanau-style (additive) scoring function with added coverage.

  Bhandanau attention as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    coverage_vector: Tensor, shape `[batch_size, num_units]`.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  coverage_vector = array_ops.expand_dims(coverage_vector, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)

  return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + coverage_vector), [2])


class CustomAttentionWrapperState(
    collections.namedtuple("CustomAttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "attention_state", "alignment_sum"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
    - `attention_state`: A single or tuple of nested objects
       containing attention mechanism state for each attention mechanism.
       The objects may contain Tensors or TensorArrays.
    - `alignment_sum`: A single or tuple of `Tensor`(s) containing the sum of alignments
       from the previous time steps for each attention mechanism.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    The new state fields' shape must match original state fields' shape. This
    will be validated, and original fields' shape will be propagated to new
    fields.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `CustomAttentionWrapperState`.

    Returns:
      A new `CustomAttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""
      if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
        return tensor_util.with_same_shape(old, new)
      return new

    return nest.map_structure(
        with_same_shape,
        self,
        super(CustomAttentionWrapperState, self)._replace(**kwargs))


def _custom_compute_attention(attention_mechanism, cell_output, attention_state,
                              attention_layer, previous_alignment_sum):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state, previous_alignment_sum=previous_alignment_sum)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments, next_attention_state


class CustomAttentionWrapper(AttentionWrapper):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None):
    """Construct the `AttentionWrapper`.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: A list of `AttentionMechanism` instances or a single
        instance.
      attention_layer_size: A list of Python integers or a single Python
        integer, the depth of the attention (output) layer(s). If None
        (default), use the context as attention at each time step. Otherwise,
        feed the context and cell output into the attention layer to generate
        attention at each time step. If attention_mechanism is a list,
        attention_layer_size must be a list of the same length.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      name: Name to use when creating ops.

    Raises:
      TypeError: `attention_layer_size` is not None and (`attention_mechanism`
        is a list but `attention_layer_size` is not; or vice versa).
      ValueError: if `attention_layer_size` is not None, `attention_mechanism`
        is a list, and its length does not match that of `attention_layer_size`.
    """
    super(CustomAttentionWrapper, self).__init__(cell=cell,
                                                 attention_mechanism=attention_mechanism,
                                                 attention_layer_size=attention_layer_size,
                                                 alignment_history=alignment_history,
                                                 cell_input_fn=cell_input_fn,
                                                 output_attention=output_attention,
                                                 initial_cell_state=initial_cell_state,
                                                 name=name)

  @property
  def state_size(self):
    """The `state_size` property of `AttentionWrapper`.

    Returns:
      An `CustomAttentionWrapperState` tuple containing shapes used by this object.
    """
    return CustomAttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._item_or_tuple(
            a.alignments_size for a in self._attention_mechanisms),
        attention_state=self._item_or_tuple(
            a.state_size for a in self._attention_mechanisms),
        alignment_history=self._item_or_tuple(
            () for _ in self._attention_mechanisms),
        alignment_sum=self._item_or_tuple(
            a.alignments_size for a in self._attention_mechanisms))  # sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `AttentionWrapper`.

    **NOTE** Please see the initializer documentation for details of how
    to call `zero_state` if using an `AttentionWrapper` with a
    `BeamSearchDecoder`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.

    Returns:
      An `CustomAttentionWrapperState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.

    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
        `batch_size` does not match the output size of the encoder passed
        to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      return CustomAttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._item_or_tuple(
              attention_mechanism.initial_alignments(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms),
          attention_state=self._item_or_tuple(
              attention_mechanism.initial_state(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms),
          alignment_history=self._item_or_tuple(
              tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                           dynamic_size=True)
              if self._alignment_history else ()
              for _ in self._attention_mechanisms),
          alignment_sum=self._item_or_tuple(
              attention_mechanism.initial_alignments(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms))

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `CustomAttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `CustomAttentionWrapperState`
         containing the state calculated at this time step.

    Raises:
      TypeError: If `state` is not an instance of `CustomAttentionWrapperState`.
    """
    if not isinstance(state, CustomAttentionWrapperState):
      raise TypeError("Expected state to be instance of CustomAttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    if self._is_multi:
      previous_attention_state = state.attention_state
      previous_alignment_history = state.alignment_history
      previous_alignment_sum = state.alignment_sum
    else:
      previous_attention_state = [state.attention_state]
      previous_alignment_history = [state.alignment_history]
      previous_alignment_sum = [state.alignment_sum]

    all_alignments = []
    all_attentions = []
    all_attention_states = []
    maybe_all_histories = []
    all_alignment_sums = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      attention, alignments, next_attention_state = _custom_compute_attention(
          attention_mechanism, cell_output, previous_attention_state[i],
          self._attention_layers[i] if self._attention_layers else None,
          previous_alignment_sum[i])
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_attention_states.append(next_attention_state)
      all_alignments.append(alignments)
      all_attentions.append(attention)
      all_alignment_sums.append(previous_alignment_sum[i] + alignments)
      maybe_all_histories.append(alignment_history)

    attention = array_ops.concat(all_attentions, 1)
    next_state = CustomAttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        attention_state=self._item_or_tuple(all_attention_states),
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(maybe_all_histories),
        alignment_sum=self._item_or_tuple(all_alignment_sums))

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state
