import tensorflow as tf
# at least this can be imported for tensorflow 1.10
from tensorflow.python.ops.rnn import _reverse_seq
from tensorflow.nn import static_rnn
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs



# @tf_export("nn.static_bidirectional_rnn")
def my_static_bidirectional_rnn(cell_fw,
                                cell_bw,
                                inputs,
                                initial_state_fw=None,
                                initial_state_bw=None,
                                dtype=None,
                                sequence_length=None,
                                scope=None):
    """Creates a bidirectional recurrent neural network.
    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.
    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size], or a nested tuple of such elements.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs is a length `T` list of outputs (one for each input), which
          are depth-concatenated forward and backward outputs.
        output_state_fw is the final state of the forward rnn.
        output_state_bw is the final state of the backward rnn.
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is None or an empty list.
    """

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = static_rnn(
                cell_fw,
                inputs,
                initial_state_fw,
                dtype,
                sequence_length=sequence_length,
                scope=fw_scope)

        # Backward direction
        with vs.variable_scope("bw") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            tmp, output_state_bw = static_rnn(
                cell_bw,
                reversed_inputs,
                initial_state_bw,
                dtype,
                sequence_length=sequence_length,
                scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    flat_output_fw = nest.flatten(output_fw)
    flat_output_bw = nest.flatten(output_bw)

    flat_outputs = tuple(
        tf.concat([fw, bw], 1)
        for fw, bw in zip(flat_output_fw, flat_output_bw))

    outputs = nest.pack_sequence_as(
        structure=output_fw, flat_sequence=flat_outputs)

    #return (outputs, output_state_fw, output_state_bw)
    return (outputs, flat_output_fw, flat_output_bw), (output_state_fw, output_state_bw)
