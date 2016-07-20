import numpy

from chainer import cuda
#from chainer.functions.connection import bilinear
from chainer import initializers
from chainer import link
import bilinear_dn_function


class Bilinear(link.Link):

    """Bilinear layer that performs tensor multiplication.

    Bilinear is a primitive link that wraps the
    :func:`~chainer.functions.bilinear` functions. It holds parameters ``W``,
    ``V1``, ``V2``, and ``b`` corresponding to the arguments of
    :func:`~chainer.functions.bilinear`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, parameters ``V1``, ``V2``, and ``b`` are
            omitted.
        initialW (3-D numpy array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (tuple): Initial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.
            May also be a tuple of callables that take ``numpy.ndarray`` or
            ``cupy.ndarray`` and edit its value.

    .. seealso:: See :func:`chainer.functions.bilinear` for details.

    Attributes:
        W (~chainer.Variable): Bilinear weight parameter.
        V1 (~chainer.Variable): Linear weight parameter for the first argument.
        V2 (~chainer.Variable): Linear weight parameter for the second
            argument.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):
        super(Bilinear, self).__init__(W=(left_size, right_size, out_size))
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        # TODO(Kenta OONO): I do not know appropriate way of
        # initializing weights in tensor network.
        # This initialization is a modification of
        # that of Linear function.

        if isinstance(initialW, (numpy.ndarray, cuda.ndarray)):
            assert initialW.shape == self.W.data.shape
        initializers.init_weight(self.W.data, initialW)

        if not self.nobias:
            self.add_param('V1', (left_size, out_size))
            self.add_param('V2', (right_size, out_size))
            self.add_param('b', out_size)

            if isinstance(initial_bias, tuple):
                V1, V2, b = initial_bias
            elif initial_bias is None:
                #V1 = V2 = None
                V1 = numpy.ones((left_size, out_size), dtype = numpy.float32)
                V2 = numpy.matrix((numpy.identity(right_size)), dtype = numpy.float32)
                b = 0
            else:
                raise ValueError('initial_bias must be tuple or None')

            if isinstance(V1, (numpy.ndarray, cuda.ndarray)):
                assert V1.shape == self.V1.data.shape
            if isinstance(V2, (numpy.ndarray, cuda.ndarray)):
                assert V2.shape == self.V2.data.shape
            if isinstance(b, (numpy.ndarray, cuda.ndarray)):
                assert b.shape == self.b.data.shape
            initializers.init_weight(self.V1.data, V1)
            initializers.init_weight(self.V2.data, V2)
            initializers.init_weight(self.b.data, b)

        else:
            self.V1.data[...] = numpy.ones((left_size, out_size), dtype = numpy.float32)
            self.V2.data[...] = numpy.matrix((numpy.identity(right_size)), dtype = numpy.float32)
            self.b.data.fill(0)

        self.left_size = left_size
        self.right_size = right_size
        self.out_size = out_size


    def __call__(self, e1, e2):
        """
        Applies the bilinear function to inputs and the internal parameters.

        Args:
            e1 (~chainer.Variable): Left input.
            e2 (~chainer.Variable): Right input.

        Returns:
            ~chainer.Variable: Output variable.

        """
        V1_tmp = numpy.ones((self.left_size, self.out_size), dtype = numpy.float32)
        V2_tmp = numpy.matrix((numpy.identity(self.right_size)), dtype = numpy.float32)
        b_tmp = numpy.zeros(self.out_size, dtype = numpy.float32)
        W_tmp = numpy.zeros((self.left_size, self.right_size, self.out_size), dtype = numpy.float32)

        self.V1.data = cuda.to_gpu(V1_tmp)
        self.V2.data = cuda.to_gpu(V2_tmp)
        self.b.data = cuda.to_gpu(b_tmp)
        self.W.data = cuda.to_gpu(W_tmp)
        
        if self.nobias:
            return bilinear_dn_function.bilinear(e1, e2, self.W)
        else:
            return bilinear_dn_function.bilinear(e1, e2, self.W, self.V1, self.V2, self.b)

    def zero_grads(self):
        # Left for backward compatibility
        self.zerograds()
